# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""SQLite state store with safe concurrency.

This is the one correctness-critical area under parallelism: with N hunter
threads all touching one ``scan_state.db``, naive writes throw
"database is locked" and can corrupt the coverage guarantee. The strategy:

  * **WAL mode** (set in :func:`fileio.open_sqlite`) so readers never block the
    single writer.
  * **Single-writer queue**: every mutation is a callable submitted to one
    dedicated writer thread. This makes "database is locked" structurally
    impossible — there is only ever one writer. ``write()`` has a finite timeout and
    fails fast if that thread ever dies, rather than hanging the caller forever.
  * **Per-thread read connections**: each thread gets its own connection on first
    use, so N hunters read concurrently (WAL guarantees this is safe). They are all
    tracked so ``close()`` can release them.
  * **Per-cell leases**: a hunter claims a cell with an owner + expiry; a crashed
    hunter's lease expires and the cell becomes reclaimable, so no cell gets
    permanently stuck.
  * **Schema versioning + migrations**: a ``schema_version`` table; the tool
    refuses to run against a newer schema and migrates older ones forward.

The public :class:`Database` exposes ``read(fn)`` and ``write(fn)`` helpers that
hand a cursor to a callable, plus a few high-level convenience methods. Stage
code should prefer the convenience methods so SQL stays in one place.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
import weakref
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

from . import fileio

T = TypeVar("T")

# Bump this whenever the schema changes AND add a matching step to _MIGRATIONS.
# v1 → v2: llm_calls.cache_read_tokens / cache_write_tokens.
# v2 → v3: cells.chunk_start / chunk_end (persisted chunk ranges).
# v3 → v4: findings.symbol / cited_snippet + UNIQUE(dedup_key) for upserts.
# v4 → v5: findings.base_severity / base_confidence (scoring.enrich's idempotency
#          latch — without these columns a resumed finding re-latches its ALREADY
#          decayed severity and drops another tier on every resume).
SCHEMA_VERSION = 5

# How long the single-scan advisory lock stays valid without a renewal. Long enough to
# cover a slow pass, short enough that a killed process cannot lock an out_dir for good.
_RUN_LOCK_TTL_SECONDS = 1800


# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

# NOTE: `runs`, `sessions`, `redactions` and `seeds` used to be created and indexed here
# and were NEVER written by any code path — so `runs.exit_code`/`runs.coverage_pct`, the
# session-resume story and the redaction ledger that module docstrings referred to simply
# did not exist. Empty tables that look like features are worse than no tables: they make
# a reviewer trust the rest of the schema less. Removed; `CREATE TABLE IF NOT EXISTS`
# means an existing DB just keeps its unused tables harmlessly.
_SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER NOT NULL,
        applied_at REAL    NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        path               TEXT UNIQUE NOT NULL,
        sha256             TEXT NOT NULL,
        language           TEXT,
        category           TEXT,
        line_count         INTEGER NOT NULL,
        last_scanned_mtime REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS functions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id    INTEGER NOT NULL REFERENCES files(id),
        name       TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line   INTEGER NOT NULL,
        visited    INTEGER NOT NULL DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cells (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id       INTEGER NOT NULL REFERENCES files(id),
        chunk_index   INTEGER NOT NULL,
        -- The chunk's ACTUAL 1-based inclusive line range, persisted at seed time (v3).
        -- Seeding chunks by symbol boundaries while the hunter re-derived ranges with
        -- fixed windows meant most cells were handed the wrong (or whole-file) range.
        chunk_start   INTEGER,
        chunk_end     INTEGER,
        attack_class  TEXT NOT NULL,
        state         TEXT NOT NULL DEFAULT 'PENDING',
        thoroughness  REAL NOT NULL DEFAULT 0.0,
        weight        REAL NOT NULL DEFAULT 1.0,
        lease_owner   TEXT,
        lease_expires REAL,
        attempts      INTEGER NOT NULL DEFAULT 0,
        last_error    TEXT,
        UNIQUE(file_id, chunk_index, attack_class)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reads (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id   TEXT NOT NULL,
        file_id    INTEGER NOT NULL REFERENCES files(id),
        start_line INTEGER NOT NULL,
        end_line   INTEGER NOT NULL,
        ts         REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS findings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id       INTEGER REFERENCES files(id),
        line_start    INTEGER,
        line_end      INTEGER,
        category      TEXT,
        cwe           TEXT,
        severity      TEXT,
        confidence    REAL,
        votes         INTEGER NOT NULL DEFAULT 1,
        hedged        INTEGER NOT NULL DEFAULT 0,
        reachable     TEXT,
        status        TEXT NOT NULL DEFAULT 'new',
        dedup_key     TEXT,
        -- Enclosing symbol + cited snippet (v4). Both are part of a finding's
        -- reported detail, and `symbol` is half of its fingerprint — without it a
        -- resumed finding would hash differently than the one that was stored.
        symbol        TEXT,
        cited_snippet TEXT,
        -- The model's ORIGINAL severity/confidence, latched by scoring.enrich() on its
        -- first call (v5). enrich() derives from these instead of from the current
        -- values, so it is idempotent. They MUST round-trip: with the latch missing on
        -- reload, enrich() re-latched the already-discounted severity and a genuine
        -- `critical` decayed critical -> high -> medium -> low across resumes,
        -- silently falling below output.severity_threshold. See scoring.enrich.
        base_severity   TEXT,
        base_confidence REAL,
        sarif_rule_id TEXT,
        source        TEXT NOT NULL DEFAULT 'llm',
        title         TEXT,
        description   TEXT,
        remediation   TEXT,
        created_at    REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoint (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        phase       TEXT,
        pass_no     INTEGER NOT NULL DEFAULT 0,
        config_hash TEXT,
        updated_at  REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS finding_evidence (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id      INTEGER NOT NULL REFERENCES findings(id),
        pass_no         INTEGER,
        model           TEXT,
        evidence        TEXT,
        raw_response_ref TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS verifications (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id INTEGER NOT NULL REFERENCES findings(id),
        human_result TEXT NOT NULL,
        notes      TEXT,
        ts         REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_calls (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        role          TEXT,
        model_id      TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        -- Prompt-cache usage (v2). Priced differently from ordinary input tokens,
        -- so the cost roll-up needs them stored per call, not just in the ledger.
        cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd      REAL,
        latency_ms    REAL,
        stop_reason   TEXT,
        status        TEXT,
        ts            REAL
    );
    """,
    # Hot-path indexes.
    "CREATE INDEX IF NOT EXISTS idx_cells_state ON cells(state);",
    # NOTE (measured, deliberately NOT "optimised"): claim_next_cell()'s
    # `ORDER BY weight DESC, attempts ASC, id ASC` is resolved with a TEMP B-TREE over
    # the candidate set on every claim. An index on (weight DESC, attempts, id) removes
    # that sort and makes the EARLY claims dramatically cheaper — but (a) the planner
    # only picks it once ANALYZE has produced sqlite_stat1, and (b) claimed rows stay in
    # the index and must be skipped, so per-claim cost then grows linearly as the queue
    # drains. Full-drain timing at 20k cells: 27.7s as-is vs 20.5s with
    # index+ANALYZE — only 1.35x overall, and the TAIL claims get *slower*
    # (1.11ms -> 1.73ms). Both plans are quadratic overall, so an index is not the fix;
    # batching the claim (one transaction claiming N cells) is. Left out on purpose so
    # nobody "restores" a dead index plus a stale-stats dependency for a 1.35x win.
    # UNIQUE so _persist_findings can upsert on dedup_key (stable row ids across
    # passes → `scan mark` verifications never orphan).
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_findings_dedup ON findings(dedup_key);",
    "CREATE INDEX IF NOT EXISTS idx_reads_file ON reads(file_id);",
    # coverage.mark_functions_visited joins functions against reads by overlapping line
    # range, once per analyzed cell, on the single writer thread. Without the range
    # columns in the index that join re-scans every read ever recorded for the file.
    "CREATE INDEX IF NOT EXISTS idx_reads_file_lines "
    "ON reads(file_id, start_line, end_line);",
    # coverage.mark_functions_visited() filters on (file_id, visited) once per
    # analyzed cell; without this it full-scans `functions` every time (measured
    # 13.7x slower on a 60k-function ledger).
    "CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_id, visited);",
    # _persist_findings() deletes evidence by finding_id for every finding on every
    # pass — O(findings × evidence) table scans without this.
    "CREATE INDEX IF NOT EXISTS idx_evidence_finding ON finding_evidence(finding_id);",
    "CREATE INDEX IF NOT EXISTS idx_verifications_finding ON verifications(finding_id);",
]


# ──────────────────────────────────────────────────────────────────────────────
# Single-writer queue
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _WriteJob:
    fn: Callable[[sqlite3.Cursor], object]
    result: "queue.Queue"
    # Set when write() gives up waiting. The job stays in the queue (there is no way to
    # pull one item out of a Queue), so without this flag the writer executed it ANYWAY,
    # long after the caller had been told the write FAILED. For claim_next_cell that
    # meant a phantom claim: the cell went IN_PROGRESS with attempts+1 and a 600s lease
    # owned by nobody, unclaimable and one attempt poorer, while the caller believed
    # nothing had happened.
    cancelled: bool = False


def _like_prefix(value: str) -> str:
    """Escape LIKE wildcards so a path containing ``_`` or ``%`` matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _close_quietly(conn: sqlite3.Connection) -> None:
    """Close a connection, ignoring "already closed"/"busy" noise."""
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


class _Sentinel:
    pass


_STOP = _Sentinel()


# ──────────────────────────────────────────────────────────────────────────────
# Migrations
# ──────────────────────────────────────────────────────────────────────────────

def _table_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    """Existing column names for ``table`` (empty set if the table is absent)."""
    cur.execute(f"PRAGMA table_info({table});")
    return {row["name"] for row in cur.fetchall()}


def _add_column(cur: sqlite3.Cursor, table: str, column: str, decl: str) -> bool:
    """Idempotently ``ALTER TABLE ... ADD COLUMN``. True if it was actually added.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we check PRAGMA first; this makes
    the migration safe to re-run (and safe on a DB created fresh at the new version).
    """
    if column in _table_columns(cur, table):
        return False
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl};")
    return True


def _migrate_v2(cur: sqlite3.Cursor) -> None:
    """v1 → v2: record prompt-cache token counts per LLM call.

    Bedrock/OpenAI report cache read/write tokens separately from ordinary input
    tokens, and they are priced differently. Storing them per call is what lets the
    cost roll-up (and the kill-switch) stay accurate once caching is enabled.
    """
    _add_column(cur, "llm_calls", "cache_read_tokens", "INTEGER NOT NULL DEFAULT 0")
    _add_column(cur, "llm_calls", "cache_write_tokens", "INTEGER NOT NULL DEFAULT 0")


def _migrate_v3(cur: sqlite3.Cursor) -> None:
    """v2 → v3: persist each cell's real chunk line range.

    Chunk ranges used to be re-derived at hunt time with a DIFFERENT algorithm than
    the one used at seed time (fixed windows vs. symbol boundaries), so most cells
    were told to analyze the wrong lines. Storing the range at seed time makes the
    ledger the single source of truth.

    Existing rows get NULL, which ``hunt._cell_location`` treats as "unknown" and
    falls back to re-deriving — so an in-flight resume still works, just without the
    fix until those cells are re-seeded.
    """
    _add_column(cur, "cells", "chunk_start", "INTEGER")
    _add_column(cur, "cells", "chunk_end", "INTEGER")


def _migrate_v4(cur: sqlite3.Cursor) -> None:
    """v3 → v4: findings.symbol / findings.cited_snippet + a dedup_key upsert index.

    ``symbol`` is half of a finding's fingerprint, so it MUST round-trip through the
    DB or a resumed finding would hash differently than the one that was persisted.
    ``cited_snippet`` is the evidence a reviewer needs and was previously dropped.

    The UNIQUE index turns ``_persist_findings`` into a real upsert, which keeps
    finding row ids stable across passes so ``verifications`` rows never orphan.
    """
    _add_column(cur, "findings", "symbol", "TEXT")
    _add_column(cur, "findings", "cited_snippet", "TEXT")
    # Collapse any pre-existing duplicate dedup_keys, keeping the lowest id, so the
    # UNIQUE index below can be created on legacy data.
    cur.execute(
        "DELETE FROM findings WHERE id NOT IN "
        "(SELECT MIN(id) FROM findings WHERE dedup_key IS NOT NULL GROUP BY dedup_key) "
        "AND dedup_key IS NOT NULL;"
    )
    cur.execute("DROP INDEX IF EXISTS idx_findings_dedup;")
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_findings_dedup "
        "ON findings(dedup_key);"
    )


def _migrate_v5(cur: sqlite3.Cursor) -> None:
    """v4 → v5: findings.base_severity / base_confidence (enrich's idempotency latch).

    ``scoring.enrich`` recomputes severity from a REACHABILITY multiplier, so it must
    derive from the model's original values, not from its own previous output. It
    latches them onto the Finding the first time it runs — but the latch lived only in
    memory, so a ``--resume`` reloaded the already-discounted severity, latched THAT,
    and discounted it again: critical -> high -> medium -> low over three resumes,
    quietly dropping real criticals below the reporting threshold.

    Existing rows get NULL, which is exactly "not yet latched": the next ``enrich``
    latches whatever is stored. That is the pre-existing behaviour for old rows and
    cannot be reconstructed retroactively — but from this version on it is stable.
    """
    _add_column(cur, "findings", "base_severity", "TEXT")
    _add_column(cur, "findings", "base_confidence", "REAL")


# version → migration step that upgrades the schema TO that version.
_MIGRATIONS: dict[int, Callable[[sqlite3.Cursor], None]] = {
    2: _migrate_v2,
    3: _migrate_v3,
    4: _migrate_v4,
    5: _migrate_v5,
}


class Database:
    """Thread-safe SQLite store. One writer thread; readers query directly."""

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 30_000,
                 write_timeout: float = 120.0, close_timeout: float = 10.0):
        self.db_path = Path(db_path)
        self._busy_timeout_ms = busy_timeout_ms
        # How long write() waits for the writer thread before raising instead of
        # hanging. Generous (a big transaction under contention is legitimately slow)
        # but finite, so a wedged writer surfaces as an error not a frozen scan.
        self._write_timeout = write_timeout
        self._close_timeout = close_timeout
        # Dedicated writer connection, used only by the writer thread.
        self._wconn = fileio.open_sqlite(self.db_path, busy_timeout_ms=busy_timeout_ms)
        # Reads use a PER-THREAD connection (WAL makes concurrent readers safe). One
        # shared connection behind a global lock serialized all N hunters' reads.
        self._local = threading.local()
        # Keyed by thread ident, and RELEASED when the owning thread exits. A plain
        # append-only list kept every connection alive after its thread had died, and a
        # NEW ThreadPoolExecutor is built for every pass and every validation phase — so
        # a 10-pass run at hunters=12 retained ~240 SQLite connections, each holding 3
        # file descriptors and a ~2 MB page cache. On a CI container with the common
        # 1024-fd limit that is "OSError: Too many open files" mid-pass, which loses every
        # in-flight cell's paid work. (sqlite3.Connection is not weak-referenceable, so
        # the finalizer hangs off the Thread object instead.)
        self._read_conns: dict[int, sqlite3.Connection] = {}
        self._conns_lock = threading.Lock()
        self._wqueue: "queue.Queue" = queue.Queue()
        # Set if the writer thread ever dies, so write() can fail fast with the cause
        # instead of blocking forever on a reply that will never come.
        self._writer_died: BaseException | None = None
        # Guards the (closed? dead? alive?) checks and the enqueue as ONE step. Without
        # it a caller could pass all three checks, be descheduled, and then enqueue its
        # job BEHIND the _STOP sentinel that close() had meanwhile pushed — where nobody
        # would ever answer it, so the caller blocked for the full write_timeout (120s in
        # production) and then reported "timed out" for what was really "closed", losing
        # the in-flight ANALYZED transitions in the process.
        self._state_lock = threading.Lock()
        self._writer = threading.Thread(target=self._writer_loop, name="db-writer", daemon=True)
        self._writer.start()
        self._closed = False

    # ── writer thread ─────────────────────────────────────────────────────────
    def _writer_loop(self) -> None:
        """Serve write jobs until stopped.

        Every job MUST get a reply. If this thread ever dies with a job unanswered,
        the caller blocks forever on ``result.get()`` — so the whole body is wrapped
        and ``_writer_died`` is recorded to unblock everyone still waiting.
        """
        stopped_cleanly = False
        try:
            while True:
                job = self._wqueue.get()
                if isinstance(job, _Sentinel):
                    self._wqueue.task_done()
                    stopped_cleanly = True
                    return
                if not isinstance(job, _WriteJob):
                    self._wqueue.task_done()
                    continue
                self._run_job(job)
        except BaseException as exc:  # noqa: BLE001 — the thread is going down
            # Record the cause, then fail every waiter instead of hanging them.
            self._writer_died = exc
            raise
        finally:
            # Drain on EVERY exit path, including the clean _STOP one. Returning without
            # draining orphaned any job that raced close() into the queue behind the
            # sentinel; those callers then waited out the full write_timeout for a reply
            # that was never coming.
            self._drain_pending_jobs(
                self._writer_died or RuntimeError(
                    "database is closed" if stopped_cleanly
                    else "database writer thread stopped unexpectedly"),
                died=self._writer_died is not None)

    def _run_job(self, job: "_WriteJob") -> None:
        """Execute one write job in its own transaction; always answer the caller."""
        if job.cancelled:
            # The caller already gave up and raised. Applying this now would be a write
            # nobody is expecting — see _WriteJob.cancelled.
            self._wqueue.task_done()
            return
        cur = None
        try:
            # cursor() is INSIDE the try: if the connection is broken this raises,
            # and doing it outside meant the exception escaped _writer_loop with the
            # job unanswered — the classic "write() hangs forever" failure.
            cur = self._wconn.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            out = job.fn(cur)
            self._wconn.commit()
            job.result.put(("ok", out))
        except BaseException as exc:  # noqa: BLE001 — surface to caller
            try:
                self._wconn.rollback()
            except Exception:  # noqa: BLE001 — rollback on a dead conn is moot
                pass
            job.result.put(("err", exc))
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    pass
            self._wqueue.task_done()

    def _drain_pending_jobs(self, exc: BaseException, *, died: bool = True) -> None:
        """Fail every queued job so no caller waits on a stopped writer thread.

        ``died`` distinguishes a crash from an orderly shutdown, so the error a caller
        sees names the real cause instead of always claiming the thread "died".
        """
        reason = (f"database writer thread died: {type(exc).__name__}: {exc}" if died
                  else f"database writer stopped before this write ran: {exc}")
        while True:
            try:
                job = self._wqueue.get_nowait()
            except queue.Empty:
                return
            if isinstance(job, _WriteJob) and not job.cancelled:
                job.result.put(("err", RuntimeError(reason)))
            self._wqueue.task_done()

    # ── public API ──────────────────────────────────────────────────────────
    def write(self, fn: Callable[[sqlite3.Cursor], T], *,
              timeout: float | None = None) -> T:
        """Run ``fn(cursor)`` on the writer thread inside a transaction.

        Blocks until the write completes and returns its result. All mutations go
        through here, guaranteeing a single writer at all times.

        Raises ``RuntimeError`` rather than blocking forever if the writer thread has
        died or stops responding — a hung hunter thread is much harder to diagnose
        than an exception, and every write here is on a scan's critical path.
        """
        result: "queue.Queue" = queue.Queue(maxsize=1)
        job = _WriteJob(fn=fn, result=result)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("database is closed")
            if self._writer_died is not None:
                raise RuntimeError(
                    f"database writer thread is dead: {self._writer_died}")
            if not self._writer.is_alive():
                raise RuntimeError("database writer thread is not running")
            self._wqueue.put(job)
        try:
            status, payload = result.get(
                timeout=self._write_timeout if timeout is None else timeout)
        except queue.Empty as exc:
            # Cancel before raising: the job is still in the queue and would otherwise
            # be applied behind our back.
            job.cancelled = True
            raise RuntimeError(
                f"database write timed out after "
                f"{self._write_timeout if timeout is None else timeout}s "
                f"(writer alive={self._writer.is_alive()})") from exc
        if status == "err":
            raise payload  # type: ignore[misc]
        return payload  # type: ignore[return-value]

    def read(self, fn: Callable[[sqlite3.Cursor], T]) -> T:
        """Run ``fn(cursor)`` against a per-thread read connection (WAL-safe).

        Each thread gets its OWN connection, created on first use. Sharing one
        connection behind a global lock serialized every hunter's reads — the exact
        contention the module docstring claims not to have.
        """
        if self._closed:
            raise RuntimeError("database is closed")
        cur = self._thread_conn().cursor()
        try:
            return fn(cur)
        finally:
            cur.close()

    def _thread_conn(self) -> sqlite3.Connection:
        """The calling thread's read connection, opened on first use.

        Closed automatically when the owning thread exits (see ``_read_conns``), so a
        long run that churns thread pools does not accumulate connections.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = fileio.open_sqlite(self.db_path,
                                      busy_timeout_ms=self._busy_timeout_ms)
            self._local.conn = conn
            ident = threading.get_ident()
            with self._conns_lock:
                self._read_conns[ident] = conn
            # Tie the connection's lifetime to this thread: weakref.finalize on the
            # Thread object fires once the thread has exited and been collected.
            self._local.closer = weakref.finalize(
                threading.current_thread(), self._release_thread_conn, ident)
        return conn

    def _release_thread_conn(self, ident: int) -> None:
        """Close and forget the connection owned by a thread that has exited."""
        with self._conns_lock:
            conn = self._read_conns.pop(ident, None)
        if conn is not None:
            _close_quietly(conn)

    def close(self) -> None:
        """Stop the writer and close every connection.

        Ordering matters: we must NOT close the writer's connection while it may
        still be mid-transaction, or SQLite can raise from another thread and the
        final writes are lost.
        """
        with self._state_lock:
            if self._closed:
                return
            # Under the lock: any write() that has already passed its checks holds the
            # lock, so it enqueues BEFORE the sentinel and is served normally.
            self._closed = True
            self._wqueue.put(_STOP)
        self._writer.join(timeout=self._close_timeout)
        if self._writer.is_alive():
            # The writer is wedged (e.g. a pathological lock wait). Leave its
            # connection alone — closing it underneath a live thread is undefined —
            # and say so instead of corrupting state silently. The thread is a daemon,
            # so it won't keep the process alive.
            print(f"⚠ database writer did not stop within {self._close_timeout}s; "
                  f"leaving its connection open to avoid a mid-transaction close. "
                  f"State on disk is still consistent (WAL).")
        else:
            try:
                self._wconn.close()
            except Exception:  # noqa: BLE001
                pass
        with self._conns_lock:
            # Snapshot first: a finalizer firing mid-iteration would mutate the dict.
            for conn in list(self._read_conns.values()):
                # NOTE: this closes connections owned by OTHER threads. WAL keeps the
                # FILE consistent either way, but a worker still inside read() can see
                # sqlite3.ProgrammingError. read() checks `_closed` first, so the window
                # is small and only reachable if a worker outlives close() — which now
                # only happens on an abandoned pool.
                _close_quietly(conn)
            self._read_conns.clear()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── schema / migrations ───────────────────────────────────────────────────
    def acquire_run_lock(self, holder: str) -> tuple[bool, str]:
        """Take the single-scan advisory lock. Returns ``(acquired, current_holder)``.

        Two scans sharing one ``out_dir`` corrupt each other's accounting in ways no
        per-row guard can fix: the second run's ``seed_ledger`` resets the first run's
        live IN_PROGRESS cells to PENDING and then re-hunts them, so the same cells are
        paid for twice. The lease owner guard prevents the CLOBBER, not the double spend —
        only a run-level lock does that. Advisory on purpose: a stale lock from a killed
        process must never make an out_dir permanently unusable, so it expires.
        """
        def _lock(cur: sqlite3.Cursor) -> tuple[bool, str]:
            cur.execute("CREATE TABLE IF NOT EXISTS run_lock ("
                        "id INTEGER PRIMARY KEY CHECK (id = 1), "
                        "holder TEXT, expires REAL);")
            now = time.time()
            cur.execute("SELECT holder, expires FROM run_lock WHERE id = 1;")
            row = cur.fetchone()
            if row is not None and row["holder"] != holder and (row["expires"] or 0) > now:
                return False, str(row["holder"])
            cur.execute(
                "INSERT INTO run_lock(id, holder, expires) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET holder=excluded.holder, "
                "expires=excluded.expires;",
                (holder, now + _RUN_LOCK_TTL_SECONDS),
            )
            return True, holder
        return self.write(_lock)

    def renew_run_lock(self, holder: str) -> None:
        """Push the advisory lock's expiry forward (called at each pass boundary)."""
        def _renew(cur: sqlite3.Cursor) -> None:
            cur.execute("UPDATE run_lock SET expires = ? WHERE id = 1 AND holder = ?;",
                        (time.time() + _RUN_LOCK_TTL_SECONDS, holder))
        try:
            self.write(_renew)
        except Exception:  # noqa: BLE001 — advisory only
            pass

    def release_run_lock(self, holder: str) -> None:
        def _rel(cur: sqlite3.Cursor) -> None:
            cur.execute("DELETE FROM run_lock WHERE id = 1 AND holder = ?;", (holder,))
        try:
            self.write(_rel)
        except Exception:  # noqa: BLE001 — advisory only
            pass

    def initialize(self) -> None:
        """Create the schema if needed and run migrations forward.

        Ordering matters: MIGRATIONS RUN FIRST on an existing DB. Some statements in
        ``_SCHEMA_STATEMENTS`` (notably the UNIQUE index on ``findings.dedup_key``)
        cannot be applied to legacy data until a migration has cleaned it up, so
        creating the schema first would abort the upgrade with an IntegrityError.
        """
        def _init(cur: sqlite3.Cursor) -> None:
            # Read the version BEFORE touching the schema. On a brand-new DB the
            # table doesn't exist yet, which is itself the "fresh install" signal.
            current = self._read_version(cur)

            if current is not None and current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema v{current} is newer than this tool (v{SCHEMA_VERSION}); "
                    "upgrade the harness or use a fresh state DB."
                )
            if current is not None and current < SCHEMA_VERSION:
                # Bring existing tables up to date first (adds columns, cleans data),
                # so the CREATE ... statements below can apply cleanly.
                self._migrate(cur, current, SCHEMA_VERSION)

            for stmt in _SCHEMA_STATEMENTS:
                cur.execute(stmt)

            if current is None:
                cur.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?);",
                    (SCHEMA_VERSION, time.time()),
                )
        self.write(_init)

    @staticmethod
    def _read_version(cur: sqlite3.Cursor) -> Optional[int]:
        """Highest recorded schema version, or None for a brand-new database."""
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';")
        if cur.fetchone() is None:
            return None
        cur.execute("SELECT MAX(version) AS v FROM schema_version;")
        row = cur.fetchone()
        return row["v"] if row and row["v"] is not None else None

    def _migrate(self, cur: sqlite3.Cursor, frm: int, to: int) -> None:
        """Apply ordered migrations frm→to, then record the new version.

        Each step is keyed on the version it upgrades *to* and must be idempotent —
        the whole ``initialize`` runs inside one transaction, so a failure rolls back
        and leaves ``schema_version`` untouched (we retry cleanly next launch).

        NOTE: the schema is created with ``CREATE TABLE IF NOT EXISTS``, which means a
        table added to ``_SCHEMA_STATEMENTS`` appears automatically on old DBs, but a
        *column* added to an existing table does NOT. New columns therefore need a
        real ``ALTER TABLE`` step here.
        """
        for version in range(frm + 1, to + 1):
            step = _MIGRATIONS.get(version)
            if step is None:
                continue
            step(cur)
        cur.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?);",
            (to, time.time()),
        )

    # ── cell leasing ───────────────────────────────────────────
    def claim_next_cell(self, owner: str, *, lease_seconds: int = 600,
                        max_attempts: int = 10) -> Optional[sqlite3.Row]:
        """Atomically claim the highest-weight PENDING (or expired-lease) cell.

        Returns the claimed cell row, or None if no work is available. The claim
        + state flip happen inside one writer transaction so two hunters can never
        grab the same cell.

        ``FAILED`` cells ARE reclaimable, up to ``max_attempts`` total tries. They
        used to be excluded entirely, so a single transient error (one throttle
        storm, one read timeout) stranded that cell forever: coverage could never
        reach 100%, which pinned the exit code at "coverage incomplete" for the rest
        of the repo's life. ``attempts`` — previously written but only ever used as
        an ORDER BY tiebreak — now bounds the retries so a genuinely poisonous cell
        can't be re-attempted without limit. Callers should pass
        ``max_attempts=budget.max_passes`` so the per-cell cap matches the run's
        pass budget (one claim per cell per pass), which keeps the existing SHALLOW
        re-hunt behavior intact.
        """
        now = time.time()
        expires = now + lease_seconds

        def _claim(cur: sqlite3.Cursor) -> Optional[sqlite3.Row]:
            cur.execute(
                """
                SELECT id FROM cells
                 WHERE (
                        state = 'PENDING'
                     OR (state = 'IN_PROGRESS' AND lease_expires IS NOT NULL AND lease_expires < ?)
                     OR state = 'SHALLOW'
                     OR state = 'FAILED'
                       )
                   AND attempts < ?
              ORDER BY weight DESC, attempts ASC, id ASC
                 LIMIT 1;
                """,
                (now, max_attempts),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cell_id = row["id"]
            cur.execute(
                """
                UPDATE cells
                   SET state = 'IN_PROGRESS', lease_owner = ?, lease_expires = ?,
                       attempts = attempts + 1
                 WHERE id = ?;
                """,
                (owner, expires, cell_id),
            )
            cur.execute("SELECT * FROM cells WHERE id = ?;", (cell_id,))
            return cur.fetchone()

        return self.write(_claim)

    def renew_lease(self, cell_id: int, owner: str, *, lease_seconds: int = 600) -> bool:
        """Extend a live claim. Returns False if we no longer own it.

        Leases were written exactly twice — at claim and at the terminal transition — with
        no renewal anywhere, while one hunt is allowed ``max_turns`` turns and a single
        turn can legitimately block for minutes. A hunt therefore routinely outlived the
        600s lease it was working under, at which point any other claimant could take the
        cell and pay for the same work again. Called from the agent's per-turn checkpoint.
        """
        def _renew(cur: sqlite3.Cursor) -> bool:
            cur.execute(
                "UPDATE cells SET lease_expires = ? "
                " WHERE id = ? AND lease_owner = ? AND state = 'IN_PROGRESS';",
                (time.time() + lease_seconds, cell_id, owner),
            )
            return bool(cur.rowcount)
        return bool(self.write(_renew))

    def set_cell_state(
        self,
        cell_id: int,
        state: str,
        *,
        thoroughness: float | None = None,
        last_error: str | None = None,
        owner: str | None = None,
        also: Callable[[sqlite3.Cursor] , None] | None = None,
    ) -> bool:
        """Transition a cell to a terminal/intermediate state, clearing its lease.

        ``also`` runs in the SAME writer transaction as the state change, so the two
        either both commit or both roll back. That atomicity is load-bearing for
        ANALYZED: cells were committed one-by-one as each hunter finished, but their
        findings were only written at the END of the pass, so an ungraceful death
        (SIGKILL/OOM/power loss) in between left durable ANALYZED cells whose findings
        never reached disk. ``seed_ledger(resume=True)`` preserves ANALYZED, so those
        cells were never re-hunted and the run reported 100% coverage with a hole in
        it. Pass the finding writer here and that window closes.

        ``owner`` makes the transition CONDITIONAL on still holding the lease. Returns
        True if the row was updated, False if the claim had been taken over. Without it
        the UPDATE was ``WHERE id = ?`` alone, so a hunter whose lease had expired and
        been reclaimed still stamped its result over the new owner's live claim AND
        cleared that owner's lease — both hunters paid, one result was discarded, and
        coverage counted the cell once. Omit ``owner`` for the unconditional legacy
        behaviour (used by the salvage paths, which run after the pool has joined).
        Returns True when ``owner`` is None.
        """
        def _set(cur: sqlite3.Cursor) -> bool:
            if also is not None:
                # BEFORE the flip: if writing the findings fails, the whole job rolls
                # back and the cell keeps its lease, so the work is retried rather
                # than being marked done with nothing to show for it.
                also(cur)
            if owner is None:
                cur.execute(
                    """
                    UPDATE cells
                       SET state = ?,
                           thoroughness = COALESCE(?, thoroughness),
                           last_error = ?,
                           lease_owner = NULL,
                           lease_expires = NULL
                     WHERE id = ?;
                    """,
                    (state, thoroughness, last_error, cell_id),
                )
                return True
            cur.execute(
                """
                UPDATE cells
                   SET state = ?,
                       thoroughness = COALESCE(?, thoroughness),
                       last_error = ?,
                       lease_owner = NULL,
                       lease_expires = NULL
                 WHERE id = ? AND lease_owner = ?;
                """,
                (state, thoroughness, last_error, cell_id, owner),
            )
            return bool(cur.rowcount)
        return bool(self.write(_set))

    def boost_cell_weights(self, path_prefixes: list[str], delta: float = 1.0) -> int:
        """Raise ``weight`` for cells whose file path starts with any given prefix.

        This is how the recon stage's output finally reaches the scan. ``run_recon``
        builds a tool-using agent, spends real tokens partitioning the attack surface,
        and its ``focus_areas`` were then used for NOTHING but a count in an audit event —
        while ``discovery._priority_weight``'s docstring promised "Recon will further
        raise weights for tagged trust boundaries". Cells are claimed in ``weight DESC``
        order, so a boost genuinely changes what gets hunted first.
        """
        if not path_prefixes:
            return 0

        def _boost(cur: sqlite3.Cursor) -> int:
            total = 0
            for prefix in path_prefixes:
                if not prefix:
                    continue
                cur.execute(
                    "UPDATE cells SET weight = weight + ? WHERE file_id IN "
                    "(SELECT id FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\');",
                    (delta, prefix, _like_prefix(prefix) + "%"),
                )
                total += max(0, cur.rowcount or 0)
            return total
        return self.write(_boost)

    # ── coverage ──────────────────────────────────────────────────────────────
    def coverage_pct(self) -> float:
        """ANALYZED cells / total cells, as a fraction in [0, 1].

        Zero cells returns **0.0**, not 1.0. An empty ledger means we analyzed
        nothing, and reporting that as "100% covered" made a scan that silently
        seeded no work exit 0 with ``coverage_complete: true`` — false assurance in
        CI, the worst possible direction for a security gate. "Nothing to scan" is
        detected earlier and separately (discovery finding no files).
        """
        def _cov(cur: sqlite3.Cursor) -> float:
            cur.execute("SELECT COUNT(*) AS n FROM cells;")
            total = cur.fetchone()["n"]
            if not total:
                return 0.0
            cur.execute("SELECT COUNT(*) AS n FROM cells WHERE state = 'ANALYZED';")
            done = cur.fetchone()["n"]
            return done / total
        return self.read(_cov)

    def thin_cell_count(self, thoroughness_min: float) -> int:
        """ANALYZED cells whose thoroughness never reached the bar.

        These were settled by the shallow-plateau rule: the model read the code and
        reported nothing on repeated attempts. They legitimately count as covered — the
        code WAS read — but reporting "100% coverage" without saying how many got only a
        thin look would overstate the result, so the summary and report name them.
        """
        def _q(cur: sqlite3.Cursor) -> int:
            cur.execute(
                "SELECT COUNT(*) AS n FROM cells "
                " WHERE state = 'ANALYZED' AND COALESCE(thoroughness, 0) < ?;",
                (float(thoroughness_min),),
            )
            row = cur.fetchone()
            return int(row["n"]) if row else 0
        try:
            return self.read(_q)
        except Exception:  # noqa: BLE001 — advisory only, never break reporting
            return 0

    # ── checkpoint (crash resume) ───────────────────────────────────────────
    def set_checkpoint(self, phase: str, *, pass_no: int = 0,
                       config_hash: str | None = None) -> None:
        """Record the current scan phase so a crashed run can resume from it."""
        def _set(cur: sqlite3.Cursor) -> None:
            cur.execute(
                "INSERT INTO checkpoint(id, phase, pass_no, config_hash, updated_at) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET phase=excluded.phase, "
                "pass_no=excluded.pass_no, config_hash=excluded.config_hash, "
                "updated_at=excluded.updated_at;",
                (phase, pass_no, config_hash, time.time()),
            )
        self.write(_set)

    def get_checkpoint(self) -> Optional[sqlite3.Row]:
        def _q(cur: sqlite3.Cursor):
            cur.execute("SELECT phase, pass_no, config_hash, updated_at "
                        "FROM checkpoint WHERE id = 1;")
            return cur.fetchone()
        return self.read(_q)

    def load_findings(self) -> list[dict]:
        """Load persisted findings (as dicts) for resume. Joins the file path back."""
        def _q(cur: sqlite3.Cursor):
            cur.execute(
                "SELECT f.*, files.path AS file_rel FROM findings f "
                "LEFT JOIN files ON files.id = f.file_id;")
            return [dict(r) for r in cur.fetchall()]
        return self.read(_q)

    def record_read(self, agent_id: str, file_id: int, start_line: int, end_line: int) -> None:
        """Log an observed tool read (proof-of-read)."""
        def _ins(cur: sqlite3.Cursor) -> None:
            cur.execute(
                "INSERT INTO reads(agent_id, file_id, start_line, end_line, ts) "
                "VALUES (?, ?, ?, ?, ?);",
                (agent_id, file_id, start_line, end_line, time.time()),
            )
        self.write(_ins)
