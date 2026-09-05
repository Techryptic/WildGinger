# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Coverage ledger seeding, proof-of-read & line verification.

Ties discovery + indexing + chunking + attack-surface into the SQLite ledger:
  * seed ``files``, ``functions`` and ``(chunk × attack_class)`` ``cells``;
  * weight cells higher for trust-boundary files (attack surface);
  * support **incremental** mode (skip unchanged files by sha256);
  * evaluate **proof-of-read** from observed tool reads;
  * **programmatically verify** a finding's cited snippet against the real file.

This module is deterministic: the same repo + config seeds identical cells, which
is what makes checkpoint/resume and reproducible CI possible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import chunking, fileio
from .db import Database
from .discovery import DiscoveredFile
from .indexing import Index
from .settings import Settings


@dataclass
class SeedStats:
    files: int = 0
    files_skipped_unchanged: int = 0
    # Files whose sha256 moved since the run being resumed. Their cells are reset to
    # PENDING (see seed_ledger) so edited code is genuinely re-analyzed.
    files_changed_on_resume: int = 0
    # Files that are in the ledger but no longer in discovery (deleted/renamed/now out
    # of scope). Their work rows are dropped — see _prune_missing_files.
    files_pruned: int = 0
    functions: int = 0
    cells: int = 0


# The sentinel attack class that means "open-ended: look for ANYTHING, including
# business-logic flaws, broken invariants and novel chains" (prompts.HUNT_OPEN_PROMPT).
OPEN_ENDED_CLASS = "all"


def resolve_attack_classes(settings) -> list[str]:
    """The per-chunk cell buckets to seed — ``"all"`` INCLUDED when asked for.

    When ``hunting.attack_class_groups`` is set, each GROUP becomes one bucket
    (a comma-joined label like ``"injection,path_traversal,ssrf"``): one cell per
    (chunk, group) instead of one per (chunk, class), so full called-out coverage
    costs (groups)x instead of (classes)x. The label is only a bucket name — the
    hunter's prompt expands it back into per-class guidance, and findings carry
    whichever single class best fits (see stages/hunt._default_category).
    ``hunting.open_ended: true`` adds one open-ended (``"all"``) bucket ALONGSIDE
    the groups — being in group mode used to silently drop that sweep (``all``
    cannot join a group), and with it the pass that finds cross-file chains.

    Otherwise the flat ``hunting.attack_classes`` list means what it says:

    ``"all"`` used to be stripped unconditionally and only reinstated as a fallback when
    stripping emptied the list. Two things followed, both wrong:

      * The shipped config lists 11 classes **plus** ``all`` and comments that "each
        multiplies the work", but the ``all`` entry seeded ZERO cells — so users paid for
        11 buckets while believing they had 12.
      * ``all`` is the only value that can make ``orchestrator._hunt_one`` set
        ``open_ended=True``, so ``hunting.open_ended: true`` (the shipped default) was
        unreachable and ``HUNT_OPEN_PROMPT`` was dead code. The open-ended hunt — the
        thing this harness has over a pattern scanner — never ran. The only way to switch
        it on was ``attack_classes: [all]``, which simultaneously disabled all 11
        directed classes; there was no configuration where both ran.

    Now the list means what it says: list ``all`` and you get an open-ended cell per
    chunk in ADDITION to the directed ones. ``hunting.open_ended: false`` still removes
    it, so there are two consistent ways to turn it off and neither is silent.
    """
    groups = list(getattr(settings.hunting, "attack_class_groups", None) or [])
    if groups:
        labels = [",".join(g) for g in groups]
        # open_ended is orthogonal to grouping: one open-ended cell per chunk
        # ALONGSIDE the directed groups. In group mode this sweep used to vanish
        # silently ("all" cannot join a group), quietly dropping the pass that
        # finds business-logic flaws and cross-file chains.
        if getattr(settings.hunting, "open_ended", True):
            labels.append(OPEN_ENDED_CLASS)
        return labels
    configured = list(getattr(settings.hunting, "attack_classes", []) or [])
    directed = [c for c in configured if c != OPEN_ENDED_CLASS]
    wants_open = (OPEN_ENDED_CLASS in configured
                  and getattr(settings.hunting, "open_ended", True))
    if not directed:
        # Nothing directed was configured: fall back to a purely open-ended sweep rather
        # than seeding no work at all.
        return [OPEN_ENDED_CLASS]
    return directed + ([OPEN_ENDED_CLASS] if wants_open else [])


def seed_ledger(
    db: Database,
    settings: Settings,
    files: list[DiscoveredFile],
    index: Index,
    trust_boundary_files: set[str],
    *,
    resume: bool = False,
    prune_missing: bool = False,
) -> SeedStats:
    """Populate files/functions/cells. Returns counts. Honors incremental mode.

    ``resume`` declares the caller's INTENT, which decides what happens to existing
    cell progress:

      * ``False`` (a fresh scan) — every cell is reset to PENDING with a clean
        attempt budget, so the whole scope is genuinely re-hunted. This must happen
        because findings are only reloaded from the DB under ``--resume``: keeping a
        completed ledger on a fresh run left the two halves of state disagreeing and
        produced a scan that hunted nothing, restored nothing, and reported a clean
        100% — a false pass, the worst way for a security gate to fail.
      * ``True`` (``--resume``) — ANALYZED cells are preserved so their (expensive)
        hunt is not re-paid, and everything unfinished is revived to PENDING with a
        fresh attempt budget.

    It is deliberately an explicit argument rather than something inferred from the
    DB: whether this is a continuation is knowledge only the caller has.

    ``prune_missing`` drops the work rows of files that are in the ledger but NOT in
    ``files`` (see :func:`_prune_missing_files`). Off by default because only the caller
    knows whether ``files`` is the whole scope or a deliberate subset.
    """
    stats = SeedStats()
    attack_classes = resolve_attack_classes(settings)
    chunk_lines = settings.scope.chunk_lines
    overlap = settings.scope.chunk_overlap_lines
    incremental = settings.scope.mode == "incremental"

    for df in files:
        existing = _existing_file(db, df.rel_path)
        if incremental and existing and existing["sha256"] == df.sha256:
            stats.files_skipped_unchanged += 1
            continue

        # A file whose CONTENT changed cannot keep its ANALYZED cells on resume. The
        # upsert below refreshes chunk_start/chunk_end to the NEW geometry while the
        # resume branch preserves state <> 'ANALYZED' — so an edited file ended up with
        # cells marked analyzed that pointed at line ranges nobody had ever looked at,
        # and coverage counted them. `--resume` means "continue the same code"; the
        # config_hash guard enforces that for config but nothing enforced it for source.
        file_changed = bool(resume and existing and existing["sha256"] != df.sha256)

        weight = df.weight + (1.0 if df.rel_path in trust_boundary_files else 0.0)
        file_id = _upsert_file(db, df)

        # Functions (proof-of-read targets).
        symbols = index.functions_in(df.rel_path)
        symbol_ranges = [(s.start_line, s.end_line) for s in symbols]
        _replace_functions(db, file_id,
                           [(s.name, s.start_line, s.end_line) for s in symbols],
                           resume=resume and not file_changed)
        stats.functions += len(symbols)

        # Chunks → cells (chunk × attack_class).
        try:
            text = fileio.read_text_meta(df.path).text
        except (OSError, ValueError):
            continue
        chunks = chunking.chunk_by_symbols(
            text, symbol_ranges, chunk_lines=chunk_lines, overlap_lines=overlap
        )
        # Persist each chunk's REAL line range with the cell. The hunter reads these
        # back instead of re-deriving them, which previously used a different
        # algorithm (fixed windows) and handed most cells the wrong line range.
        n_cells = _replace_cells(db, file_id, chunks, attack_classes, weight,
                                 resume=resume and not file_changed)
        if file_changed:
            stats.files_changed_on_resume += 1
        stats.cells += n_cells
        stats.files += 1

    if prune_missing:
        stats.files_pruned = _prune_missing_files(db, {df.rel_path for df in files})

    return stats


def _prune_missing_files(db: Database, discovered: set[str]) -> int:
    """Drop the WORK rows (cells + functions) of ledger files not in ``discovered``.

    Nothing used to prune a file that had been deleted, renamed, or moved out of scope:
    ``seed_ledger`` only ever iterates the files discovery just found, and
    ``_replace_cells`` deletes per-file_id. A vanished file therefore kept its cells
    forever, with two live consequences on any reused ``out_dir`` (and ``out_dir:
    ./wildginger_output`` is a fixed default, so it IS reused):

      * an unfinished cell (PENDING/SHALLOW/FAILED) stayed claimable, and
        ``hunt.run_hunt_cell`` builds the prompt from the ledger WITHOUT touching the
        file — so a hunter spent a full paid conversation, tool call by tool call,
        discovering that the path does not exist;
      * those cells never reach ANALYZED, so ``coverage_pct`` can never reach 1.0,
        which pins an otherwise-clean scan at exit 2 under
        ``output.fail_on_incomplete_coverage``.

    Neither symptom requires ``--resume``.

    What is deliberately NOT deleted: the ``files`` row and any ``findings`` attached to
    it. The findings table is the audit record, and ``findings.file_id`` has no cascade —
    dropping the parent row would leave findings whose ``file_rel`` comes back NULL from
    ``load_findings``, i.e. a MANGLED identity rather than an absent one. Stale findings
    for deleted code are a separate concern, handled where they are reloaded.

    Guard: deletions are incremental, but scanning a subdirectory (a narrowed ``--path``)
    makes most known files look missing at once. Past a simple majority we refuse and say
    so, rather than silently discarding a full scan's progress.
    """
    def _q(cur):
        cur.execute("SELECT id, path FROM files;")
        return [(r["id"], r["path"]) for r in cur.fetchall()]
    known = db.read(_q)
    if not known:
        return 0
    missing = [fid for (fid, path) in known if path not in discovered]
    if not missing:
        return 0
    if len(missing) * 2 > len(known):
        print(f"⚠ {len(missing)} of {len(known)} ledger file(s) are not in this scan's "
              f"scope — that looks like a narrowed --path or a changed include list, not "
              f"deletions.\n"
              f"  Their cells were LEFT IN PLACE (pruning them would discard a previous "
              f"full scan's progress).\n"
              f"  If those files really are gone, use a fresh --output directory.")
        return 0

    def _w(cur):
        marks = ",".join("?" * len(missing))
        cur.execute(f"DELETE FROM cells WHERE file_id IN ({marks});", tuple(missing))
        cur.execute(f"DELETE FROM functions WHERE file_id IN ({marks});", tuple(missing))
    db.write(_w)
    return len(missing)


# ── DB helpers (all writes go through the single-writer queue) ─────────────────

def _existing_file(db: Database, rel_path: str) -> sqlite3.Row | None:
    def _q(cur):
        cur.execute("SELECT id, sha256 FROM files WHERE path = ?;", (rel_path,))
        return cur.fetchone()
    return db.read(_q)


def _upsert_file(db: Database, df: DiscoveredFile) -> int:
    def _w(cur):
        cur.execute(
            """
            INSERT INTO files(path, sha256, language, category, line_count, last_scanned_mtime)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
                sha256=excluded.sha256, language=excluded.language,
                category=excluded.category, line_count=excluded.line_count,
                last_scanned_mtime=excluded.last_scanned_mtime;
            """,
            (df.rel_path, df.sha256, df.language, df.category, df.line_count, df.mtime),
        )
        cur.execute("SELECT id FROM files WHERE path = ?;", (df.rel_path,))
        return cur.fetchone()["id"]
    return db.write(_w)


def _replace_functions(db: Database, file_id: int, syms: list[tuple[str, int, int]],
                       *, resume: bool = False) -> None:
    """Re-seed the symbol rows for one file.

    On ``--resume`` a symbol that survives the re-index keeps its ``visited`` flag,
    so proof-of-read earned by an earlier run isn't thrown away. On a fresh scan
    everything starts unvisited, matching the fresh-scan cell reset.
    """
    def _w(cur):
        prior: dict[tuple[str, int], int] = {}
        if resume:
            cur.execute("SELECT name, start_line, visited FROM functions "
                        "WHERE file_id = ?;", (file_id,))
            prior = {(r["name"], r["start_line"]): r["visited"]
                     for r in cur.fetchall()}
        cur.execute("DELETE FROM functions WHERE file_id = ?;", (file_id,))
        cur.executemany(
            "INSERT INTO functions(file_id, name, start_line, end_line, visited) "
            "VALUES (?,?,?,?,?);",
            [(file_id, name, s, e, prior.get((name, s), 0)) for (name, s, e) in syms],
        )
    db.write(_w)


def _replace_cells(db: Database, file_id: int, chunks: list,
                   attack_classes: list[str], weight: float,
                   *, resume: bool = False) -> int:
    """Re-seed the (chunk × attack_class) cells for one file.

    ``chunks`` is the list of :class:`chunking.Chunk` produced at seed time; each
    cell stores its chunk's 1-based inclusive ``[start_line, end_line]`` so the
    hunter is prompted with the exact range that was chunked.

    Rows are always upserted (never blanket-deleted) so cell identity is stable and
    only the chunk geometry/weight is refreshed. What differs is the PROGRESS
    columns, per ``resume``:

      * fresh scan — every cell returns to PENDING with a clean slate, so the scope
        is genuinely re-analyzed. A plain ``DELETE`` would achieve this too, but it
        also churned row ids; resetting in place is equivalent and cheaper.
      * ``--resume`` — ANALYZED cells are left completely untouched (their hunt cost
        is not re-paid, which is the whole point of resuming); every unfinished cell
        (SHALLOW / FAILED / a stale IN_PROGRESS lease) is revived to PENDING.

    Either way ``attempts`` is zeroed for the cells this run will work on, which
    makes the retry cap PER-RUN. Letting it accumulate across runs meant a cell that
    exhausted its budget once could never be retried again, in any future run.
    """
    # A file with no chunks (e.g. empty) still gets one cell per attack class so it
    # is represented in the ledger; its range is unknown → NULL.
    spans: list[tuple[int, int | None, int | None]] = (
        [(c.index, c.start_line, c.end_line) for c in chunks] if chunks
        else [(0, None, None)]
    )
    rows = [
        (file_id, ci, start, end, ac, weight)
        for (ci, start, end) in spans
        for ac in attack_classes
    ]
    max_chunk_index = max((ci for (ci, _s, _e) in spans), default=-1)

    def _w(cur):
        # Drop only cells that no longer correspond to a chunk (the file shrank) or
        # to a configured attack class — never the ones we're about to re-seed.
        cur.execute("DELETE FROM cells WHERE file_id = ? AND chunk_index > ?;",
                    (file_id, max_chunk_index))
        if attack_classes:
            placeholders = ",".join("?" * len(attack_classes))
            cur.execute(
                f"DELETE FROM cells WHERE file_id = ? "
                f"AND attack_class NOT IN ({placeholders});",
                (file_id, *attack_classes),
            )
        cur.executemany(
            "INSERT INTO cells(file_id, chunk_index, chunk_start, chunk_end, "
            "attack_class, weight) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(file_id, chunk_index, attack_class) DO UPDATE SET "
            "  chunk_start = excluded.chunk_start, "
            "  chunk_end   = excluded.chunk_end, "
            "  weight      = excluded.weight;",
            rows,
        )
        # Reset the progress columns. On resume, ANALYZED is the one state we keep.
        reset = (
            "UPDATE cells SET state = 'PENDING', attempts = 0, last_error = NULL, "
            "                 lease_owner = NULL, lease_expires = NULL"
        )
        if resume:
            cur.execute(reset + " WHERE file_id = ? AND state <> 'ANALYZED';",
                        (file_id,))
        else:
            cur.execute(reset + ", thoroughness = 0.0 WHERE file_id = ?;", (file_id,))
        return len(rows)
    return db.write(_w)


# ── proof-of-read ──────────────────────────────────────────────────────────────

def mark_functions_visited(db: Database, file_id: int) -> int:
    """Mark functions visited where observed reads cover their line range.

    A function counts as visited if some recorded read for its file overlaps its
    [start_line, end_line]. Returns the number newly marked visited.
    """
    def _w(cur):
        # ONE set-based statement. This used to fetch every unvisited function AND every
        # recorded read for the file, then run a nested PYTHON loop with an individual
        # UPDATE per match — all inside the single writer transaction, once per analyzed
        # cell. `reads` grows monotonically for the whole run and is never pruned, and a
        # file has (chunks x attack_classes) cells, so the cost was quadratic in run
        # length: ~200 functions x ~1000 accumulated reads = 200k comparisons per call,
        # with all N hunters blocked behind the writer for the duration. SQLite does the
        # same join in O(log n) per function via idx_reads_file_lines.
        cur.execute(
            """
            UPDATE functions SET visited = 1
             WHERE file_id = ? AND visited = 0
               AND EXISTS (SELECT 1 FROM reads r
                            WHERE r.file_id = functions.file_id
                              AND r.start_line <= functions.end_line
                              AND r.end_line   >= functions.start_line);
            """,
            (file_id,),
        )
        # `rowcount` read defensively: a real sqlite3.Cursor always has it, but this
        # function is also driven by lightweight duck-typed cursors in tests.
        n = getattr(cur, "rowcount", 0)
        return n if isinstance(n, int) and n > 0 else 0
    return db.write(_w)


# Programmatic line verification

def verify_snippet_at_lines(
    root: Path, file_rel: str, line_start: int, line_end: int, cited_snippet: str
) -> bool:
    """Return True if ``cited_snippet`` actually appears within the cited lines.

    Thin wrapper kept for callers that only need the yes/no. Prefer
    :func:`locate_snippet`, which also reports WHERE the snippet really is.
    """
    return locate_snippet(root, file_rel, line_start, line_end, cited_snippet) is not None


# A citation has to be specific enough to actually pin a location. Without any floor,
# the model chooses both the line range AND the snippet, so quoting one common bare
# word ("code", "return", "data") satisfied the anti-hallucination gate at almost any
# line — letting a fabricated description ride in with a "programmatically verified
# citation" stamp on it.
#
# The test is deliberately narrow: reject only a SHORT, SINGLE, ALL-WORD-CHARACTER
# token. A raw length floor would have been the wrong fix — real sinks are often short
# (`eval(x)`, `os.system(c)`), and dropping those would recreate, in a new place, the
# very false-negative problem this gate's window sizing exists to remove.
_MIN_BARE_WORD_CHARS = 12


def _too_generic(needle: str) -> bool:
    """True if ``needle`` is a lone short word, i.e. it pins nothing."""
    if len(needle) >= _MIN_BARE_WORD_CHARS:
        return False
    # Any punctuation/structure (parens, dots, operators, quotes) or a space makes a
    # citation specific enough to be worth checking.
    return needle.isidentifier() or needle.isalnum()


def locate_snippet(
    root: Path, file_rel: str, line_start: int, line_end: int, cited_snippet: str
) -> tuple[int, int] | None:
    """Verify a citation and return the ACTUAL ``(start_line, end_line)`` it matches.

    LLMs hallucinate line numbers; before committing a finding we confirm the quoted
    code really exists at the claimed location. Whitespace-insensitive (the model often
    reformats indentation).

    Two things this fixes over a fixed ±2-line window:

    * **The window is sized from the SNIPPET.** Because whitespace normalization
      collapses newlines, only the window's height mattered — so a model that quoted
      the vulnerable 6-line block but reported only the line of the sink (extremely
      common, and exactly what the prompt's "verbatim line(s)" invites) had its finding
      silently dropped as a hallucination. The window now covers the cited range plus
      as many lines as the snippet itself spans.
    * **It returns the real location.** The ±2 tolerance ACCEPTED a citation up to two
      lines early but left the wrong line numbers on the finding, and ``hunt`` then
      stamped ``symbol`` from them — so the same bug cited correctly and cited two lines
      early got two different enclosing symbols, hence two different fingerprints: no
      dedupe, double judge cost, two report entries, and unstable baselines. Callers use
      the returned range to correct the finding.

    Returns None when the citation cannot be verified.
    """
    if not cited_snippet or not cited_snippet.strip():
        return None
    needle = _normalize_ws(cited_snippet)
    if _too_generic(needle):
        return None
    try:
        from .paths import resolve_in_jail
        target = resolve_in_jail(root, file_rel)
        ls, le = int(line_start), int(line_end)
        if le < ls:
            # A reversed range collapsed to a single line and rejected everything.
            ls, le = le, ls
        # Height of the quoted block, so a multi-line snippet cited at one line still
        # has somewhere to match. Strip first: a trailing newline would over-count.
        span = cited_snippet.strip().count("\n") + 1
        lo = max(1, ls - 2)
        hi = max(le, ls + span - 1) + 2
        actual = fileio.read_line_range(target, lo, hi)
    except (OSError, ValueError):
        return None
    if needle not in _normalize_ws(actual):
        return None
    # Find which line the match really starts on, so the caller can correct the range.
    lines = actual.splitlines()
    for offset in range(len(lines)):
        if needle in _normalize_ws("\n".join(lines[offset:offset + span])):
            start = lo + offset
            return start, start + span - 1
    # Verified, but we could not pin the offset. Keep the model's own range rather than
    # inventing one.
    return ls, le


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())
