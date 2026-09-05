# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform file + SQLite I/O helpers.

Centralizes the portability footguns (encoding, line endings, byte hashing) and
the SQLite connection setup (WAL mode + busy_timeout) so the rest of the codebase
never has to think about Windows-vs-Unix differences.

Rules enforced here:
  * Always read text as UTF-8 with ``errors="replace"`` (Windows defaults to
    cp1252 otherwise — a classic source of decode crashes).
  * Hash the *raw bytes* of a file, never the decoded text, so the coverage
    ledger's sha256 is identical on CRLF (Windows) and LF (Unix) checkouts.
  * Count lines with universal-newline semantics so line numbers match what the
    model sees regardless of CRLF/LF.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

# Files larger than this are treated as binary/oversized by default callers.
DEFAULT_MAX_BYTES = 5_000_000

# Heuristic: a NUL byte in the first chunk almost always means binary.
_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class FileText:
    """Decoded text plus metadata needed by chunking / coverage."""
    text: str
    line_count: int
    sha256: str
    size_bytes: int


def read_bytes(path: Path, *, max_bytes: int | None = DEFAULT_MAX_BYTES) -> bytes:
    """Read raw bytes, optionally guarding against oversized files.

    The size check happens BEFORE the read. It used to run afterwards, so refusing a
    file still required loading all of it into memory first — one large vendored
    artifact (a bundled .js, a checked-in model weight) could exhaust memory even
    though the guard was about to reject it. ``stat`` is also what the caller wants
    semantically: "don't touch this file", not "read it then throw it away".
    """
    if max_bytes is not None:
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        if size is not None and size > max_bytes:
            raise ValueError(f"file exceeds max_bytes ({size} > {max_bytes}): {path}")
    data = path.read_bytes()
    # Belt-and-braces: a file can grow between stat() and read(), and st_size is not
    # meaningful for some virtual filesystems.
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError(f"file exceeds max_bytes ({len(data)} > {max_bytes}): {path}")
    return data


def sha256_bytes(data: bytes) -> str:
    """Stable content hash of raw bytes (CRLF/LF agnostic by construction)."""
    return hashlib.sha256(data).hexdigest()


def is_probably_binary(path: Path) -> bool:
    """Cheap binary sniff: NUL byte in the first few KB."""
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False


def decode_text(data: bytes) -> str:
    """Decode raw bytes to text using the project-wide UTF-8 policy."""
    return data.decode("utf-8", errors="replace")


def count_lines(text: str) -> int:
    """Universal-newline line count.

    ``str.splitlines()`` handles ``\\n``, ``\\r\\n`` and ``\\r`` uniformly, so the
    count matches what the model sees after we normalize line endings. An empty
    file counts as 0 lines; a non-empty file with no trailing newline still
    counts its final line.
    """
    if not text:
        return 0
    return len(text.splitlines())


def read_text_meta(path: Path, *, max_bytes: int | None = DEFAULT_MAX_BYTES) -> FileText:
    """Read a file once and return decoded text + line count + byte hash + size.

    The hash and size come from the *raw bytes*; the text/line-count come from the
    UTF-8 decode. One read, all the metadata chunking/coverage need.
    """
    data = read_bytes(path, max_bytes=max_bytes)
    text = decode_text(data)
    return FileText(
        text=text,
        line_count=count_lines(text),
        sha256=sha256_bytes(data),
        size_bytes=len(data),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Decoded-line cache
# ──────────────────────────────────────────────────────────────────────────────
# Every ranged read used to re-read, re-decode and re-splitlines the ENTIRE file, and
# `search_code` re-walked and re-decoded the whole tree on every call — so with N hunters
# each issuing several tool calls the same bytes were read from disk dozens of times per
# cell. Slow tool turns extend cell wall-time, which is what pushes a hunt past its lease.
#
# Keyed on (path, mtime_ns, size) so an edited file is never served stale: a changed file
# gets a different key and is re-read. Bounded LRU, so a huge repo cannot grow it without
# limit. Not thread-safe by design — dict operations are atomic under the GIL, and the
# worst case of a race is a redundant read, never wrong content.
_LINE_CACHE: "OrderedDict[tuple[str, int, int], list[str]]" = OrderedDict()
_LINE_CACHE_MAX = 256


def cached_lines(path: Path, *, max_bytes: int | None = DEFAULT_MAX_BYTES) -> list[str]:
    """Decoded lines of ``path``, cached on (path, mtime, size)."""
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        # Cannot stat → do not cache, just read (and let read_bytes raise if it must).
        return decode_text(read_bytes(path, max_bytes=max_bytes)).splitlines()
    hit = _LINE_CACHE.get(key)
    if hit is not None:
        _LINE_CACHE.move_to_end(key)
        return hit
    lines = decode_text(read_bytes(path, max_bytes=max_bytes)).splitlines()
    _LINE_CACHE[key] = lines
    while len(_LINE_CACHE) > _LINE_CACHE_MAX:
        _LINE_CACHE.popitem(last=False)
    return lines


def clear_line_cache() -> None:
    """Drop the decoded-line cache (tests, and between scans in one process)."""
    _LINE_CACHE.clear()


def read_line_range(path: Path, start_line: int, end_line: int) -> str:
    """Return lines [start_line, end_line] (1-based, inclusive) as text.

    Used by the ``read_file`` navigation tool. Out-of-range bounds are clamped so
    the tool never raises on a too-large ``end_line``.
    """
    if start_line < 1:
        start_line = 1
    lines = cached_lines(path)
    if end_line < start_line:
        end_line = start_line
    # Slice is 0-based, end exclusive; our API is 1-based, end inclusive.
    selected = lines[start_line - 1 : end_line]
    return "\n".join(selected)


# ──────────────────────────────────────────────────────────────────────────────
# SQLite connection setup — WAL + busy_timeout
# ──────────────────────────────────────────────────────────────────────────────

def open_sqlite(
    db_path: Path,
    *,
    busy_timeout_ms: int = 30_000,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a SQLite connection configured for safe concurrent access.

    WAL mode lets many readers run alongside a single writer (our single-writer
    queue). ``busy_timeout`` makes a contended write wait rather than instantly
    raising "database is locked". ``check_same_thread=False`` is required because
    the writer thread and reader threads are different threads — callers are
    responsible for funneling *writes* through the single-writer queue (see
    ``db.py``); reads are safe to do directly under WAL.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=busy_timeout_ms / 1000.0,
        check_same_thread=False,
        isolation_level=None,  # autocommit; we manage transactions explicitly
    )
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)};")
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("PRAGMA synchronous=NORMAL;")  # WAL-safe, much faster than FULL
    cur.close()
    return conn


def restrict_permissions(path: Path) -> None:
    """Best-effort owner-only permissions for state/audit files.

    On POSIX this sets 0o600/0o700. On Windows ``chmod`` is largely a no-op; we
    swallow errors so data-protection is best-effort and never crashes the run.
    """
    try:
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    except (OSError, NotImplementedError):
        pass
