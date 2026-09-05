# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Deterministic chunking.

Splits a file into deterministic, **overlapping**, line-numbered chunks so that:
  * a vulnerability spanning a boundary still appears intact in at least one chunk
    (overlap), and
  * the same file + same config always yields the same chunks (reproducible
    ledger + resume).

Small files become a single chunk. Where the symbol index knows function
boundaries, we prefer to align chunk starts to function starts (so a function is
not split mid-body); otherwise we fall back to fixed line windows. Either way the
output is a list of :class:`Chunk` with 1-based inclusive line ranges.

``number_lines`` renders a chunk with line-number prefixes — this is what we send
to the model so it can cite exact lines (and we can verify them).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int          # 0-based chunk index within the file
    start_line: int     # 1-based inclusive
    end_line: int       # 1-based inclusive
    text: str           # raw chunk text (no line numbers)


def chunk_text(text: str, *, chunk_lines: int, overlap_lines: int) -> list[Chunk]:
    """Fixed line-window chunking with overlap. Deterministic.

    The last chunk absorbs any remainder. Overlap means consecutive chunks share
    ``overlap_lines`` lines so a vuln on a boundary is wholly visible in one of them.
    """
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be > 0")
    if overlap_lines < 0 or overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be in [0, chunk_lines)")

    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    if n <= chunk_lines:
        return [Chunk(index=0, start_line=1, end_line=n, text="\n".join(lines))]

    step = chunk_lines - overlap_lines
    chunks: list[Chunk] = []
    start = 0  # 0-based
    idx = 0
    while start < n:
        end = min(start + chunk_lines, n)  # exclusive
        chunk_text_str = "\n".join(lines[start:end])
        chunks.append(Chunk(index=idx, start_line=start + 1, end_line=end, text=chunk_text_str))
        idx += 1
        if end >= n:
            break
        start += step
    return chunks


def chunk_by_symbols(
    text: str,
    symbol_ranges: list[tuple[int, int]],
    *,
    chunk_lines: int,
    overlap_lines: int,
) -> list[Chunk]:
    """Symbol-aware chunking: keep each function/symbol whole when it fits.

    ``symbol_ranges`` is a list of (start_line, end_line) 1-based inclusive ranges
    (from the index). A symbol that fits within ``chunk_lines`` becomes its own
    chunk; an oversized symbol is line-window split internally. Gaps between
    symbols (module-level code) are line-window chunked too, so coverage is total.

    Falls back to :func:`chunk_text` if there are no symbol ranges.
    """
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    if not symbol_ranges:
        return chunk_text(text, chunk_lines=chunk_lines, overlap_lines=overlap_lines)

    # Build a covering set of [start, end] segments over the whole file.
    # Drop degenerate ranges first. A reversed range (end < start) survived clamping
    # and then moved `cursor` backwards, so the NEXT symbol re-emitted lines already
    # covered — duplicate chunks, and every duplicate becomes a paid cell per attack
    # class. An index bug upstream must not multiply the bill.
    ranges = sorted((s, e) for (s, e) in symbol_ranges if e >= s)
    if not ranges:
        return chunk_text(text, chunk_lines=chunk_lines, overlap_lines=overlap_lines)
    segments: list[tuple[int, int]] = []
    cursor = 1
    for s, e in ranges:
        s = max(1, s)
        e = min(n, e)
        # NESTED ranges are not an upstream bug — the tree-sitter backend emits them by
        # construction, because indexing._walk appends the parent def-node and THEN
        # recurses into its children, so a class and every method inside it are all
        # reported. Appending unconditionally therefore emitted each method body a
        # SECOND time inside its enclosing class segment. Measured over this package's
        # own modules: 692 chunks where 502 were needed (2.5x on class-heavy files like
        # db.py / orchestrator.py), and every duplicate chunk is one paid cell PER
        # ATTACK CLASS — i.e. a directly proportional increase in the scan bill.
        #
        # `cursor` is "the first line not yet covered", so:
        #   * e < cursor  → this range lies entirely inside a segment we already
        #     emitted: skip it (the lines are covered, just at a coarser granularity).
        #   * s < cursor <= e → a partial overlap: TRIM to the uncovered tail rather
        #     than skipping, or coverage would lose the lines past `cursor`.
        if e < cursor:
            continue
        if s > cursor:
            segments.append((cursor, s - 1))  # gap before this symbol
        segments.append((max(s, cursor), e))
        cursor = max(cursor, e + 1)
    if cursor <= n:
        segments.append((cursor, n))

    chunks: list[Chunk] = []
    idx = 0
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len <= 0:
            continue
        if seg_len <= chunk_lines:
            body = "\n".join(lines[seg_start - 1 : seg_end])
            chunks.append(Chunk(index=idx, start_line=seg_start, end_line=seg_end, text=body))
            idx += 1
        else:
            # Oversized segment: window it with overlap.
            sub = "\n".join(lines[seg_start - 1 : seg_end])
            for c in chunk_text(sub, chunk_lines=chunk_lines, overlap_lines=overlap_lines):
                chunks.append(Chunk(
                    index=idx,
                    start_line=seg_start + c.start_line - 1,
                    end_line=seg_start + c.end_line - 1,
                    text=c.text,
                ))
                idx += 1
    return chunks


def number_lines(chunk: Chunk) -> str:
    """Render a chunk with 1-based absolute line-number prefixes for the model.

    Example: ``  42 | def login(...):``. The model cites these numbers; the
    coverage layer verifies cited snippets against the real file at those lines.
    """
    out_lines = []
    for offset, line in enumerate(chunk.text.splitlines()):
        out_lines.append(f"{chunk.start_line + offset:>6} | {line}")
    return "\n".join(out_lines)
