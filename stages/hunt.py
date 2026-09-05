# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Hunt stage.

Processes one ledger cell: runs a hunter agent (directed or open-ended) scoped to a
file's chunk + attack class, parses the structured <findings> block, and
**programmatically verifies** each cited snippet against the source before
accepting it. Returns parsed Findings + a thoroughness score for the cell.

The hunter navigates code itself via the read-only tools; observed reads are
recorded by the Navigator for proof-of-read.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .. import coverage, paths
from ..agent import find_last_json
from ..context import ScanContext
from ..findings import Finding, detect_hedging, resolve_symbol
from ..prompts.templates import HUNT_DIRECTED_PROMPT, HUNT_OPEN_PROMPT, class_guidance


@dataclass
class HuntOutcome:
    findings: list[Finding]
    thoroughness: float        # 0..1
    analyzed: bool             # True if the cell can be marked ANALYZED
    error: str | None = None


def run_hunt_cell(ctx: ScanContext, cell: sqlite3.Row, *, open_ended: bool = False) -> HuntOutcome:
    """Run one hunt task for a claimed cell row."""
    file_rel, language, start, end = _cell_location(ctx, cell)
    agent_id = f"hunt-{cell['id']}{'-open' if open_ended else ''}"
    # Pass the lease so each completed turn renews the claim (see ScanContext.make_agent):
    # a hunt with a large turn budget otherwise outlives its own 600s lease.
    lease = None
    owner = _row_value(cell, "lease_owner")
    if owner:
        lease = (cell["id"], owner)
    agent = ctx.make_agent("hunt", agent_id, with_tools=True, lease=lease)

    template = HUNT_OPEN_PROMPT if open_ended else HUNT_DIRECTED_PROMPT
    prompt = template.format(
        attack_class=cell["attack_class"],
        focus_area=file_rel,
        file_rel=file_rel,
        start_line=start,
        end_line=end,
        class_guidance=class_guidance(cell["attack_class"]),
    )

    result = agent.run(prompt)
    if result.error:
        return HuntOutcome(findings=[], thoroughness=0.0, analyzed=False, error=result.error)

    raw = result.tagged("findings") or result.final_text
    parsed = find_last_json(raw)
    truncated = getattr(result, "stop_reason", "") == "max_tokens"
    if parsed is None:
        # Could not parse — treat as shallow, re-queue. Name the REAL cause: the
        # max_tokens check used to sit after this early return, so a reply cut off
        # mid-block was always reported as "the model emitted an unparseable block",
        # sending the operator to look at prompt quality instead of the token budget.
        return HuntOutcome(
            findings=[], thoroughness=0.3, analyzed=False,
            error=("response truncated at max_tokens before the findings block was "
                   "complete — raise the role's max_tokens" if truncated
                   else "unparseable findings block"))

    findings, shape_ok = _parse_block(ctx, parsed, default_file=file_rel,
                                      default_category=cell["attack_class"])
    thoroughness = _score_thoroughness(result, findings)
    analyzed = thoroughness >= ctx.settings.quality.thoroughness_min
    if not shape_ok:
        # We could not read (all of) the answer. Keep whatever parsed — it is already
        # paid for — but never call the cell ANALYZED, or the unreadable part becomes a
        # silent hole that resume will never re-hunt.
        return HuntOutcome(
            findings=findings,
            thoroughness=min(thoroughness, 0.5),
            analyzed=False,
            error="findings block had an unrecognized shape — cell re-queued",
        )
    # A response truncated at max_tokens is INCOMPLETE by definition: any finding
    # after the cut is gone, and find_last_json() happily salvages the one complete
    # object out of a partial array — which used to look like a fully successful
    # cell. Keep whatever we parsed (it's already paid for) but never call the cell
    # ANALYZED, so it stays claimable and gets re-hunted rather than leaving a silent
    # hole in coverage.
    if truncated:
        return HuntOutcome(
            findings=findings,
            thoroughness=min(thoroughness, 0.5),
            analyzed=False,
            error="response truncated at max_tokens — findings may be incomplete",
        )
    return HuntOutcome(findings=findings, thoroughness=thoroughness, analyzed=analyzed)


# ── helpers ──────────────────────────────────────────────────────────────────

def _cell_location(ctx: ScanContext, cell: sqlite3.Row) -> tuple[str, str, int, int]:
    """Resolve a cell to (file_rel, language, chunk_start_line, chunk_end_line).

    The range comes from the ledger, where ``coverage.seed_ledger`` stored the chunk's
    real ``[start_line, end_line]``. It must NOT be re-derived here: seeding uses
    symbol-aware chunking (:func:`chunking.chunk_by_symbols`) so re-chunking with fixed
    windows produced a different, usually wrong, range for the same ``chunk_index``.

    Only pre-v3 rows (migrated mid-scan) have a NULL range; those fall back to
    re-deriving with the SAME symbol-aware algorithm used at seed time.
    """
    def _q(cur):
        cur.execute("SELECT path, language, line_count FROM files WHERE id = ?;",
                    (cell["file_id"],))
        return cur.fetchone()
    row = ctx.db.read(_q)
    file_rel = row["path"]
    language = row["language"] or ""

    start = _row_value(cell, "chunk_start")
    end = _row_value(cell, "chunk_end")
    if start is not None and end is not None and int(end) >= int(start) >= 1:
        return file_rel, language, int(start), int(end)

    # Legacy row (seeded before chunk ranges were persisted): re-derive with the
    # symbol-aware chunker so the range at least matches how the cell was created.
    derived = _derive_chunk_range(ctx, file_rel, cell["chunk_index"])
    if derived is not None:
        return file_rel, language, derived[0], derived[1]
    return file_rel, language, 1, row["line_count"] or 1


def _row_value(row, key: str):
    """Read ``key`` from a sqlite3.Row/dict, tolerating an older schema without it."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _derive_chunk_range(ctx: ScanContext, file_rel: str, idx: int) -> tuple[int, int] | None:
    """Fallback: recompute a chunk range using the SEED-TIME algorithm.

    Mirrors ``coverage.seed_ledger``: symbol ranges from the index, then
    :func:`chunking.chunk_by_symbols`. Returns None if it can't be resolved.
    """
    from .. import chunking, fileio
    from ..paths import resolve_in_jail
    try:
        target = resolve_in_jail(ctx.root, file_rel)
        text = fileio.read_text_meta(target).text
        symbols = ctx.index.functions_in(file_rel)
        symbol_ranges = [(s.start_line, s.end_line) for s in symbols]
        chunks = chunking.chunk_by_symbols(
            text,
            symbol_ranges,
            chunk_lines=ctx.settings.scope.chunk_lines,
            overlap_lines=ctx.settings.scope.chunk_overlap_lines,
        )
        if 0 <= idx < len(chunks):
            return chunks[idx].start_line, chunks[idx].end_line
    except (OSError, ValueError, AttributeError):
        pass
    return None


# Keys a model commonly wraps the findings array in. The prompt asks for a bare
# array, but an envelope object is the single most common deviation.
_ENVELOPE_KEYS = ("findings", "vulnerabilities", "vulns", "issues", "results",
                  "items", "data")


def _unwrap_items(parsed) -> tuple[list, bool]:
    """Normalize a parsed reply into a list of finding dicts.

    Returns ``(items, shape_ok)``. ``shape_ok`` is False when the reply was a shape we
    could not confidently interpret — the caller must then refuse to mark the cell
    ANALYZED, because "we could not read the answer" is not "there is nothing here".

    ``items = parsed if isinstance(parsed, list) else [parsed]`` treated an envelope
    object as ONE finding: it had no ``cited_snippet``, so it was rejected as
    ``citation_missing`` (blaming the model for the harness's parse failure), the cell
    produced zero findings, and — because the agent had made tool calls — it still
    scored 1.0 thoroughness and was committed ANALYZED. A permanent, silent coverage
    hole that reports as 100%.
    """
    if isinstance(parsed, dict):
        # A finding-shaped dict is a single finding; otherwise look for an envelope.
        if not any(k in parsed for k in ("cited_snippet", "title", "description",
                                         "line_start", "severity")):
            for key in _ENVELOPE_KEYS:
                inner = parsed.get(key)
                if isinstance(inner, list):
                    return list(inner), True
            # Exactly one list-valued key, whatever it is called.
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if len(lists) == 1:
                return list(lists[0]), True
            # An empty dict is an explicit "nothing found", which IS interpretable.
            if not parsed:
                return [], True
            return [], False
        return [parsed], True
    if isinstance(parsed, list):
        # A singleton nested list is another common wrapper.
        if len(parsed) == 1 and isinstance(parsed[0], list):
            return list(parsed[0]), True
        return list(parsed), True
    # A bare string/number/None is not a findings block at all.
    return [], False


def _to_findings(ctx: ScanContext, parsed, *, default_file: str,
                 default_category: str) -> list[Finding]:
    """Build Findings from a parsed reply (findings only; see :func:`_parse_block`)."""
    findings, _shape_ok = _parse_block(ctx, parsed, default_file=default_file,
                                       default_category=default_category)
    return findings


def _parse_block(ctx: ScanContext, parsed, *, default_file: str,
                 default_category: str) -> tuple[list[Finding], bool]:
    """Build Findings from a parsed reply. Returns ``(findings, shape_ok)``.

    ``shape_ok`` is False when any part of the reply could not be interpreted, which
    the caller turns into "do not mark this cell ANALYZED".
    """
    items, shape_ok = _unwrap_items(parsed)
    if not shape_ok:
        ctx.audit.event("findings_shape_unrecognized",
                        parsed_type=type(parsed).__name__,
                        preview=str(parsed)[:200])
    out: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            # Previously skipped with NO audit event at all, so the drop was invisible.
            ctx.audit.event("finding_item_not_an_object",
                            item_type=type(item).__name__, item=str(item)[:200])
            shape_ok = False
            continue
        # Per-item isolation: one malformed entry must never discard the cell's other
        # (valid) findings, which is what an uncaught coercion error used to do.
        try:
            finding = _one_finding(ctx, item, default_file=default_file,
                                   default_category=default_category)
        except Exception as exc:  # noqa: BLE001 — a bad item is skipped, not fatal
            ctx.audit.event("finding_parse_error", error=str(exc),
                            item=str(item)[:200])
            shape_ok = False
            continue
        if finding is not None:
            out.append(finding)
    return out, shape_ok


def _one_finding(ctx: ScanContext, item: dict, *, default_file: str,
                 default_category: str) -> Finding | None:
    """Build one verified Finding from a parsed item, or None if it fails the gate."""
    # CANONICALIZE the model's spelling before anything keys on it (see
    # paths.canonical_rel): './app/x.py' and '/app/x.py' used to pass the citation gate
    # and then behave as different files everywhere else.
    raw_file = str(item.get("file") or default_file)
    file_rel = paths.canonical_rel(ctx.root, raw_file) or default_file
    if file_rel != raw_file:
        ctx.audit.event("finding_path_normalized", raw=raw_file[:200],
                        normalized=file_rel)
    snippet = str(item.get("cited_snippet", "") or "")
    ls = _int(item.get("line_start"), 0)
    le = _int(item.get("line_end"), ls)

    # ANTI-HALLUCINATION GATE. A citation is MANDATORY: with no
    # snippet there is nothing to verify, so omitting it used to be an easy way to
    # smuggle an unverifiable finding straight past this check.
    if not snippet.strip():
        ctx.audit.event("citation_missing", file_rel=file_rel,
                        line_start=ls, line_end=le,
                        title=str(item.get("title", ""))[:120])
        return None
    located = coverage.locate_snippet(ctx.root, file_rel, ls, le, snippet)
    if located is None:
        ctx.audit.event("citation_rejected", file_rel=file_rel,
                        line_start=ls, line_end=le, snippet=snippet[:120])
        return None
    # CORRECT the range to where the snippet actually is, BEFORE stamping the symbol.
    # The gate tolerates a citation that is a couple of lines off, and `symbol` is half
    # of the fingerprint — so an uncorrected off-by-two at a function boundary resolved
    # to the PREVIOUS function and gave the same bug a second identity.
    if located != (ls, le):
        ctx.audit.event("citation_relocated", file_rel=file_rel,
                        claimed=[ls, le], actual=list(located))
        ls, le = located

    desc = str(item.get("description", ""))
    return Finding(
        file_rel=file_rel,
        line_start=ls,
        line_end=le,
        # `default_category` is the cell's attack_class, which for an open-ended cell is
        # the literal "all" — not a category. "all" has no CWE/OWASP mapping and gives the
        # same bug a different location_key than when it is found under a real class, so
        # the two never merge. The prompt asks for "novel/other"; use that as the floor.
        category=str(item.get("category") or _default_category(default_category)),
        title=str(item.get("title", "")),
        description=desc,
        severity=_severity(item.get("severity")),
        confidence=_confidence(item.get("confidence")),
        cwe=item.get("cwe"),
        hedged=detect_hedging(desc),
        cited_snippet=snippet,
        # Stamp the enclosing symbol NOW, while we have the index. This is the stable
        # half of the finding's identity (see findings.Finding.fingerprint).
        symbol=resolve_symbol(getattr(ctx, "index", None), file_rel, ls),
        remediation=item.get("remediation"),
        source="llm",
    )


# Models often answer with a word where the schema asks for a 0..1 number.
_CONFIDENCE_WORDS = {
    "very high": 0.95, "high": 0.8, "medium": 0.5, "moderate": 0.5,
    "low": 0.3, "very low": 0.15, "certain": 0.95, "unsure": 0.3,
}
_VALID_SEVERITIES = ("informational", "low", "medium", "high", "critical")


def _confidence(value, default: float = 0.5) -> float:
    """Coerce a model-supplied confidence into 0..1, never raising.

    ``float("high")`` used to raise ValueError out of the whole cell, discarding EVERY
    finding it produced (not just the malformed one). Accepts numbers, numeric strings,
    explicit percentages ("80%") and the common word scale.

    Out-of-range numbers are interpreted conservatively, since the schema asks for
    0..1 and we can't know the model's intended scale:
      * ``> 100``    → 1.0 (nonsense; clamp)
      * ``10..100``  → treated as a percentage (85 → 0.85), a very common habit
      * ``1..10``    → clamped to 1.0 (an overshoot of the 0..1 scale, NOT a percent —
        reading 3 as 0.03 would silently bury a finding the model was confident about)
    """
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[text]
        is_pct = text.endswith("%")
        try:
            num = float(text.rstrip("%").strip())
        except ValueError:
            return default
        if is_pct:
            return max(0.0, min(1.0, num / 100.0))
    else:
        return default

    if num > 100.0:
        return 1.0
    if num > 10.0:
        return max(0.0, min(1.0, num / 100.0))
    if num > 1.0:
        # Deliberately NOT monotonic at the 10 boundary (10 -> 1.0, 10.5 -> 0.105). My own
        # audit flagged that as a defect; on reflection the existing choice is the right
        # one and the "fix" is worse. A bare 3 or 7 is far more likely an overshoot of the
        # 0..1 scale than "3%", and reading it as a percentage would BURY a finding the
        # model was confident about — the dangerous direction for a security gate. Any
        # threshold has a discontinuity somewhere; this one errs toward keeping findings.
        return 1.0
    return max(0.0, num)


def _severity(value, default: str = "medium") -> str:
    """Normalize severity to a known bucket so scoring/reporting can't be surprised."""
    text = str(value or "").strip().lower()
    if text in _VALID_SEVERITIES:
        return text
    # Common synonyms seen from models.
    return {"info": "informational", "informational/low": "informational",
            "moderate": "medium", "severe": "high", "crit": "critical"}.get(text, default)


def _score_thoroughness(result, findings: list[Finding]) -> float:
    """Heuristic thoroughness: did the agent actually navigate + cite specifics?

    Tool usage + specific citations raise the score; a one-shot "looks fine" with no
    tool calls scores low and gets re-queued.

    Only SUCCESSFUL tool calls count. ``tool_calls`` is incremented before dispatch and
    a failing handler is converted into an error block, so an agent that called an
    unknown tool three times, or asked for three path-jailed/binary files, scored a
    perfect 1.0 and had its cell committed ANALYZED with nothing read and nothing found.
    """
    calls = getattr(result, "successful_tool_calls", None)
    if calls is None:  # duck-typed result objects in tests
        calls = max(0, getattr(result, "tool_calls", 0)
                    - getattr(result, "tool_errors", 0))
    score = 0.0
    if calls >= 1:
        score += 0.4
    if calls >= 3:
        score += 0.2
    if findings:
        score += 0.4  # produced concrete, verified citations
    else:
        # No findings is valid IF the agent actually looked (successful tool calls).
        score += 0.4 if calls >= 2 else 0.0
    return min(1.0, score)


def _default_category(attack_class: str) -> str:
    """The category to stamp when the model did not supply one."""
    ac = (attack_class or "").strip().lower()
    # A comma-joined GROUP label (attack_class_groups) is a bucket, not a category —
    # the same rule as "all": stamping it would give the same bug a different
    # location_key than when it is found under a real class, so the two never merge.
    return "novel/other" if ac in ("", "all") or "," in ac else ac


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
