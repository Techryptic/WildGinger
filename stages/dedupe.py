# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Dedupe stage.

Two layers:
  1. **Deterministic** — signature/fingerprint + location merge (always on; free).
  2. **AI semantic** (:func:`semantic_dedupe`) — an LLM groups findings that share a
     root cause but slip past the heuristic (same bug phrased differently, or a sink
     reported at both the handler and the data-access layer). This runs BEFORE the
     multi-judge validation so we never pay judges to confirm duplicates.
"""
from __future__ import annotations

from ..agent import find_last_json
from ..audit import BudgetExceeded
from ..context import ScanContext
from ..findings import (
    Finding,
    _absorb,
    _severity_rank,
    merge_findings,
    sort_canonical,
)
from ..prompts.templates import DEDUPE_PROMPT


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Collapse duplicates by fingerprint, accumulate votes, and order canonically.

    The canonical sort matters here because this is the last thing to touch the list
    before reporting: dict insertion order reflects whichever hunter thread finished
    first, so without it the report's finding order changed between identical runs.
    """
    return sort_canonical(merge_findings(findings))


def _merge_group(members: list[Finding]) -> Finding:
    """Merge a group of duplicate findings into the strongest representative."""
    # Keep the highest-severity, highest-confidence one as the base.
    rep = max(members, key=lambda f: (_severity_rank(f.severity), f.confidence))
    for f in members:
        if f is rep:
            continue
        # Shared fold logic: accumulates votes, keeps the strongest claim, and
        # backfills detail (snippet/symbol/cwe) the representative is missing.
        _absorb(rep, f)
    return rep


def _as_index(value: object, n: int) -> int | None:
    """Coerce one model-supplied group member to a valid finding index, or None.

    The old check was ``isinstance(x, (int, float))``, which rejected the very common
    case of a model quoting its ids as strings: ``[["0","3"],["1"]]`` silently yielded
    ZERO usable ids, every finding became its own singleton, and the pass reported
    ``removed=0`` — identical to "no duplicates found" — while the entire raw set went
    on to the 3-judge panel. Booleans are excluded deliberately: ``True`` is an ``int``
    in Python and would otherwise resolve to index 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        idx = int(value)
    elif isinstance(value, str):
        token = value.strip()
        # Tolerate "3", " 3 ", "[3]", "#3", and "3." shapes.
        token = token.strip("[]#().")
        try:
            idx = int(token)
        except ValueError:
            return None
    else:
        return None
    return idx if 0 <= idx < n else None


# Findings per LLM dedupe call. ~150 lines is a few thousand prompt tokens and a
# partition the model can comfortably emit inside a 4000-token budget.
_MAX_BATCH = 150


def _one_line(text: str, *, limit: int = 120) -> str:
    """Collapse a model-authored title to one short line for the prompt.

    ``title`` is unbounded, unsanitized model text that may quote repository content, so
    a newline in it would break the numbered-list contract the reply is parsed against.
    """
    return " ".join(str(text or "").split())[:limit]


def _dedupe_in_batches(ctx: ScanContext, findings: list[Finding], *,
                       role: str) -> tuple[list[Finding], int]:
    """Run :func:`semantic_dedupe` over fixed-size slices and concatenate the results."""
    out: list[Finding] = []
    removed = 0
    batches = (len(findings) + _MAX_BATCH - 1) // _MAX_BATCH
    ctx.audit.event("semantic_dedupe_batched", findings=len(findings),
                    batches=batches, batch_size=_MAX_BATCH)
    for start in range(0, len(findings), _MAX_BATCH):
        chunk = findings[start:start + _MAX_BATCH]
        print(f"  [dedupe batch {start // _MAX_BATCH + 1}/{batches}] "
              f"{len(chunk)} finding(s)…")
        merged, n = semantic_dedupe(ctx, chunk, role=role)
        out.extend(merged)
        removed += n
    return out, removed


def semantic_dedupe(ctx: ScanContext, findings: list[Finding], *,
                    role: str = "dedupe") -> tuple[list[Finding], int]:
    """AI pass: group same-root-cause findings and merge each group to one.

    Returns ``(deduped_findings, removed_count)``. Fail-open: on any error or an
    unparseable reply, returns the input unchanged (removed=0) so dedupe never drops
    a real finding or breaks the scan. Skips the LLM call for trivially-small sets.
    """
    if len(findings) < 3:
        # Benign, but still recorded so "no duplicates removed" is never ambiguous.
        ctx.audit.event("semantic_dedupe_skipped", reason="too_few_findings",
                        count=len(findings))
        return findings, 0
    if role not in ctx.settings.roles:
        # A misconfigured dedupe.semantic_role silently disabled this whole pass,
        # indistinguishable in the output from "found no duplicates". Settings now
        # rejects it at load time, so reaching here means the role was removed
        # programmatically — say so loudly either way.
        ctx.audit.event("semantic_dedupe_skipped", reason="role_not_configured",
                        role=role, known_roles=sorted(ctx.settings.roles))
        print(f"⚠ AI semantic dedupe SKIPPED: role '{role}' is not configured "
              f"(known roles: {', '.join(sorted(ctx.settings.roles))}). "
              f"Duplicates will NOT be merged by the LLM pass.")
        return findings, 0

    # BATCHED. One line per finding in a single prompt is unbounded: 6000 findings is
    # ~571k prompt characters (~143k input tokens), and the partition the model must
    # return exceeds the dedupe role's max_tokens at roughly 2700 findings — so the reply
    # was truncated, the "dropped ids become singletons" fallback silently fired, and the
    # most expensive prompt in the system stopped working at exactly the scale it exists
    # for. Grouping is a LOCAL decision, and cross-batch duplicates are still caught by
    # the fingerprint merge, so batching costs no recall.
    if len(findings) > _MAX_BATCH:
        return _dedupe_in_batches(ctx, findings, role=role)

    lines = []
    for i, f in enumerate(findings):
        lines.append(f"  [{i}] {f.file_rel}:{f.line_start}-{f.line_end} "
                     f"({f.category}) {_one_line(f.title)}")
    prompt = DEDUPE_PROMPT.format(items="\n".join(lines))

    try:
        agent = ctx.make_agent(role, "dedupe", with_tools=False)
        result = agent.run(prompt)
        if result.error:
            ctx.audit.event("semantic_dedupe_skipped", reason="agent_error",
                            error=result.error, stop_reason=result.stop_reason)
            return findings, 0
        groups = find_last_json(result.tagged("groups") or result.final_text)
    except BudgetExceeded:
        # The hard cost ceiling is NOT a "fail-open" condition — it must stop the
        # whole scan, not silently skip this dedupe pass.
        raise
    except Exception as exc:  # noqa: BLE001 — never break the scan
        ctx.audit.event("semantic_dedupe_error", error=str(exc))
        return findings, 0

    if not isinstance(groups, list) or not groups:
        return findings, 0

    # Validate the partition: every index exactly once, all in range.
    n = len(findings)
    seen: set[int] = set()
    clean_groups: list[list[int]] = []
    parsed_any = False
    for g in groups:
        if not isinstance(g, list):
            continue
        ids = []
        for x in g:
            idx = _as_index(x, n)
            if idx is not None:
                ids.append(idx)
        parsed_any = parsed_any or bool(ids)
        # Drop repeats WITHIN this group before the cross-group filter. The `seen` filter
        # only looked at PREVIOUS groups, so `[[0, 0, 1]]` survived intact and
        # _merge_group absorbed finding 0 into the representative twice — inflating votes
        # (5+1 became 11) and duplicating its evidence, which scoring.enrich then turned
        # into a real confidence bonus for a parsing artefact.
        ids = list(dict.fromkeys(ids))
        ids = [x for x in ids if x not in seen]
        for x in ids:
            seen.add(x)
        if ids:
            clean_groups.append(ids)
    # Any indices the model dropped → keep them as their own singletons (fail-safe).
    for i in range(n):
        if i not in seen:
            clean_groups.append([i])

    if not parsed_any:
        # The model DID return groups but not one id was usable. That is a prompt/
        # parsing failure, not "no duplicates found" — and the two used to be
        # indistinguishable in the output (both reported removed=0) while the full raw
        # set went on to the expensive judge panel.
        ctx.audit.event("semantic_dedupe_skipped", reason="no_parsable_ids",
                        groups_returned=len(groups), findings=n)
        print(f"⚠ AI semantic dedupe returned {len(groups)} group(s) but no usable "
              f"finding ids — treating as NO duplicates merged. Judge cost will not "
              f"be reduced by this pass.")
        return findings, 0

    # A single group covering most of the input is not a credible dedupe result — it is a
    # finding-SUPPRESSION primitive (only one representative survives each group), and
    # `title` is model-authored text that a scanned repo can influence. Fail open.
    biggest = max((len(g) for g in clean_groups), default=0)
    if n >= 4 and biggest > max(2, n // 2):
        ctx.audit.event("semantic_dedupe_refused", reason="group_too_large",
                        biggest_group=biggest, findings=n)
        print(f"⚠ AI semantic dedupe proposed collapsing {biggest} of {n} finding(s) "
              f"into one — refusing as implausible; nothing was merged.")
        return findings, 0

    merged: list[Finding] = [_merge_group([findings[i] for i in g]) for g in clean_groups]
    removed = n - len(merged)
    if removed:
        ctx.audit.event("semantic_dedupe", before=n, after=len(merged), removed=removed)
    return merged, removed
