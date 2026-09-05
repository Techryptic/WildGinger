# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Trace stage — real source→sink reachability.

Answers the distinct question "can attacker-controlled input actually REACH this
finding?" — but *deterministically first*, by walking the symbol-level call graph
UP from the finding's enclosing function toward an **attack-surface entrypoint**
(an HTTP route / request handler / deserialization / CLI / message consumer that
the ``attack_surface`` pass detected as a source).

Why this matters for cost: this deterministic gate runs BEFORE the expensive
multi-judge validation. Findings that are provably NOT reachable from any external
entry point are demoted to *informational* (per ``reachability.unreachable_handling``)
and never sent to the judges — so we don't spend judge money confirming a bug an
attacker can't trigger.

Verdicts:
  * ``reachable``   — the enclosing function is itself an entrypoint, or a caller
                      chain leads up to one.
  * ``unreachable`` — the enclosing function has no callers and is not an
                      entrypoint (dead / internal-only code).
  * ``unclear``     — we couldn't resolve an enclosing symbol (fail-open: keep it).

An optional LLM deep-trace (the previous behavior) can still refine ``unclear`` /
confirmed findings after validation; it is intentionally NOT on the gating path.
"""
from __future__ import annotations

from ..agent import find_last_json
from ..attack_surface import AttackSurface
from ..context import ScanContext
from ..findings import Finding
from ..indexing import Index

from ..prompts.templates import TRACE_PROMPT

# Bound the caller-graph walk so a pathological cycle / huge graph can't hang.
_MAX_DEPTH = 25


# ── deterministic reachability ─────────────────────────────────────────────────

def entrypoint_symbols(index: Index, surface: AttackSurface) -> set[str]:
    """Names of functions that contain (or are decorated at) an attack-surface
    *source* — i.e. externally reachable entry points (routes, request handlers,
    deserialization, CLI, message consumers)."""
    names: set[str] = set()
    for hit in surface.sources:
        sym = index.enclosing_symbol(hit.file_rel, hit.line)
        if sym is not None:
            names.add(sym.name)
    return names


def reachability_verdict(index: Index, surface: AttackSurface,
                         finding: Finding, entrypoints: set[str] | None = None) -> str:
    """Deterministically classify a finding as reachable / unreachable / unclear
    by walking the symbol-level call graph up toward an entrypoint.

    ``entrypoints`` should be passed by callers that classify more than one finding.
    Recomputing it here made the gate quadratic: ``entrypoint_symbols`` calls
    ``index.enclosing_symbol`` once per attack-surface source hit, and it was rebuilt
    from scratch for EVERY finding even though both ``index`` and ``surface`` are
    immutable for the whole scan. Measured on a 1500-file corpus with 54k source hits:
    1.08 MILLION enclosing_symbol calls to classify 20 findings, 76% of the gate's
    runtime — extrapolating to minutes per pass of pure CPU that produces nothing.
    """
    origin = index.enclosing_symbol(finding.file_rel, finding.line_start)
    if origin is None:
        # Can't tie the finding to a function → don't gate it out (fail-open).
        return "unclear"

    if entrypoints is None:
        entrypoints = entrypoint_symbols(index, surface)
    if not entrypoints:
        # No external entry points detected anywhere → we have no basis to declare
        # anything "unreachable". Fail open so we never gate out a real bug just
        # because the source heuristics found nothing (e.g. a library/CLI codebase).
        return "unclear"


    # BFS up the caller graph (cycle-guarded, depth-capped).
    visited: set[str] = set()
    frontier: list[tuple[str, int]] = [(origin.name, 0)]
    while frontier:
        name, depth = frontier.pop()
        if name in visited or depth > _MAX_DEPTH:
            continue
        visited.add(name)
        if name in entrypoints:
            return "reachable"
        for caller in index.function_callers(name):
            if caller.name not in visited:
                frontier.append((caller.name, depth + 1))

    # Nothing in the visited set was an entrypoint. If the origin function has no
    # callers at all, it's dead/internal → unreachable. Otherwise callers exist but
    # none lead to a source we recognize → unclear (fail-open, keep for judges).
    if not index.function_callers(origin.name):
        return "unreachable"
    return "unclear"


def gate_unreachable(ctx: ScanContext, findings: list[Finding],
                     surface: AttackSurface | None = None
                     ) -> tuple[list[Finding], list[Finding]]:
    """Split ``findings`` into ``(to_validate, demoted)`` using the deterministic
    reachability gate, honoring ``reachability.unreachable_handling``:

      * ``informational`` — unreachable findings are demoted (reachable="unreachable")
        and returned in ``demoted`` so they bypass the expensive judges but still
        appear in the report's informational section.
      * ``hide``          — unreachable findings are dropped entirely.
      * ``report``        — nothing is gated; every finding goes to validation.

    Returns ``(to_validate, demoted)``. ``reachable`` / ``unclear`` findings always
    go to validation (fail-open). If reachability is disabled, gating is a no-op.

    ``surface`` MUST be passed explicitly by the caller. It used to be read via
    ``getattr(ctx, "_surface", None)``, but ``ScanContext`` has no such field — only
    the Orchestrator does — so the gate silently ran against an empty surface and
    never demoted anything. Passing it as a parameter makes the dependency real and
    impossible to lose again. ``None`` falls back to an empty surface, which
    fail-opens (every verdict "unclear") rather than gating incorrectly.
    """
    rc = ctx.settings.reachability
    handling = rc.unreachable_handling
    getter = getattr(ctx.settings, "effective_unreachable_handling", None)
    if getter is not None:
        handling = getter()
    if not rc.enabled or handling == "report":
        return list(findings), []

    index = ctx.index
    if surface is None:
        surface = AttackSurface()
    to_validate: list[Finding] = []
    demoted: list[Finding] = []
    # Computed ONCE for the whole batch (see reachability_verdict).
    entrypoints = entrypoint_symbols(index, surface)
    for f in findings:
        verdict = reachability_verdict(index, surface, f, entrypoints=entrypoints)
        if verdict == "unreachable":
            if handling == "hide":
                continue  # drop entirely
            # informational: demote and keep out of the judge queue.
            f.reachable = "unreachable"
            f.evidence.append("reachability: no caller path to an external entry "
                              "point (deterministic call-graph walk) — demoted to "
                              "informational, not sent to judges.")
            demoted.append(f)
        else:
            f.reachable = verdict if verdict == "reachable" else f.reachable
            to_validate.append(f)
    return to_validate, demoted


# ── optional LLM deep-trace (post-validation refinement; NOT on gating path) ────

def trace_reachability(ctx: ScanContext, finding: Finding, surface: AttackSurface) -> Finding:
    """LLM-grounded reachability refinement for a confirmed finding.

    The deterministic gate has already classified obvious dead code; this asks a
    model to confirm/refine the reachability verdict + path for a confirmed finding,
    grounded by the deterministic source hints. Kept for depth on ``unclear`` cases.
    """
    agent = ctx.make_agent("trace", f"trace-{finding.fingerprint()}", with_tools=True)
    hints = _source_hints(surface)
    prompt = TRACE_PROMPT.format(
        file_rel=finding.file_rel,
        line_start=finding.line_start,
        line_end=finding.line_end,
        title=finding.title,
        source_hints=hints,
    )
    result = agent.run(prompt)
    if not result.error:
        parsed = find_last_json(result.tagged("reachability") or result.final_text)
        if isinstance(parsed, dict):
            r = str(parsed.get("reachable", "unclear")).lower()
            if r in ("reachable", "unreachable", "unclear"):
                finding.reachable = r
            path = parsed.get("path")
            if path:
                finding.evidence.append(f"reachability: {path}")
    return finding


def _source_hints(surface: AttackSurface, *, max_items: int = 25) -> str:
    lines = [f"  {h.label} {h.file_rel}:{h.line}" for h in surface.sources[:max_items]]
    return "\n".join(lines) if lines else "  (no deterministic sources detected)"
