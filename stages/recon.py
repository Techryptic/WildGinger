# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Recon stage.

A lightweight agent explores the repo and proposes a focus-area partition. The
deterministic attack-surface inventory is passed in as a hint so the model starts
from real trust boundaries. Output is a list of focus areas (best-effort; the scan
proceeds on the full ledger regardless, so recon failure is non-fatal).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..agent import find_last_json
from ..attack_surface import AttackSurface
from ..context import ScanContext
from ..prompts.templates import RECON_PROMPT


@dataclass
class ReconResult:
    focus_areas: list[dict] = field(default_factory=list)
    error: str | None = None


def run_recon(ctx: ScanContext, surface: AttackSurface) -> ReconResult:
    agent = ctx.make_agent("recon", "recon", with_tools=True)
    hint = _surface_hint(surface)
    prompt = RECON_PROMPT.format(attack_surface_hint=hint)
    result = agent.run(prompt)
    if result.error:
        return ReconResult(error=result.error)
    parsed = find_last_json(result.tagged("focus_areas") or result.final_text)
    if isinstance(parsed, list):
        areas = [a for a in parsed if isinstance(a, dict)]
        ctx.audit.event("recon_focus_areas", count=len(areas))
        return ReconResult(focus_areas=areas)
    return ReconResult(focus_areas=[])


def _surface_hint(surface: AttackSurface, *, max_items: int = 30) -> str:
    lines: list[str] = []
    for h in (surface.sources[:max_items]):
        lines.append(f"  source[{h.label}] {h.file_rel}:{h.line}")
    for h in (surface.sinks[:max_items]):
        lines.append(f"  sink[{h.label}] {h.file_rel}:{h.line}")
    if not lines:
        return "  (no deterministic entrypoints/sinks detected; explore broadly)"
    return "\n".join(lines)
