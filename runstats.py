# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Run-stats helpers: the model-config string + the end-of-run metrics block.

Shared by the reporter (writes the block at the top of report.md + into report.json)
and the orchestrator (prints it at the end of a run). Kept dependency-free so it
never breaks a scan.
"""
from __future__ import annotations

from typing import Any


def model_config_string(settings: Any) -> str:
    """Build the human-readable model/config line, e.g.::

        Recon:opus47, hunt:sonnet46, validate:sonnet46, trace:opus47, judges:sonnet46,opus47,sonnet46

    Sourced from ``roles`` + ``validation.judges`` in config.yaml so it always
    reflects the config actually used for this run.
    """
    roles = getattr(settings, "roles", {}) or {}

    def _model(role: str) -> str | None:
        rc = roles.get(role)
        return getattr(rc, "model", None) if rc is not None else None

    parts: list[str] = []
    for label, role in (("Recon", "recon"), ("hunt", "hunt")):
        m = _model(role)
        if m:
            parts.append(f"{label}:{m}")

    # validate: show the judge panel if configured, else the single 'validate' role.
    judges = list(getattr(getattr(settings, "validation", None), "judges", []) or [])
    val_model = _model("validate")
    if val_model:
        parts.append(f"validate:{val_model}")

    trace_model = _model("trace")
    if trace_model:
        parts.append(f"trace:{trace_model}")

    judge_models = [m for m in (_model(j) for j in judges) if m]
    if judge_models:
        parts.append("judges:" + ",".join(judge_models))
    return ", ".join(parts)


def format_runtime(seconds: float) -> str:
    """Format elapsed seconds like ``10h14m`` / ``12m03s`` / ``42s``."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_cost(cost: float) -> str:
    """Format a USD cost like ``$4.32`` (or ``$4,320.00`` for large sums)."""
    return f"${cost:,.2f}"


def build_run_stats(settings: Any, cost_snapshot: dict[str, Any],
                    elapsed_seconds: float, *, price_mismatches: int = 0) -> dict[str, Any]:
    """Assemble the end-of-run stats dict from a cost-ledger snapshot + elapsed time.

    ``price_mismatches`` counts calls where a provider reported what it actually charged
    and that figure disagreed materially with our configured price (only OpenRouter
    reports one today). It is surfaced because a wrong price means a wrong ledger, and
    therefore a hard cost ceiling that fires at the wrong time in one direction or the
    other.
    """
    cs = cost_snapshot or {}
    return {
        "model_config": model_config_string(settings),
        # Which config file produced this run, next to the hash of its contents. The hash
        # alone cannot distinguish config.yaml from config_api.yaml.
        "config_file": getattr(settings, "config_source", "<unknown>"),
        "config_hash": settings.config_hash() if hasattr(settings, "config_hash") else "",
        "tokens_in": int(cs.get("input_tokens", 0) or 0),
        "tokens_out": int(cs.get("output_tokens", 0) or 0),
        "cache_read": int(cs.get("cache_read_tokens", 0) or 0),
        "cache_write": int(cs.get("cache_write_tokens", 0) or 0),
        "runtime_seconds": int(max(0, elapsed_seconds)),
        "runtime": format_runtime(elapsed_seconds),
        "cost_usd": float(cs.get("cost_usd", 0.0) or 0.0),
        "calls": int(cs.get("calls", 0) or 0),
        "price_mismatches": int(price_mismatches or 0),
    }


def summary_lines(stats: dict[str, Any]) -> list[str]:
    """The metrics block as a list of plain lines (shared by console + report)."""
    lines = [
        f"Models:      {stats.get('model_config', '')}",
        f"Config:      {stats.get('config_file', '')}"
        + (f" (hash {stats['config_hash']})" if stats.get("config_hash") else ""),
        f"Tokens In:   {stats.get('tokens_in', 0):,}",
        f"Tokens Out:  {stats.get('tokens_out', 0):,}",
        f"Cache Read:  {stats.get('cache_read', 0):,}",
        f"Cache Write: {stats.get('cache_write', 0):,}",
        f"Runtime:     {stats.get('runtime', '')}",
        f"Total Cost:  {format_cost(stats.get('cost_usd', 0.0))}",
    ]
    mismatches = int(stats.get("price_mismatches", 0) or 0)
    if mismatches:
        lines.append(
            f"⚠ Cost check:  {mismatches} call(s) billed materially differently from the "
            f"configured price — see provider_reported_cost_usd in audit/events.jsonl")
    return lines


def markdown_block(stats: dict[str, Any]) -> list[str]:
    """The metrics block as Markdown lines for the top of report.md."""
    return [
        "## Run summary", "",
        f"- **Models:** `{stats.get('model_config', '')}`",
        f"- **Config:** `{stats.get('config_file', '')}`"
        + (f" (hash `{stats['config_hash']}`)" if stats.get("config_hash") else ""),
        f"- **Tokens In:** {stats.get('tokens_in', 0):,}",
        f"- **Tokens Out:** {stats.get('tokens_out', 0):,}",
        f"- **Cache Read:** {stats.get('cache_read', 0):,}",
        f"- **Cache Write:** {stats.get('cache_write', 0):,}",
        f"- **Runtime:** {stats.get('runtime', '')}",
        f"- **Total Cost:** {format_cost(stats.get('cost_usd', 0.0))}",
        "",
    ]
