# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Dry-run estimation — projects scope/cost with ZERO LLM calls.

Walks discovery + chunking + ledger math to print the projected cell count and a
rough token/cost/time estimate, so an expensive run can be planned before any spend.
"""
from __future__ import annotations

from . import chunking, coverage, discovery, fileio, indexing
from .paths import resolve_target
from .settings import Settings

# Rough per-cell token assumption (prompt + a few tool reads + response).
_TOKENS_PER_CELL = 6000
# Assumed input/output split of those tokens. Hunting is read-heavy: a big prompt plus
# several file reads, against a comparatively short findings block.
_INPUT_SHARE = 0.85
# Conservative blended price (USD per 1,000,000 tokens) used ONLY when a model has no
# configured price. Deliberately not cheap: an estimate that under-promises cost is
# the dangerous direction.
_FALLBACK_PRICE_PER_1M = 15.0
# Rough wall-clock seconds per cell at the configured concurrency.
_SECONDS_PER_CELL = 8.0


def _blended_price_per_1m(settings: Settings) -> tuple[float, float, str]:
    """(input, output) $/1M for the roles that actually do the work, + a note.

    The old estimate hardcoded a single $3/1M blended rate and ignored config
    entirely, so a run using a $10/$30 model under-estimated by roughly 5x. We weight
    by where the tokens really go: hunting dominates, with judges adding a slice per
    surviving finding.
    """
    weights = {"hunt": 1.0}
    judges = list(getattr(settings.validation, "judges", []) or [])
    if judges:
        # ~10% of cells yield a finding, each judged by the whole panel.
        for role in judges:
            weights[role] = weights.get(role, 0.0) + 0.1
    else:
        weights["validate"] = 0.2

    total_w = 0.0
    in_sum = out_sum = 0.0
    missing: list[str] = []
    free: list[str] = []
    for role, weight in weights.items():
        try:
            _rc, mp = settings.model_for_role(role)
        except Exception:  # noqa: BLE001 — role not configured; skip it
            continue
        in_price = mp.price_per_1m_input
        out_price = mp.price_per_1m_output
        if getattr(mp, "billing", "metered") == "free":
            # A model on your own hardware bills nothing, and $0 here is the TRUTH, not
            # missing data. Substituting the pessimistic fallback would project hundreds
            # of dollars for a run that cannot cost anything — an estimate that scares
            # the operator away from a free configuration is as useless as one that
            # under-promises a paid one.
            in_price = out_price = 0.0
            free.append(f"{role}→{mp.model_id[:28]}")
        elif in_price <= 0 and out_price <= 0:
            missing.append(f"{role}→{mp.model_id[:28]}")
            in_price = out_price = _FALLBACK_PRICE_PER_1M
        in_sum += in_price * weight
        out_sum += out_price * weight
        total_w += weight

    if total_w <= 0:
        return _FALLBACK_PRICE_PER_1M, _FALLBACK_PRICE_PER_1M, "no priced roles"
    notes: list[str] = []
    if missing:
        notes.append(f"assumed ${_FALLBACK_PRICE_PER_1M:.0f}/1M for unpriced model(s): "
                     f"{', '.join(missing)}")
    if free:
        notes.append(f"priced at $0 because billing: free is declared for: "
                     f"{', '.join(free)} (only budget.max_time_minutes bounds those)")
    return in_sum / total_w, out_sum / total_w, "; ".join(notes)



def price_known_cells(settings: Settings, *, cells: int, files: int, skipped: int,
                     total_lines: int, quiet: bool = False) -> dict:
    """Project cost/time for a ledger whose EXACT size is already known.

    The scan path calls this instead of :func:`estimate`. ``_run_inner`` has already
    walked discovery, read every file, built the index and seeded the ledger — so it
    knows ``stats.cells`` exactly — yet it then called ``estimate()``, which repeated all
    of it (a second discovery walk, a second full read of every file, a second
    ``build_index`` including a second tree-sitter parse, and a re-chunk of every file)
    purely to APPROXIMATE a number it already had. On a large repo that roughly doubled
    pre-scan wall-clock, CPU and peak memory on the critical path of every run, and
    ``--yes``/``--ci`` skipped only the prompt, not the work.
    """
    # MUST mirror coverage.seed_ledger exactly, or the projected cost silently
    # under-states the real ledger (the dangerous direction for an estimate).
    attack_classes = coverage.resolve_attack_classes(settings)
    chunks = cells // max(1, len(attack_classes))
    return _project(settings, cells=cells, files=files, skipped=skipped,
                    total_lines=total_lines, total_chunks=chunks,
                    attack_classes=attack_classes, quiet=quiet)


def estimate(settings: Settings) -> dict:
    # A single FILE target roots the scan at its parent and narrows what is hunted to that
    # one file (see paths.resolve_target). The index below is still built over the whole
    # folder — matching the scan — but only the target's chunks are priced, or the dry-run
    # would quote the cost of the entire directory.
    root, only_file = resolve_target(settings.scope.path)
    disc = discovery.discover(root, settings.scope)
    # MUST mirror coverage.seed_ledger exactly, or the projected cost silently
    # under-states the real ledger (the dangerous direction for an estimate).
    attack_classes = coverage.resolve_attack_classes(settings)

    # Build the symbol index so chunking here matches SEEDING exactly. Using
    # chunk_text while coverage.seed_ledger uses chunk_by_symbols under-counted cells
    # by ~6x on real code (symbol-aware chunking splits per function), so the dry-run
    # promised a fraction of the true cost — the dangerous direction for an estimate.
    file_texts: list[tuple[str, str, str]] = []
    for df in disc.files:
        try:
            file_texts.append((df.rel_path, df.language,
                               fileio.read_text_meta(df.path).text))
        except (OSError, ValueError):
            continue
    index = indexing.build_index(file_texts, backend=settings.indexing.backend)

    total_chunks = 0
    total_lines = 0
    texts_by_rel = {rel: text for rel, _lang, text in file_texts}
    priced_files = [df for df in disc.files
                    if only_file is None or df.rel_path == only_file]
    for df in priced_files:
        text = texts_by_rel.get(df.rel_path)
        if text is None:
            continue
        total_lines += df.line_count
        # Mirror coverage.seed_ledger: symbol-aware chunking, same parameters.
        symbol_ranges = [(sym.start_line, sym.end_line)
                         for sym in index.functions_in(df.rel_path)]
        chunks = chunking.chunk_by_symbols(
            text,
            symbol_ranges,
            chunk_lines=settings.scope.chunk_lines,
            overlap_lines=settings.scope.chunk_overlap_lines,
        )
        total_chunks += max(len(chunks), 1)

    return _project(settings, cells=total_chunks * len(attack_classes),
                    # `files` is what will be HUNTED, so a single-file scope reports 1 —
                    # quoting "8 files" for a one-file scan would misstate the scope of
                    # the run the number belongs to.
                    files=len(priced_files), skipped=disc.skipped,
                    total_lines=total_lines, total_chunks=total_chunks,
                    attack_classes=attack_classes)


def _project(settings: Settings, *, cells: int, files: int, skipped: int,
             total_lines: int, total_chunks: int, attack_classes: list[str],
             quiet: bool = False) -> dict:
    """Pure pricing/projection math — no filesystem work."""
    # Each cell may be hunted; confirmed findings add validate+trace calls (assume ~10%).
    est_calls = cells + int(cells * 0.1 * 2)
    est_tokens = est_calls * _TOKENS_PER_CELL
    # Price from the ACTUAL configured models, split input/output (output is 3-5x
    # dearer, so a single blended number materially under-states cost).
    in_price, out_price, price_note = _blended_price_per_1m(settings)
    est_in = est_tokens * _INPUT_SHARE
    est_out = est_tokens * (1.0 - _INPUT_SHARE)
    est_cost = (est_in / 1_000_000.0) * in_price + (est_out / 1_000_000.0) * out_price
    hunters = max(1, settings.concurrency.hunters)
    est_seconds = (cells * _SECONDS_PER_CELL) / hunters

    out = {
        "files": files,
        "skipped": skipped,
        "total_lines": total_lines,
        "chunks": total_chunks,
        "attack_classes": len(attack_classes),
        "cells": cells,
        "est_llm_calls": est_calls,
        "est_tokens": est_tokens,
        "est_cost_usd": round(est_cost, 2),
        "price_in_per_1m": round(in_price, 2),
        "price_out_per_1m": round(out_price, 2),
        "price_note": price_note,
        "est_minutes": round(est_seconds / 60.0, 1),
        "budget_ceiling_minutes": settings.budget.max_time_minutes,
        "budget_ceiling_usd": settings.budget.max_cost_usd,
    }
    if not quiet:
        _print(out, settings)
    return out


def _print(out: dict, settings: Settings) -> None:
    print("Dry-run estimate (no LLM calls were made):")
    print(f"  files in scope:        {out['files']}  (skipped {out['skipped']})")
    print(f"  total lines:           {out['total_lines']:,}")
    print(f"  chunks:                {out['chunks']:,}")
    print(f"  attack classes:        {out['attack_classes']}")
    print(f"  ledger cells:          {out['cells']:,}")
    print(f"  est. LLM calls:        {out['est_llm_calls']:,}")
    print(f"  est. tokens:           {out['est_tokens']:,}")
    print(f"  price used:            ${out['price_in_per_1m']}/1M in, "
          f"${out['price_out_per_1m']}/1M out")
    if out.get("price_note"):
        print(f"    ⚠ {out['price_note']}")
    print(f"  est. cost:             ${out['est_cost_usd']:,} "
          f"(ceiling ${out['budget_ceiling_usd']})")
    print(f"  est. wall-clock:       {out['est_minutes']} min "
          f"(ceiling {out['budget_ceiling_minutes']} min)")
    if out["est_cost_usd"] > out["budget_ceiling_usd"] > 0:
        print("  ⚠ estimated cost exceeds the configured ceiling; the loop will stop early.")
