# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Audit logging + cost/token ledger.

Every LLM call and every tool call is recorded so a run is fully explainable and
reproducible. Logs are written as JSONL (one event per line, fsync-friendly) under
``out_dir/audit/`` and, for LLM calls, mirrored into the ``llm_calls`` table for
cost roll-ups.

Cost estimation is intentionally simple and configurable: the provider returns token
usage per call; we multiply by a per-model price (USD per 1,000,000 tokens — see
:func:`estimate_cost`). Prices live in ``PRICES``, populated from ``config.yaml`` at
startup via :func:`set_prices`; unknown models default to 0.0 so a missing price
never crashes a scan (it just under-reports cost, which the dry-run estimate and
audit log make visible).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import fileio

# USD per 1,000,000 tokens, keyed by model_id:
#   (input, output, cache_read, cache_write)
# Populated from config at startup via set_prices(). Unknown model_id → all zeros.
# Two-tuples are still accepted for backward compatibility (cache prices default to
# the input price, which over-counts slightly rather than under-counting — the safe
# direction for a cost kill-switch).
PRICES: dict[str, tuple[float, ...]] = {}
DEFAULT_PRICE = (0.0, 0.0, 0.0, 0.0)


def set_prices(prices: dict[str, tuple[float, ...]]) -> None:
    """Install per-model $/1M-token prices from config.

    Each value is ``(input, output)`` or ``(input, output, cache_read, cache_write)``.
    """
    global PRICES
    PRICES = dict(prices)


def _price_tuple(model_id: str) -> tuple[float, float, float, float]:
    """Normalize a configured price entry to 4 components.

    A missing cache price falls back to the INPUT price. Cache reads are really ~10%
    of input and writes ~125%, but guessing cheap would let the kill-switch
    under-count real spend; falling back to full input price cannot.
    """
    raw = PRICES.get(model_id) or DEFAULT_PRICE
    vals = list(raw) + [0.0] * (4 - len(raw))
    in_price, out_price, cache_read, cache_write = vals[:4]
    if cache_read <= 0.0:
        cache_read = in_price
    if cache_write <= 0.0:
        cache_write = in_price
    return in_price, out_price, cache_read, cache_write


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """Cost in USD. Prices are per 1,000,000 tokens.

    Cache tokens are priced SEPARATELY from ``input_tokens``: the contract with every
    provider is that the two are DISJOINT by the time they reach here, so they must both
    be billed or enabling prompt caching would silently under-report spend and the hard
    cost ceiling would fire late.

    That contract is native to Bedrock Converse (``inputTokens`` excludes
    ``cacheRead/WriteInputTokens``) but NOT to OpenAI-compatible APIs, where
    ``prompt_tokens`` includes ``cached_tokens`` — so the GPT provider subtracts the
    cached count before handing them over (see ``providers/openai_gpt.py``). Do not
    "fix" the double-billing here: this function cannot tell the two conventions apart,
    and compensating for one would corrupt the other.
    """
    in_price, out_price, cache_read_price, cache_write_price = _price_tuple(model_id)
    return (
        (input_tokens / 1_000_000.0) * in_price
        + (output_tokens / 1_000_000.0) * out_price
        + (cache_read_tokens / 1_000_000.0) * cache_read_price
        + (cache_write_tokens / 1_000_000.0) * cache_write_price
    )


class BudgetExceeded(RuntimeError):
    """Raised the instant cumulative cost crosses the hard ceiling (kill-switch)."""



@dataclass
class CostLedger:
    """Thread-safe running totals for budget enforcement."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int, cost_usd: float,
            cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> None:
        self.add_and_check(input_tokens, output_tokens, cost_usd,
                           cache_read_tokens=cache_read_tokens,
                           cache_write_tokens=cache_write_tokens, ceiling=0.0)

    def add_and_check(self, input_tokens: int, output_tokens: int, cost_usd: float,
                      cache_read_tokens: int = 0, cache_write_tokens: int = 0,
                      *, ceiling: float = 0.0) -> tuple[float, bool]:
        """Record a call's spend and decide the ceiling verdict ATOMICALLY.

        Returns ``(cumulative_cost_usd, crossed_the_ceiling)``.

        The verdict MUST be computed inside the same critical section as the ``+=``.
        The kill-switch used to add under this lock and then re-read ``cost_usd``
        unlocked, dozens of lines later, *after* a file flush + a single-writer DB
        round-trip + up to three MLflow network hops. Two things went wrong there:
        the thread that actually crossed the ceiling could have its verdict computed
        from a total that other threads had already moved, and — worse — anything
        that raised in that tail (a DB write timeout) skipped the check entirely, so
        the ceiling was silently not enforced for the very call that broke it.
        Deciding here makes the verdict a pure function of this call's own spend.

        ``ceiling <= 0`` disables the check (the documented "0 = no limit").
        """
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read_tokens
            self.cache_write_tokens += cache_write_tokens
            self.cost_usd += cost_usd
            self.calls += 1
            total = self.cost_usd
        return total, bool(ceiling > 0 and total >= ceiling)

    def over_ceiling(self, ceiling: float) -> tuple[float, bool]:
        """``(cumulative_cost_usd, already_at_or_over_ceiling)``; 0 = no limit.

        Read under the lock so the total and the verdict agree. Used as a cheap
        PRE-call gate: without it, the ceiling can only ever be detected after a call
        has been billed, so every request already in flight when the ceiling is
        crossed is paid for in full (measured overshoot: ~+$11 at 12 hunters,
        ~+$23 with the judge pool also running).
        """
        with self._lock:
            total = self.cost_usd
        return total, bool(ceiling > 0 and total >= ceiling)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cost_usd": round(self.cost_usd, 4),
            }


class Audit:
    """Structured JSONL audit logger with a cost ledger.

    Thread-safe: multiple hunter threads append concurrently. Each event is one
    JSON line, flushed immediately so a mid-run kill leaves a readable log.
    """

    def __init__(self, out_dir: Path, *, restrict_permissions: bool = True,
                 hard_cost_ceiling: float = 0.0, mlflow=None):
        self.audit_dir = Path(out_dir) / "audit"

        self.audit_dir.mkdir(parents=True, exist_ok=True)
        if restrict_permissions:
            fileio.restrict_permissions(self.audit_dir)
        self._path = self.audit_dir / "events.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")
        if restrict_permissions:
            fileio.restrict_permissions(self._path)
        self._lock = threading.Lock()
        self.cost = CostLedger()
        # Hard kill-switch: once cumulative cost >= this, the NEXT llm_call raises
        # BudgetExceeded so the whole scan stops immediately (0 = disabled).
        self.hard_cost_ceiling = float(hard_cost_ceiling or 0.0)
        # Optional MLflow logger (monitoring.mlflow.enabled). This is the single
        # choke point every LLM call flows through, so setting it here covers BOTH
        # Bedrock and Azure/GPT providers with one hook. A disabled/no-op logger is
        # fine; log_call never raises.
        self.mlflow = mlflow
        # Wall-clock deadline for the whole run (see set_deadline). None = no limit.
        self._deadline: float | None = None
        # How many calls came back with a provider-reported cost that materially
        # disagreed with our configured price. Surfaced in the run summary: a wrong price
        # means a wrong ledger, and therefore a kill-switch that fires at the wrong time.
        self.price_mismatches = 0


    # ── core ──────────────────────────────────────────────────────────────────
    def event(self, kind: str, **fields: Any) -> None:

        """Write a generic audit event."""
        rec = {"ts": time.time(), "kind": kind, **fields}
        line = json.dumps(rec, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def llm_call(
        self,
        *,
        role: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        latency_ms: float,
        stop_reason: str | None,
        status: str,
        db=None,
        system: str | None = None,
        prompt: str | None = None,
        response: str | None = None,
        reported_cost_usd: float | None = None,
    ) -> float:
        """Record one LLM call; returns its estimated cost and updates the ledger.

        ``system``/``prompt``/``response`` are optional and only used to enrich the
        MLflow trace span (prompt-in / response-out); they are never written to the
        JSONL audit log or the SQLite ledger.

        ``cache_read_tokens``/``cache_write_tokens`` are prompt-cache usage counts
        (0 when the provider/model doesn't report them). They are billed separately
        from ``input_tokens``, so they ARE included in the cost math.

        ``reported_cost_usd`` is the provider's OWN figure for this call, when it gives
        one (OpenRouter does). It is logged beside our computed cost but deliberately does
        NOT feed the ledger or the kill-switch: the dry-run estimate has to project a cost
        before any call exists, preflight refuses to start on an unpriced model, and a
        ceiling that depended on a field most providers omit would be a ceiling that
        silently stops working. Having both numbers in the log is what makes a configured
        price checkable against what was actually charged — including for a broker like
        OpenRouter, which re-routes between upstreams at different prices.
        """
        cost = estimate_cost(model_id, input_tokens, output_tokens,
                             cache_read_tokens, cache_write_tokens)
        # Decide the ceiling verdict in the SAME critical section as the spend, before
        # any bookkeeping runs. Everything after this point is best-effort reporting:
        # it can fail, but it can no longer change whether the kill-switch fires.
        total_cost, over_ceiling = self.cost.add_and_check(
            input_tokens, output_tokens, cost,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            ceiling=self.hard_cost_ceiling,
        )
        event_fields: dict[str, Any] = {
            "role": role,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": round(latency_ms, 1),
            "stop_reason": stop_reason,
            "status": status,
        }
        if reported_cost_usd is not None:
            event_fields["provider_reported_cost_usd"] = round(
                float(reported_cost_usd), 6)
            # Surface a material disagreement between the configured price and the bill.
            # A silently wrong price is a silently wrong ledger, and therefore a
            # kill-switch that fires at the wrong time in either direction.
            if cost > 0 and abs(reported_cost_usd - cost) > max(0.01, cost * 0.25):
                event_fields["cost_mismatch"] = True
                self.price_mismatches += 1
        self.event("llm_call", **event_fields)
        if db is not None:
            def _ins(cur):
                cur.execute(
                    "INSERT INTO llm_calls(role, model_id, input_tokens, output_tokens, "
                    "cache_read_tokens, cache_write_tokens, "
                    "cost_usd, latency_ms, stop_reason, status, ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?);",
                    (role, model_id, input_tokens, output_tokens,
                     cache_read_tokens, cache_write_tokens, cost,
                     latency_ms, stop_reason, status, time.time()),
                )
            try:
                db.write(_ins)
            except Exception as exc:  # noqa: BLE001
                # The `llm_calls` table is a convenience MIRROR: the durable record is
                # the JSONL line above and the authoritative total is the in-memory
                # ledger, both already written. Letting this raise cost us the whole
                # conversation: the RuntimeError from a writer-queue timeout is neither
                # a ProviderError (so agent._turn_with_retry ignores it) nor a
                # _TurnFailed (so Agent.run ignores it), so it escaped to the
                # orchestrator, the cell was marked FAILED, and every turn of an
                # already-paid-for conversation was thrown away and re-hunted.
                self.event("llm_ledger_write_failed", role=role, model_id=model_id,
                           error=str(exc))

        # Mirror to MLflow (model/tokens/cost/latency/stage + prompt/response for
        # tracing) — covers Bedrock AND Azure/GPT because every provider call funnels
        # through here. Never raises.
        if self.mlflow is not None:
            self.mlflow.log_call(
                role=role, model_id=model_id, input_tokens=input_tokens,
                output_tokens=output_tokens, cost_usd=cost, latency_ms=latency_ms,
                stop_reason=stop_reason, status=status,
                system=system, prompt=prompt, response=response,
            )

        # HARD KILL-SWITCH: stop the entire scan the moment we cross the ceiling.
        # `over_ceiling` was decided atomically above, so no failure in the
        # bookkeeping between here and there can suppress it.
        if over_ceiling:
            self.event("budget_kill_switch", cost_usd=round(total_cost, 4),
                       ceiling=self.hard_cost_ceiling)
            raise BudgetExceeded(
                f"cost ${total_cost:.2f} reached the hard ceiling "
                f"${self.hard_cost_ceiling:.2f} — stopping immediately.")
        return cost

    def set_deadline(self, deadline_epoch: float | None) -> None:
        """Install the run's wall-clock deadline so ``check_ceiling`` can enforce it.

        ``budget.max_time_minutes`` is documented as a "hard wall-clock stop", but it was
        only ever consulted BETWEEN passes — there was no mid-pass, mid-cell or pre-call
        time check anywhere. With a large turn budget and thousands of cells a single pass
        can run for many hours, so the only thing that actually stopped a run was the cost
        ceiling. Feeding the deadline through the same gate the cost ceiling uses makes it
        a real limit.
        """
        self._deadline = float(deadline_epoch) if deadline_epoch else None

    def check_ceiling(self) -> None:
        """Raise :class:`BudgetExceeded` if a ceiling has ALREADY been crossed.

        A cheap pre-call gate for callers that are about to spend money. The
        post-call check can only ever notice the ceiling once a request has been
        billed, so without this every call in flight at the moment of crossing is
        paid for in full — the overshoot scales with ``concurrency.hunters``.
        """
        deadline = getattr(self, "_deadline", None)
        if deadline is not None and time.time() >= deadline:
            raise BudgetExceeded(
                "wall-clock limit (budget.max_time_minutes) reached — refusing to start "
                "another call.")
        total, over = self.cost.over_ceiling(self.hard_cost_ceiling)
        if over:
            raise BudgetExceeded(
                f"cost ${total:.2f} is at/over the hard ceiling "
                f"${self.hard_cost_ceiling:.2f} — refusing to start another call.")

    def tool_call(self, *, agent_id: str, tool: str, args: dict[str, Any],

                  result_bytes: int, truncated: bool) -> None:
        """Record one navigation tool call."""
        self.event(
            "tool_call",
            agent_id=agent_id,
            tool=tool,
            args=args,
            result_bytes=result_bytes,
            truncated=truncated,
        )

    def anomaly(self, what: str, **fields: Any) -> None:
        """Record a security-relevant anomaly (injection attempt, redaction bypass)."""
        self.event("anomaly", what=what, **fields)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "Audit":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
