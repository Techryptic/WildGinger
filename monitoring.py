# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""MLflow experiment tracking + tracing for every LLM call.

When ``monitoring.mlflow.enabled`` is true, EVERY LLM call — Bedrock *and*
Azure/GPT — is recorded to MLflow from the single choke point
:meth:`Audit.llm_call`, so both providers are covered by one hook:

  * **Metrics** — per-call tokens/cost/latency (step-indexed), cumulative totals,
    and **per-model + per-role cost/token roll-ups** so you can see exactly how much
    each model (and each stage) cost on a given scan.
  * **Tracing** (``tracing: true``) — one span per LLM call with the prompt in and
    the response out, populating the Traces / Overview / Sessions pages. All spans
    for a scan share a ``session`` id so a whole scan groups together.
  * **Prompt registry** (``log_prompts: true``) — the stage prompt templates are
    registered so the Prompts page is populated.

Design principles:
  * **Never break a scan.** If mlflow (or any sub-API) is missing or raises, we
    degrade to a silent no-op.
  * **Disabled by default.** ``build_logger`` returns a no-op unless the flag is on.
  * **Thread-safe.** Hunters/judges log concurrently; a lock guards the counters.
  * **Per-app separation.** The experiment can be set from the CLI (``--experiment``),
    auto-named per scanned target, or taken from config — each is auto-created.

Important: MLflow metric/param/tag KEYS may only contain alphanumerics plus
``_ - . / space`` — NOT colons. Bedrock model ids are ARNs full of colons, so we
key per-model roll-ups by a short config alias (e.g. ``opus48``) and sanitize all
keys defensively via :func:`_key`.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

_KEY_BAD = re.compile(r"[^A-Za-z0-9_\-./ ]+")


def _key(name: str) -> str:
    """Sanitize a string into a valid MLflow metric/param/tag key."""
    return _KEY_BAD.sub("_", str(name)).strip("_") or "unknown"


def _preview(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit} chars]"


class MLflowLogger:
    """Thread-safe MLflow metrics + tracing logger for LLM calls.

    A no-op instance (``enabled=False``) is safe to call and never imports mlflow.
    Use :func:`build_logger` to construct one from settings.
    """

    def __init__(self, *, enabled: bool = False, tracking_uri: str | None = None,
                 experiment: str = "WildGinger", run_name: str | None = None,
                 tracing: bool = True, log_io: bool = True, log_prompts: bool = True,
                 session_id: str | None = None, prompts: dict[str, str] | None = None,
                 model_aliases: dict[str, str] | None = None,
                 model_prices: dict[str, tuple[float, float]] | None = None,
                 tags: dict[str, Any] | None = None):
        self.enabled = enabled
        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._run_name = run_name
        self._tracing = tracing
        self._log_io = log_io
        self._log_prompts = log_prompts
        self._session_id = session_id or f"scan-{int(time.time())}"
        self._prompts = prompts or {}
        self._aliases = model_aliases or {}       # model_id → short alias
        self._prices = model_prices or {}         # alias → (in, out) per 1M
        self._tags = tags or {}
        self._mlflow = None
        self._active = False
        self._lock = threading.Lock()
        self._step = 0
        # Cumulative roll-ups.
        self._cum_calls = 0
        self._cum_input = 0
        self._cum_output = 0
        self._cum_cost = 0.0
        # Per-model and per-role breakdowns: name → {calls, input, output, cost}.
        self._by_model: dict[str, dict[str, float]] = {}
        self._by_role: dict[str, dict[str, float]] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Begin an MLflow run. Safe no-op when disabled or on any failure."""
        if not self.enabled:
            return
        try:
            import mlflow  # local import: only needed when enabled
            self._mlflow = mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment)  # auto-creates if missing
            mlflow.start_run(run_name=self._run_name)
            tags = {_key(k): str(v) for k, v in self._tags.items()}
            tags["mlflow.trace.session"] = self._session_id
            mlflow.set_tags(tags)
            # Log each model's configured price so the dashboard shows the rate used.
            try:
                params = {}
                for alias, (pin, pout) in self._prices.items():
                    params[_key(f"price_in_{alias}")] = pin
                    params[_key(f"price_out_{alias}")] = pout
                if params:
                    mlflow.log_params(params)
            except Exception:  # noqa: BLE001
                pass
            self._active = True
        except Exception:  # noqa: BLE001 — metrics must never break a scan
            self._mlflow = None
            self._active = False
            self.enabled = False
            return
        # Best-effort: register the stage prompt templates (Prompts page).
        if self._log_prompts and self._prompts:
            self._register_prompts()

    def _register_prompts(self) -> None:
        try:
            genai = getattr(self._mlflow, "genai", None)
            register = getattr(genai, "register_prompt", None)
            if register is None:
                return
            for name, template in self._prompts.items():
                try:
                    register(name=f"wildginger_{_key(name)}", template=template)
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass

    # ── per-call logging ────────────────────────────────────────────────────────
    def log_call(self, *, role: str, model_id: str, input_tokens: int,
                 output_tokens: int, cost_usd: float, latency_ms: float,
                 stop_reason: str | None, status: str,
                 system: str | None = None, prompt: str | None = None,
                 response: str | None = None) -> None:
        """Log one LLM call: metrics + (optional) a trace span with prompt/response.

        Driven from the provider-blind :meth:`Audit.llm_call` choke point, so it
        covers both Bedrock and Azure/GPT. Never raises. Metrics and tracing are
        guarded separately so a failure in one can't suppress the other.
        """
        if not (self.enabled and self._active and self._mlflow is not None):
            return
        alias = self._aliases.get(model_id, model_id)
        # ── metrics (guarded on its own) ──
        try:
            with self._lock:
                step = self._step
                self._step += 1
                self._cum_calls += 1
                self._cum_input += int(input_tokens)
                self._cum_output += int(output_tokens)
                self._cum_cost += float(cost_usd)
                self._accumulate(self._by_model, alias, input_tokens, output_tokens, cost_usd)
                self._accumulate(self._by_role, role, input_tokens, output_tokens, cost_usd)
                cum = {
                    "cumulative_calls": self._cum_calls,
                    "cumulative_input_tokens": self._cum_input,
                    "cumulative_output_tokens": self._cum_output,
                    "cumulative_cost_usd": round(self._cum_cost, 6),
                    _key(f"cost_usd_model_{alias}"): round(self._by_model[alias]["cost"], 6),
                    _key(f"calls_model_{alias}"): self._by_model[alias]["calls"],
                    _key(f"cost_usd_role_{role}"): round(self._by_role[role]["cost"], 6),
                }
            self._mlflow.log_metrics(
                {
                    "llm_input_tokens": int(input_tokens),
                    "llm_output_tokens": int(output_tokens),
                    "llm_total_tokens": int(input_tokens) + int(output_tokens),
                    "llm_cost_usd": round(float(cost_usd), 6),
                    "llm_latency_ms": round(float(latency_ms), 1),
                },
                step=step,
            )
            self._mlflow.log_metrics(cum, step=step)
        except Exception:  # noqa: BLE001
            step = self._step
        # ── tracing (independent guard so a metric error never kills the span) ──
        if self._tracing:
            self._trace_call(role=role, model_id=model_id, alias=alias, step=step,
                             input_tokens=input_tokens, output_tokens=output_tokens,
                             cost_usd=cost_usd, latency_ms=latency_ms,
                             stop_reason=stop_reason, status=status,
                             system=system, prompt=prompt, response=response)

    @staticmethod
    def _accumulate(bucket: dict[str, dict[str, float]], key: str,
                    input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        agg = bucket.setdefault(key, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
        agg["calls"] += 1
        agg["input"] += int(input_tokens)
        agg["output"] += int(output_tokens)
        agg["cost"] += float(cost_usd)

    def _trace_call(self, *, role: str, model_id: str, alias: str, step: int,
                    input_tokens: int, output_tokens: int, cost_usd: float,
                    latency_ms: float, stop_reason: str | None, status: str,
                    system: str | None, prompt: str | None, response: str | None) -> None:
        try:
            with self._mlflow.start_span(name=f"{role}:{alias}", span_type="LLM") as span:
                if self._log_io:
                    span.set_inputs({
                        "role": role,
                        "system": _preview(system or ""),
                        "prompt": _preview(prompt or ""),
                    })
                    span.set_outputs({"response": _preview(response or "")})
                span.set_attributes({
                    "model": alias,
                    "model_id": model_id,
                    "role": role,
                    "step": step,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "cost_usd": round(float(cost_usd), 6),
                    "latency_ms": round(float(latency_ms), 1),
                    "stop_reason": stop_reason,
                    "status": status,
                })
                updater = getattr(self._mlflow, "update_current_trace", None)
                if updater is not None:
                    try:
                        updater(metadata={"mlflow.trace.session": self._session_id})
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass

    # ── end ──────────────────────────────────────────────────────────────────────
    def end(self) -> None:
        """Log final totals + per-model/role breakdown, then end the run. No-op safe."""
        if not (self._active and self._mlflow is not None):
            return
        try:
            self._mlflow.log_metrics({
                "total_calls": self._cum_calls,
                "total_input_tokens": self._cum_input,
                "total_output_tokens": self._cum_output,
                "total_cost_usd": round(self._cum_cost, 6),
            })
            self._mlflow.log_params({
                "total_cost_usd": round(self._cum_cost, 6),
                "total_calls": self._cum_calls,
                "session_id": self._session_id,
            })
            breakdown = {
                "by_model": {k: _round_agg(v) for k, v in self._by_model.items()},
                "by_role": {k: _round_agg(v) for k, v in self._by_role.items()},
                "total_cost_usd": round(self._cum_cost, 6),
            }
            try:
                self._mlflow.log_dict(breakdown, "model_cost_breakdown.json")
            except Exception:  # noqa: BLE001
                pass
            self._mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._active = False


def _round_agg(agg: dict[str, float]) -> dict[str, float]:
    return {"calls": int(agg["calls"]), "input_tokens": int(agg["input"]),
            "output_tokens": int(agg["output"]), "cost_usd": round(agg["cost"], 6)}


def resolve_experiment(settings: Any, *, experiment_override: str | None = None,
                       target_name: str | None = None) -> str:
    """Resolve the MLflow experiment name.

    Precedence: CLI ``--experiment`` > per-target auto-name (if enabled) > config.
    """
    mlf = getattr(getattr(settings, "monitoring", None), "mlflow", None)
    default = getattr(mlf, "experiment", "WildGinger") if mlf else "WildGinger"
    if experiment_override:
        return experiment_override
    if mlf is not None and getattr(mlf, "experiment_per_target", False) and target_name:
        return target_name
    return default


def build_logger(settings: Any, *, experiment_override: str | None = None,
                 target_name: str | None = None,
                 prompts: dict[str, str] | None = None) -> MLflowLogger:
    """Construct an :class:`MLflowLogger` from settings.

    Returns a disabled (no-op) logger if ``monitoring.mlflow.enabled`` is false or
    the monitoring section is absent — so callers can always call the same methods.
    """
    mon = getattr(settings, "monitoring", None)
    mlf = getattr(mon, "mlflow", None) if mon is not None else None
    if mlf is None or not getattr(mlf, "enabled", False):
        return MLflowLogger(enabled=False)
    experiment = resolve_experiment(settings, experiment_override=experiment_override,
                                    target_name=target_name)
    run_name = getattr(mlf, "run_name", None) or (
        f"{target_name} {time.strftime('%Y-%m-%d %H:%M:%S')}" if target_name else None)
    # model_id → short alias, and alias → configured price, for readable per-model
    # cost keys (ARNs contain colons that MLflow keys reject).
    aliases: dict[str, str] = {}
    prices: dict[str, tuple[float, float]] = {}
    try:
        for alias, mp in settings.models.items():
            aliases[mp.model_id] = alias
            prices[alias] = (mp.price_per_1m_input, mp.price_per_1m_output)
    except Exception:  # noqa: BLE001
        pass
    tags: dict[str, Any] = {}
    try:
        tags = {
            "config_hash": settings.config_hash(),
            "prompts_version": settings.versioning.prompts_version,
            "target": target_name or "",
        }
    except Exception:  # noqa: BLE001
        pass
    return MLflowLogger(
        enabled=True,
        tracking_uri=getattr(mlf, "tracking_uri", None),
        experiment=experiment,
        run_name=run_name,
        tracing=getattr(mlf, "tracing", True),
        log_io=getattr(mlf, "log_io", True),
        log_prompts=getattr(mlf, "log_prompts", True),
        prompts=prompts,
        model_aliases=aliases,
        model_prices=prices,
        tags=tags,
    )
