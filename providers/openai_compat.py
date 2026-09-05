# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat-completions provider — local models and OpenRouter.

One implementation covers every endpoint that speaks ``POST {base_url}/chat/completions``
in OpenAI's shape:

  * **local models** — Ollama, LM Studio, vLLM, llama.cpp, LiteLLM (usually no API key,
    plaintext loopback, slow, and best served a couple of requests at a time);
  * **OpenRouter** — a broker in front of many vendors, with extra routing controls and a
    reported per-call cost (see ``openrouter=True``);
  * any other hosted OpenAI-compatible gateway.

Standard library only (``urllib``), matching ``openai_gpt.py`` — no third-party HTTP
client is introduced by this file.

WHY THIS IS NOT A SUBCLASS OF ``openai_gpt``. That provider targets an explicitly
configured gateway, which is Azure-deployment-shaped in three ways this one cannot
inherit: the model name goes in the URL (``base_url.format(deployment=...)``) rather than the request
body, no ``model`` field is sent at all, and it always sends ``max_completion_tokens``,
which Ollama, llama.cpp and older vLLM reject outright.

TOOL USE. Two modes, chosen per model in config (``tool_mode``):

  * ``native`` — OpenAI's ``tools`` / ``tool_calls`` fields. Preferred: the model is
    trained on this shape, it can request several tools in one turn, and there is no
    prose to mis-parse.
  * ``react``  — the ``ACTION: name({...})`` text protocol from :mod:`._openai_common`, for
    endpoints with no tool API (some local builds ignore or 400 on ``tools``).

Either way the agent above this layer sees the same normalized ``tool_use`` blocks, so
the hunt loop, proof-of-read and thoroughness scoring behave identically to Bedrock.

EGRESS. ``base_url`` is validated against ``governance.allowed_model_endpoints`` twice
before anything is sent: at config load (``Settings._model_endpoints_are_allowed``) and
in the factory (``providers.base.build_provider``). OpenRouter's default API root is
resolved before those checks and requires an allowlist entry too. Other compatible
endpoints require an explicit ``base_url``.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from ..settings import OpenRouterRouting, resolve_model_base_url
from ._openai_common import (
    _is_temperature_rejection,
    _parse_action,
    _tools_instructions,
    malformed_action_hint,
)
from .base import LLMProvider, LLMResponse, Message, ProviderError, ToolSpec, ToolUse

# Wall-clock ceiling for ONE generate() call's retries. Same value as the other two
# providers so a throttle storm behaves identically whichever vendor serves the role, and
# so it stays well inside the 600s cell lease (a retrying call must not outlive the claim
# it is working under). Pinned by the cross-provider parity tests.
_RETRY_DEADLINE_SECONDS = 240

# HTTP statuses worth retrying. 5xx and 429 are transient; 408/409/425 are transport-ish.
# 52x are Cloudflare's, which sit in front of several hosted OpenAI-compatible APIs
# (OpenRouter included) and are routinely transient.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504,
                     520, 521, 522, 523, 524}

# Statuses that a retry can never fix, with the reason a human needs. Retrying these just
# burns the turn budget on a guaranteed failure — the mistake that once spent ~50 minutes
# of backoff per cell on a bad API key.
_FATAL_STATUS_HINTS = {
    401: "authentication rejected — check the API key in the env var named by "
         "models.<name>.api_key_env",
    403: "forbidden — the key is valid but not permitted to use this model",
    402: "insufficient credits on the account (OpenRouter returns 402 for this); "
         "top up or switch models — retrying cannot help",
    404: "no endpoint for this model id (OpenRouter: no provider serves it with the "
         "parameters requested; local: the model is not pulled/loaded)",
    422: "the endpoint rejected the request body as invalid",
}

# Error codes/messages that mean "this model has no tool-calling support". Used to
# self-heal ONCE onto the ReAct protocol instead of failing every cell of the run.
_NO_TOOL_SUPPORT_MARKERS = (
    "does not support tools",
    "does not support tool",
    "tool use is not supported",
    "tool calling is not supported",
    "no endpoints found that support tool use",
    "tools is not supported",
    "unsupported parameter: 'tools'",
    "unknown field: tools",
)


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate — retry only what a retry can actually fix."""
    return isinstance(exc, ProviderError) and exc.retryable


def _mentions_no_tool_support(message: str) -> bool:
    low = (message or "").lower()
    return any(marker in low for marker in _NO_TOOL_SUPPORT_MARKERS)


class OpenAICompatProvider(LLMProvider):
    """Chat-completions provider for any OpenAI-compatible endpoint."""

    def __init__(self, *, model_id: str, base_url: str | None = None,
                 api_key_env: str | None = None, send_temperature: bool = True,
                 tool_mode: str = "native", token_param: str = "max_tokens",
                 timeout_seconds: int = 300,
                 max_concurrent_requests: int | None = None,
                 routing: Any = None, openrouter: bool = False,
                 extra_headers: dict[str, str] | None = None):
        self.model_id = model_id
        self._openrouter = bool(openrouter)
        resolved = resolve_model_base_url(
            "openrouter" if self._openrouter else "openai_compat", base_url)
        if not resolved:
            # No guessed default. Every destination this tool may reach is written down
            # in governance.allowed_model_endpoints; inventing one here would route
            # source code somewhere nobody authorized.
            raise ValueError(
                f"provider 'openai_compat' needs models.<name>.base_url (e.g. "
                f"http://localhost:11434/v1 for Ollama), and that URL must be listed in "
                f"governance.allowed_model_endpoints. Model: {model_id}")
        self._url = resolved.rstrip("/")
        if not self._url.endswith("/chat/completions"):
            self._url = f"{self._url}/chat/completions"
        # Only the NAME is configured; the secret itself never touches config.yaml.
        self._api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self._api_key_env = api_key_env
        self._token_param = token_param
        self._timeout = max(1, int(timeout_seconds))
        self._routing = routing
        self._extra_headers = dict(extra_headers or {})
        # o-series / GPT-5-class deployments reject an explicit temperature; some local
        # servers reject it too. Self-heals on the first rejection, like Bedrock.
        self._omit_temperature = not send_temperature
        # 'native' | 'react' | 'none'. `_tools_unsupported` flips native → react for the
        # rest of the run if the endpoint turns out to have no tool API.
        self._tool_mode = tool_mode
        self._tools_unsupported = False
        # One provider instance is shared by every hunter and judge thread, so the
        # self-heal flags below are written under a lock (an unsynchronized lost update
        # turned a HEALABLE error into a hard failure for every other in-flight thread).
        self._state_lock = threading.Lock()
        # A local server is usually ONE model on ONE GPU: 12 hunters aimed at it will
        # queue in the kernel, time out, or OOM. This bounds in-flight requests
        # regardless of concurrency.hunters. None = no cap (hosted APIs).
        self._slots = (threading.BoundedSemaphore(int(max_concurrent_requests))
                       if max_concurrent_requests else None)

    # ── public ────────────────────────────────────────────────────────────────
    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """One turn, with retry/backoff on transient failures.

        The retry policy matches the Bedrock and Apigee providers exactly: full jitter
        (so N hunters throttled by one quota don't retry in lockstep) and bounded in BOTH
        attempts and wall-clock, because attempt-bounded retries alone can block far past
        the cell lease.
        """
        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=(stop_after_attempt(6) | stop_after_delay(_RETRY_DEADLINE_SECONDS)),
            reraise=True,
        )
        def _call() -> LLMResponse:
            return self._chat_once(system=system, messages=messages,
                                   max_tokens=max_tokens, temperature=temperature,
                                   tools=tools)

        return _call()

    # ── one request ───────────────────────────────────────────────────────────
    def _chat_once(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None,
    ) -> LLMResponse:
        with self._state_lock:
            mode = self._effective_tool_mode()
        use_native = bool(tools) and mode == "native"
        use_react = bool(tools) and mode == "react"

        sys_prompt = system
        if use_react:
            sys_prompt = (system or "") + "\n\n" + _tools_instructions(tools or [])
        chat = self._to_chat_messages(sys_prompt, messages, native_tools=use_native)

        def _payload(include_temperature: bool, include_tools: bool) -> bytes:
            body: dict[str, Any] = {"model": self.model_id, "messages": chat}
            # OpenAI/Azure renamed this; Ollama, llama.cpp and older vLLM only know
            # `max_tokens`. Configured per model rather than guessed.
            body[self._token_param] = max_tokens
            if include_temperature and temperature is not None:
                body["temperature"] = temperature
            if include_tools and tools:
                body["tools"] = [_tool_to_wire(t) for t in tools]
                body["tool_choice"] = "auto"
            if self._openrouter:
                # Ask for the real per-call cost so the ledger can be reconciled against
                # what OpenRouter actually charged (see LLMResponse.reported_cost_usd).
                body["usage"] = {"include": True}
                body["provider"] = OpenRouterRouting().to_body()
                if self._routing is not None:
                    body["provider"].update(self._routing.to_body()
                                            if hasattr(self._routing, "to_body")
                                            else dict(self._routing))
            return json.dumps(body).encode("utf-8")

        t0 = time.time()
        try:
            doc = self._post(_payload(not self._omit_temperature, use_native))
        except ProviderError as exc:
            healed = self._heal(exc, used_native=use_native)
            if healed == "tools":
                # Rebuild the ENTIRE request, don't just drop the `tools` array: the
                # messages were flattened for native tool use and the system prompt has
                # no ACTION instructions, so retrying the same body would leave the model
                # with neither tools nor any idea that it has any — it would answer from
                # the chunk alone, score zero successful tool calls, and the cell would be
                # re-hunted for a reason nothing reports. `_tools_unsupported` is already
                # set, so this recurses exactly once and comes back down the react path.
                return self._chat_once(system=system, messages=messages,
                                       max_tokens=max_tokens, temperature=temperature,
                                       tools=tools)
            if healed == "temperature":
                doc = self._post(_payload(False, use_native))
            else:
                raise
        latency_ms = (time.time() - t0) * 1000.0
        return _parse_chat_response(doc, latency_ms, tools_offered=bool(tools))

    def _effective_tool_mode(self) -> str:
        """Tool mode for this request, honouring a discovered lack of tool support."""
        if self._tool_mode == "native" and self._tools_unsupported:
            return "react"
        return self._tool_mode

    def _heal(self, exc: ProviderError, *, used_native: bool) -> str | None:
        """Record a healable rejection and say which one it was.

        Returns ``"temperature"``, ``"tools"`` or None. Two rejections are worth healing;
        both are permanent properties of the endpoint that can only be discovered by
        asking, and both would otherwise fail every cell of the run:

          * ``temperature`` unsupported — omit it from here on;
          * ``tools`` unsupported — fall back to the ReAct text protocol for the rest of
            the run, so a model without a tool API still navigates instead of scoring
            zero thoroughness on every cell.

        Anything else returns None and the caller re-raises with the classification
        intact, so tenacity can still retry a genuinely transient failure.

        The flags are read AND written under one lock: unsynchronized, a lost update
        turned a HEALABLE error into a hard failure for every other in-flight thread
        (thread A set the flag while B was mid-check, so B fell through to the raise —
        and a 400 is non-retryable, so B's cell FAILED outright).
        """
        message = str(exc)
        with self._state_lock:
            if used_native and not self._tools_unsupported \
                    and _mentions_no_tool_support(message):
                self._tools_unsupported = True
                return "tools"
            if _is_temperature_rejection(message) and not self._omit_temperature:
                self._omit_temperature = True
                return "temperature"
        return None


    def _post(self, data: bytes) -> dict[str, Any]:
        """POST to the endpoint; raise a classified :class:`ProviderError` on failure."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        elif self._api_key_env:
            # Configured to use a key, but the variable is empty. Say so precisely: the
            # alternative is a 401 from the far end that reads like a permissions problem.
            raise ProviderError(
                f"environment variable {self._api_key_env} is empty, so no API key can "
                f"be sent for model '{self.model_id}'. Export it, or set "
                f"api_key_env: null for an endpoint that needs no key.",
                retryable=False)
        if self._openrouter:
            # OpenRouter attribution header: identifying the caller is good manners for
            # a shared broker and shows up in the account's activity log.
            headers.setdefault("X-Title", "WildGinger")
        headers.update(self._extra_headers)

        req = urllib.request.Request(self._url, data=data, method="POST",
                                     headers=headers)
        acquired = False
        try:
            if self._slots is not None:
                self._slots.acquire()
                acquired = True
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # Read the body: it carries the reason (unsupported parameter, no tool
            # support, out of credits), which the self-heal paths need to see.
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                detail = ""
            hint = _FATAL_STATUS_HINTS.get(exc.code)
            retryable = exc.code in _RETRYABLE_STATUS and hint is None
            suffix = f" — {hint}" if hint else ""
            raise ProviderError(
                f"HTTP {exc.code} {exc.reason}{suffix}: {detail}", retryable=retryable
            ) from exc
        except urllib.error.URLError as exc:
            # DNS/connection/timeout — genuinely transient. A local server that is not
            # running looks like this, so name that possibility.
            raise ProviderError(
                f"URLError contacting {self._url}: {exc.reason} (is the endpoint "
                f"running?)", retryable=True) from exc
        except Exception as exc:  # noqa: BLE001 — socket timeouts etc.
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=True) from exc
        finally:
            if acquired and self._slots is not None:
                self._slots.release()

        try:
            doc = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"invalid JSON from {self._url}: {exc}",
                                retryable=False) from exc
        if not isinstance(doc, dict):
            raise ProviderError(f"unexpected response shape from {self._url}: "
                                f"{type(doc).__name__}", retryable=False)
        _raise_for_error_object(doc)
        if not doc.get("choices"):
            # No completion and no error object either. Falling through would produce an
            # empty reply, which upstream is indistinguishable from a model that chose to
            # say nothing: the cell scores zero thoroughness, goes SHALLOW and is
            # re-hunted, with no record of the real cause. Name it instead.
            raise ProviderError(
                f"response from {self._url} contained no 'choices' (keys: "
                f"{sorted(doc)[:6]}) — the endpoint is not returning completions in the "
                f"OpenAI shape", retryable=False)
        return doc

    # ── message mapping ───────────────────────────────────────────────────────
    @staticmethod
    def _to_chat_messages(system: str, messages: list[Message], *,
                          native_tools: bool) -> list[dict[str, Any]]:
        """Map our normalized blocks onto OpenAI chat messages.

        In NATIVE mode the mapping is structural: a ``tool_use`` block becomes an
        assistant message carrying ``tool_calls``, and each ``tool_result`` becomes its
        own ``role: "tool"`` message keyed by ``tool_call_id``. That id linkage is what
        lets the model match a result to the call it made — flattening tool traffic into
        prose (which the ReAct path must do) loses it.

        In REACT mode everything is flattened to text, exactly as the Apigee provider
        does, so the transcript still reads coherently to a model with no tool API.
        """
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if native_tools:
                out.extend(_native_blocks_to_messages(m))
                continue
            parts: list[str] = []
            for b in m.content:
                btype = b.get("type")
                if btype == "text":
                    parts.append(b.get("text", ""))
                elif btype == "tool_use":
                    # Echo the model's own ACTION so the conversation stays coherent.
                    parts.append(
                        f"ACTION: {b.get('name')}({json.dumps(b.get('input', {}))})")
                elif btype == "tool_result":
                    parts.append(f"OBSERVATION:\n{b.get('text', '')}")
            role = "assistant" if m.role == "assistant" else "user"
            out.append({"role": role, "content": "\n".join(parts)})
        return out


# ── module-level helpers (pure; unit-testable without a server) ────────────────

def _tool_to_wire(tool: ToolSpec) -> dict[str, Any]:
    """Our ToolSpec → OpenAI's function-tool declaration."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _native_blocks_to_messages(m: Message) -> list[dict[str, Any]]:
    """One normalized Message → one or more OpenAI messages (native tool shape)."""
    texts = [b.get("text", "") for b in m.content if b.get("type") == "text"]
    tool_uses = [b for b in m.content if b.get("type") == "tool_use"]
    tool_results = [b for b in m.content if b.get("type") == "tool_result"]
    out: list[dict[str, Any]] = []

    if m.role == "assistant":
        msg: dict[str, Any] = {"role": "assistant",
                               "content": "\n".join(t for t in texts if t) or None}
        if tool_uses:
            msg["tool_calls"] = [
                {
                    "id": b.get("tool_use_id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": b.get("name", ""),
                                 "arguments": json.dumps(b.get("input", {}))},
                }
                for b in tool_uses
            ]
        out.append(msg)
        return out

    # A user turn carrying tool results: each result is its own `role: tool` message,
    # and any accompanying prose follows as a normal user message.
    for b in tool_results:
        out.append({
            "role": "tool",
            "tool_call_id": b.get("tool_use_id", ""),
            "content": b.get("text", ""),
        })
    joined = "\n".join(t for t in texts if t)
    if joined or not tool_results:
        out.append({"role": "user", "content": joined})
    return out


def _raise_for_error_object(doc: dict[str, Any]) -> None:
    """Raise if the body carries an ``error`` instead of a completion.

    OpenAI-compatible gateways do not always use the HTTP status for failures: several
    (OpenRouter among them) can answer **200** with ``{"error": {...}}`` and no
    ``choices``. Left alone that becomes an empty reply, which upstream looks like a
    model that said nothing: the cell scores zero thoroughness, is marked SHALLOW and is
    re-hunted — a paid round trip whose real cause is never reported anywhere.
    """
    err = doc.get("error")
    if not err:
        return
    if isinstance(err, dict):
        message = str(err.get("message") or err)
        code = err.get("code")
    else:
        message, code = str(err), None
    hint = _FATAL_STATUS_HINTS.get(code) if isinstance(code, int) else None
    retryable = isinstance(code, int) and code in _RETRYABLE_STATUS and hint is None
    suffix = f" — {hint}" if hint else ""
    raise ProviderError(f"endpoint returned an error (code={code}){suffix}: {message}",
                        retryable=retryable)


# Phrases used when a deployment rejects an explicit temperature live in
# providers/_openai_common.py — the same detector the Apigee provider uses, so the two
# self-heal paths cannot drift apart.


def _parse_chat_response(doc: dict[str, Any], latency_ms: float,
                         *, tools_offered: bool = False) -> LLMResponse:
    """Normalize an OpenAI-compatible response into :class:`LLMResponse`.

    ``tools_offered`` says whether this turn had tools available, in EITHER mode. It
    gates the ACTION-line fallback below, which is deliberately not limited to
    ``tool_mode: react``: a weaker model sent a native ``tools`` array sometimes ignores
    it and answers with the ReAct convention it learned in training instead. Reading that
    as prose would waste the turn and, worse, look like a model that navigated nothing —
    so if tools were on the table and no ``tool_calls`` came back, an ACTION line is
    still honoured. The pattern is anchored and requires ``name({``, so ordinary prose
    cannot trip it.
    """
    choices = doc.get("choices") or []
    text = ""
    finish_reason = ""
    raw_tool_calls: list[dict[str, Any]] = []
    reasoning = None
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            # Some servers return content as a list of parts rather than a string.
            text = "".join(part.get("text", "") for part in text
                           if isinstance(part, dict))
        finish_reason = (choices[0] or {}).get("finish_reason") or ""
        raw_tool_calls = [tc for tc in (msg.get("tool_calls") or [])
                          if isinstance(tc, dict)]
        # A reasoning model's thinking, under whichever key this server uses: GLM sends
        # `reasoning`, vLLM and DeepSeek send `reasoning_content`. Captured for
        # DIAGNOSTICS only and deliberately NOT appended to `text` — see
        # LLMResponse.reasoning_text for why merging it would be dangerous.
        for key in ("reasoning", "reasoning_content"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                reasoning = value
                break
    usage = doc.get("usage") or {}

    # Truncation must be visible. `finish_reason: "length"` means the reply was cut at
    # the token cap, so any finding after the cut is gone; reporting it as a clean
    # completion let find_last_json() salvage one object from a partial array and mark
    # the cell ANALYZED with a silent hole in it.
    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    norm_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    tool_uses: list[ToolUse] = []

    if raw_tool_calls:
        parsed_any = False
        bad_names: list[str] = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            args, ok = _decode_arguments(fn.get("arguments"))
            if not ok:
                bad_names.append(name)
                continue
            tool_uses.append(ToolUse(tool_use_id=call_id, name=name, input=args))
            norm_blocks.append({"type": "tool_use", "tool_use_id": call_id,
                                "name": name, "input": args})
            parsed_any = True
        if parsed_any:
            stop_reason = "tool_use"
        if bad_names and not parsed_any:
            # `arguments` is a JSON STRING the model generates, so it can be truncated or
            # malformed. Dispatching `{}` would hand the model a "'path' is required"
            # error it cannot act on; tell it what actually went wrong instead.
            hint = malformed_action_hint(bad_names[0])
            text = (text + "\n\n" + hint) if text else hint
            norm_blocks = [{"type": "text", "text": text}]
            stop_reason = "end_turn"
    elif tools_offered and text:
        parsed = _parse_action(text)
        if parsed is not None:
            name, args, parse_ok = parsed
            if parse_ok:
                tu_id = f"react-{uuid.uuid4().hex[:12]}"
                tool_uses.append(ToolUse(tool_use_id=tu_id, name=name, input=args))
                stop_reason = "tool_use"
                # KEEP the model's prose alongside the tool_use block: dropping it
                # deleted the reasoning from history every tool turn, so the model
                # re-derived the same chain of thought next turn at full input cost.
                norm_blocks = ([{"type": "text", "text": text}] if text.strip() else [])
                norm_blocks.append({"type": "tool_use", "tool_use_id": tu_id,
                                    "name": name, "input": args})
            else:
                stop_reason = "end_turn"
                hint = malformed_action_hint(name)
                norm_blocks = [{"type": "text", "text": text + "\n\n" + hint}]
                text = norm_blocks[0]["text"]

    # CACHED TOKENS ARE A SUBSET OF prompt_tokens on this API family (unlike Bedrock
    # Converse, where the counts are disjoint). audit.estimate_cost prices them as
    # disjoint, so reporting the raw prompt_tokens here would bill the cached prefix
    # twice — at the full input price, since a missing cache price falls back to input.
    # That inflated the ledger ~1.7x on a cache-heavy workload and made the hard cost
    # ceiling fire at a fraction of real spend.
    cached = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    uncached_input = max(0, prompt_tokens - cached)

    assistant_message = (Message(role="assistant", content=norm_blocks)
                         if norm_blocks else None)
    return LLMResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason=stop_reason,
        input_tokens=uncached_input,
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=cached,
        cache_write_tokens=0,
        latency_ms=latency_ms,
        raw=doc,
        assistant_message=assistant_message,
        # OpenRouter reports what it actually charged (usage.include). Recorded for
        # reconciliation against our configured prices; the ledger and kill-switch still
        # run on config prices, which are what the dry-run estimate uses.
        reported_cost_usd=_reported_cost(usage),
        reasoning_text=reasoning,
    )


def _decode_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Decode a tool call's ``arguments`` (a JSON string) into a dict."""
    if isinstance(raw, dict):
        return raw, True          # some servers helpfully pre-decode it
    if raw is None or raw == "":
        return {}, True           # a no-argument tool call is legitimate
    if not isinstance(raw, str):
        return {}, False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, False
    return (parsed, True) if isinstance(parsed, dict) else ({}, False)


def _reported_cost(usage: dict[str, Any]) -> float | None:
    """The provider's own cost figure for this call, if it reported one."""
    for key in ("cost", "total_cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
