# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""OpenAI/GPT provider for an explicitly configured Apigee gateway.

The endpoint comes from ``base_url`` or ``GPT_BASE_URL`` and may contain
``{deployment}``. The key and app/user IDs come from constructor arguments or
``APIGEE_KEY``, ``FM_END_APP_ID``, and ``FM_END_USER_ID``. All are required;
there are no endpoint or credential defaults.

Tools use the text-based ReAct ``ACTION: tool_name({...})`` protocol, normalized
into the same ``tool_use`` blocks as the other providers.
"""
from __future__ import annotations

import json
import os
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

from .base import LLMProvider, LLMResponse, Message, ProviderError, ToolSpec, ToolUse

# Re-export the shared parsers for callers that import them from this module.
from ._openai_common import (
    _is_temperature_rejection,
    _parse_action,
    _tools_instructions,
    malformed_action_hint,
)


# Wall-clock ceiling for one generate() call's retries (see the Bedrock provider).
_RETRY_DEADLINE_SECONDS = 240


def _is_retryable(exc: BaseException) -> bool:
    """Retry only transient provider failures."""
    return isinstance(exc, ProviderError) and exc.retryable


# Authentication and invalid-request errors must fail without retries.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAIGPTProvider(LLMProvider):
    """Gateway chat completions, with ``model_id`` identifying the deployment."""

    def __init__(self, *, model_id: str, base_url: str | None = None,
                 apigee_key: str | None = None, app_id: str | None = None,
                 user_id: str | None = None, send_temperature: bool = True):
        base_url = base_url if base_url is not None else os.environ.get("GPT_BASE_URL")
        apigee_key = apigee_key if apigee_key is not None else os.environ.get("APIGEE_KEY")
        app_id = app_id if app_id is not None else os.environ.get("FM_END_APP_ID")
        user_id = user_id if user_id is not None else os.environ.get("FM_END_USER_ID")
        missing = [name for name, value in (
            ("base_url / GPT_BASE_URL", base_url),
            ("apigee_key / APIGEE_KEY", apigee_key),
            ("app_id / FM_END_APP_ID", app_id),
            ("user_id / FM_END_USER_ID", user_id),
        ) if not isinstance(value, str) or not value.strip()]
        if missing:
            raise ValueError(
                "provider 'openai_gpt' requires non-empty configuration: "
                + ", ".join(missing))
        self.model_id = model_id
        self._url = base_url.strip().format(deployment=model_id)
        self._key = apigee_key
        self._app_id = app_id
        self._user_id = user_id
        # Some deployments reject temperature; also self-heal on first rejection.
        self._omit_temperature = not send_temperature

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """One turn, with bounded retry/backoff on transient gateway failures."""
        @retry(
            retry=retry_if_exception(_is_retryable),
            # Full jitter prevents concurrent callers from retrying in lockstep.
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=(stop_after_attempt(6) | stop_after_delay(_RETRY_DEADLINE_SECONDS)),
            reraise=True,
        )
        def _call() -> LLMResponse:
            return self._chat_once(system=system, messages=messages,
                                   max_tokens=max_tokens, temperature=temperature,
                                   tools=tools)

        return _call()

    def _chat_once(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None,
    ) -> LLMResponse:
        sys_prompt = system
        if tools:
            sys_prompt = (system or "") + "\n\n" + _tools_instructions(tools)
        chat = self._to_chat_messages(sys_prompt, messages)

        def _payload(include_temperature: bool) -> bytes:
            body: dict[str, Any] = {"messages": chat,
                                    "max_completion_tokens": max_tokens}
            if include_temperature and temperature is not None:
                body["temperature"] = temperature
            return json.dumps(body).encode("utf-8")

        t0 = time.time()
        try:
            doc = self._post(_payload(not self._omit_temperature))
        except ProviderError as exc:
            # Self-heal the o-series/GPT-5 "temperature not supported" rejection once,
            # then omit it for the rest of the run.
            if _is_temperature_rejection(exc) and not self._omit_temperature:
                self._omit_temperature = True
                doc = self._post(_payload(False))
            else:
                raise
        latency_ms = (time.time() - t0) * 1000.0
        return _parse_chat_response(doc, latency_ms, tools_enabled=bool(tools))

    def _post(self, data: bytes) -> dict[str, Any]:
        """POST to the gateway; raise a classified :class:`ProviderError` on failure."""
        req = urllib.request.Request(
            self._url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "x-fm-aoai-end-app-id": self._app_id,
                "x-fm-aoai-end-user-id": self._user_id,
                "x-fm-aoai-end-transaction-id": "0",
                "apigeekey": self._key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # The response body is needed to detect temperature rejection.
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                detail = ""
            retryable = exc.code in _RETRYABLE_STATUS
            raise ProviderError(
                f"HTTP {exc.code} {exc.reason}: {detail}", retryable=retryable
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"URLError: {exc.reason}", retryable=True) from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=True) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"invalid JSON from gateway: {exc}",
                                retryable=False) from exc

    @staticmethod
    def _to_chat_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            parts: list[str] = []
            for b in m.content:
                btype = b.get("type")
                if btype == "text":
                    parts.append(b.get("text", ""))
                elif btype == "tool_use":
                    # Echo the model's own ACTION so the conversation stays coherent.
                    parts.append(f"ACTION: {b.get('name')}({json.dumps(b.get('input', {}))})")
                elif btype == "tool_result":
                    parts.append(f"OBSERVATION:\n{b.get('text', '')}")
            role = "assistant" if m.role == "assistant" else "user"
            out.append({"role": role, "content": "\n".join(parts)})
        return out


def _parse_chat_response(doc: dict[str, Any], latency_ms: float,
                         *, tools_enabled: bool = False) -> LLMResponse:
    choices = doc.get("choices") or []
    text = ""
    finish_reason = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        finish_reason = choices[0].get("finish_reason") or ""
    usage = doc.get("usage") or {}

    tool_uses: list[ToolUse] = []
    # Preserve truncation so callers do not treat partial findings as complete.
    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    norm_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []

    if tools_enabled and text:
        parsed = _parse_action(text)
        if parsed is not None:
            name, args, parse_ok = parsed
            if parse_ok:
                tu_id = f"gpt-{uuid.uuid4().hex[:12]}"
                tool_uses.append(ToolUse(tool_use_id=tu_id, name=name, input=args))
                stop_reason = "tool_use"
                # Preserve assistant prose alongside the tool call in history.
                norm_blocks = ([{"type": "text", "text": text}] if text.strip() else [])
                norm_blocks.append({"type": "tool_use", "tool_use_id": tu_id,
                                    "name": name, "input": args})
            else:
                # Ask for corrected arguments rather than dispatching an empty call.
                stop_reason = "end_turn"
                hint = malformed_action_hint(name)
                norm_blocks = [{"type": "text", "text": text + "\n\n" + hint}]
                text = norm_blocks[0]["text"]

    # prompt_tokens includes cached_tokens; audit prices them as disjoint counts.
    # This API does not report cache writes.
    cached = 0
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    # Inconsistent gateway counts must not produce a negative cost credit.
    uncached_input = max(0, prompt_tokens - cached)

    assistant_message = Message(role="assistant", content=norm_blocks) if norm_blocks else None
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
    )
