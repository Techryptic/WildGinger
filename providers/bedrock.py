# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Bedrock Converse provider.

Wraps the standard boto3 Bedrock Converse pattern:

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client  = session.client("bedrock-runtime")
    resp    = client.converse(modelId=MODEL_ID, messages=[...], inferenceConfig={...})

…and extends it with:
  * native tool use via ``toolConfig`` (so the agent can call our read-only
    navigation tools),
  * normalization between our provider-blind types and the Converse wire shape,
  * tenacity-based retry/backoff that distinguishes throttling/5xx (retryable)
    from validation errors (not retryable).

Mapping notes (Converse wire format):
  * messages: ``{"role": ..., "content": [ {"text": ...} | {"toolUse": ...}
    | {"toolResult": ...} ]}``
  * a tool result block we send back: ``{"toolResult": {"toolUseId": id,
    "content": [{"text": ...}], "status": "success"|"error"}}``
  * system prompt: ``system=[{"text": ...}]``
  * token usage: ``resp["usage"]["inputTokens"|"outputTokens"]``
  * stop reason: ``resp["stopReason"]`` ("end_turn" | "tool_use" | "max_tokens" …)
"""
from __future__ import annotations

import threading
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    ProviderError,
    ToolSpec,
    ToolUse,
)

# Bedrock error codes we consider transient/retryable.
# Wall-clock ceiling for ONE generate() call's retries, chosen to stay well inside the
# 600s cell lease so a retrying call cannot outlive the claim it is working under.
_RETRY_DEADLINE_SECONDS = 240

_RETRYABLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
    # Both are transient capacity/stream faults that succeed on a retry. Omitting
    # them meant a quota blip permanently FAILED the cell.
    "ServiceQuotaExceededException",
    "ModelStreamErrorException",
}

# Transport-level botocore exceptions. These are NOT ClientErrors: they carry no
# `.response`, so the code-based classifier below saw an empty code and declared
# them non-retryable — meaning a read timeout (very likely, given the 60s default
# vs. 8000-token responses) killed the cell outright instead of retrying.
# Matched by class NAME so this module stays importable without botocore.
_RETRYABLE_BOTOCORE_TYPES = frozenset({
    "ReadTimeoutError",
    "ConnectTimeoutError",
    "ConnectionError",
    "ConnectionClosedError",
    "EndpointConnectionError",
    "IncompleteReadError",
    "ResponseStreamingError",
    "HTTPClientError",
})


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


# Phrases Bedrock uses when a model rejects a parameter outright. Requiring one of
# these (in addition to the parameter name) stops an unrelated error that merely
# mentions e.g. "temperature" from silently disabling the parameter for a whole run.
_UNSUPPORTED_MARKERS = (
    "not supported",
    "unsupported",
    "deprecated",
    "invalid",
    "unexpected",
    "unknown field",
    "does not support",
)


def _is_unsupported_param(exc: BaseException, param: str) -> bool:
    """True if ``exc`` says ``param`` specifically is unsupported by this model."""
    msg = str(exc).lower()
    if param.lower() not in msg:
        return False
    return any(marker in msg for marker in _UNSUPPORTED_MARKERS)


class BedrockProvider(LLMProvider):
    """Concrete provider over AWS Bedrock ``converse``."""

    def __init__(self, *, model_id: str, profile: str | None, region: str,
                 send_temperature: bool = True, prompt_caching: bool = False,
                 max_concurrency: int = 12):
        self.model_id = model_id
        # None/empty = use botocore's default credential chain instead of a named
        # profile. See _get_client for why that matters.
        self._profile = profile or None
        self._region = region
        # botocore's connection pool defaults to 10. One provider instance is shared by
        # every hunter AND (during validation) every judge thread, so at the shipped
        # hunters=12 the pool was already undersized: urllib3 logs "Connection pool is
        # full, discarding connection" and then pays a fresh TLS handshake on the next
        # call — added latency and socket churn that lengthens every cell.
        self._max_concurrency = max(10, int(max_concurrency))
        self._client = None  # lazy — only import boto3 when actually used
        # One provider instance is shared across all concurrency.hunters threads, so
        # the lazy client build and the self-healing flags below need a lock.
        self._client_lock = threading.Lock()
        # Some Bedrock models reject 'temperature' ("deprecated for this model").
        # We detect that once and omit it for the rest of the run (self-healing).
        self._omit_temperature = not send_temperature
        # Prompt caching (opt-in per model). When on we insert `cachePoint` markers so
        # the stable prefix (system prompt + tool config) is billed at the cheaper
        # cache-read rate on subsequent turns. Reading `cacheReadInputTokens` without
        # ever SENDING a cachePoint meant cache stats were always 0.
        self._prompt_caching = prompt_caching
        # Disabled automatically if the model turns out not to support cachePoint.
        self._caching_unsupported = False


    # ── lazy client ───────────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                # Double-checked: 12 concurrent hunters share one provider via
                # ScanContext._provider_cache, and the unsynchronized check-then-set
                # built up to 12 duplicate clients (12 credential-resolution chains).
                if self._client is None:
                    import boto3  # local import keeps the package importable without boto3
                    from botocore.config import Config

                    # A named profile is only ONE way to authenticate, and forcing one
                    # made every other way impossible: boto3.Session(profile_name=...)
                    # raises ProfileNotFound on a machine whose credentials live in the
                    # environment rather than in ~/.aws/credentials. Leaving
                    # `aws.profile: null` hands the decision to botocore's default chain,
                    # which covers environment keys, an EC2/ECS instance role, SSO, and a
                    # Bedrock API key in AWS_BEARER_TOKEN_BEDROCK.
                    session_kwargs: dict[str, Any] = {"region_name": self._region}
                    if self._profile:
                        session_kwargs["profile_name"] = self._profile
                    session = boto3.Session(**session_kwargs)
                    self._client = session.client(
                        "bedrock-runtime",
                        config=Config(
                            # The default read_timeout is 60s, but an 8000-token
                            # response takes several minutes to generate — so every
                            # long answer timed out, got billed by AWS anyway, and
                            # then FAILED the cell.
                            read_timeout=900,
                            connect_timeout=10,
                            # We already retry in two outer layers (tenacity here and
                            # the agent's resume loop). botocore's own 5 attempts
                            # multiplied against those into ~630 requests per turn;
                            # keep it minimal and let the outer layers own the policy.
                            retries={"max_attempts": 1, "mode": "standard"},
                            # Enough sockets for hunters + judges, with headroom.
                            max_pool_connections=max(10, self._max_concurrency * 2),
                        ),
                    )
        return self._client

    # ── wire mapping ───────────────────────────────────────────────────────────
    @staticmethod
    def _to_wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            wire.append({"role": m.role, "content": _blocks_to_wire(m.content)})
        return wire

    @staticmethod
    def _tool_config(tools: list[ToolSpec] | None) -> dict[str, Any] | None:
        if not tools:
            return None
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.input_schema},
                    }
                }
                for t in tools
            ]
        }

    # ── main entrypoint ─────────────────────────────────────────────────────────
    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        @retry(
            retry=retry_if_exception(_is_retryable),
            # FULL jitter (uniform(0, min(cap, 2**n))), matching agent.py's resume loop.
            # `wait_exponential_jitter` looks equivalent but tenacity's `jitter`
            # defaults to 1.0 SECOND on top of a dominating exponential term, so the
            # sleep spread was ~1s at every attempt (measured: attempt 6 => 32.0-33.0s).
            # Under a shared Bedrock/Apigee quota all N hunters therefore re-fired
            # inside the same 1-second window on every wave and re-throttled each
            # other — a textbook thundering herd on the innermost, most frequently
            # taken retry layer.
            wait=wait_random_exponential(multiplier=1, max=60),
            # Attempts AND time. An attempt cap alone is not a bound: read_timeout is
            # 900s and ReadTimeoutError is retryable, so six attempts could legitimately
            # block ~91 MINUTES inside a single generate() — six generations AWS bills in
            # full and the client discards, one of N hunter slots gone for the duration,
            # a lease that expired 81 minutes earlier, and 75% of a 120-minute run budget
            # spent on one cell. agent.max_retry_seconds cannot help: it is only checked
            # BETWEEN tenacity cycles, never inside one.
            stop=(stop_after_attempt(6) | stop_after_delay(_RETRY_DEADLINE_SECONDS)),
            reraise=True,
        )
        def _call() -> LLMResponse:
            return self._converse_once(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )

        return _call()

    def _converse_once(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None,
    ) -> LLMResponse:
        client = self._get_client()

        def _build_kwargs(include_temperature: bool,
                          include_cache_points: bool) -> dict[str, Any]:
            inference: dict[str, Any] = {"maxTokens": max_tokens}
            if include_temperature:
                inference["temperature"] = temperature
            kw: dict[str, Any] = {
                "modelId": self.model_id,
                "messages": self._to_wire_messages(messages),
                "inferenceConfig": inference,
            }
            if system:
                # A cachePoint AFTER the system prompt makes that (large, identical
                # every call) prefix cacheable across turns and across cells.
                sys_blocks: list[dict[str, Any]] = [{"text": system}]
                if include_cache_points:
                    sys_blocks.append({"cachePoint": {"type": "default"}})
                kw["system"] = sys_blocks
            tool_cfg = self._tool_config(tools)
            if tool_cfg:
                if include_cache_points:
                    # Tool schemas are also byte-identical every call → cache them too.
                    tool_cfg = {
                        **tool_cfg,
                        "tools": [*tool_cfg["tools"],
                                  {"cachePoint": {"type": "default"}}],
                    }
                kw["toolConfig"] = tool_cfg
            return kw

        use_cache = self._prompt_caching and not self._caching_unsupported
        t0 = time.time()
        try:
            resp = client.converse(**_build_kwargs(
                include_temperature=not self._omit_temperature,
                include_cache_points=use_cache,
            ))
        except Exception as exc:  # noqa: BLE001
            resp = self._retry_without_unsupported_params(
                client, exc, _build_kwargs, use_cache)
        latency_ms = (time.time() - t0) * 1000.0
        return _parse_converse_response(resp, latency_ms)

    def _retry_without_unsupported_params(self, client, exc: Exception,
                                          build_kwargs, use_cache: bool):
        """Self-heal a request the model rejected for an unsupported parameter.

        Two known cases, each disabled for the REST of the run once detected:
          * ``temperature`` — some models only accept their default;
          * ``cachePoint``  — prompt caching isn't available for every model.

        Anything else is re-raised classified. We deliberately do NOT match on the
        bare word "temperature": an unrelated failure whose message merely mentions it
        would otherwise silently drop the parameter for the whole run.
        """
        # Read AND update the self-heal flags under the lock. Unsynchronized, a lost
        # update turned a HEALABLE error into a hard failure for every other thread:
        # thread A set _omit_temperature while B was still in flight, so B's
        # `not self._omit_temperature` became False, B fell through to
        # `raise _wrap_boto_error(exc)`, and a ValidationException is classified
        # NON-retryable — so B's cell FAILED outright. On a send_temperature
        # misconfiguration one thread healed and the other N-1 cells died and had to be
        # re-hunted (re-paid) on a later pass.
        with self._client_lock:
            retry_temp = (_is_unsupported_param(exc, "temperature")
                          and not self._omit_temperature)
            retry_cache = _is_unsupported_param(exc, "cachepoint") and use_cache
            already_healed = (_is_unsupported_param(exc, "temperature")
                              and self._omit_temperature)
            if retry_temp:
                self._omit_temperature = True
            if retry_cache:
                self._caching_unsupported = True
            omit_temp_now = self._omit_temperature
            cache_now = self._prompt_caching and not self._caching_unsupported

        if not (retry_temp or retry_cache):
            if already_healed:
                # Another thread already learned this lesson: retry with the healed
                # settings instead of failing this cell for a problem we have fixed.
                try:
                    return client.converse(**build_kwargs(
                        include_temperature=not omit_temp_now,
                        include_cache_points=cache_now,
                    ))
                except Exception as exc2:  # noqa: BLE001
                    raise _wrap_boto_error(exc2) from exc2
            raise _wrap_boto_error(exc) from exc
        try:
            return client.converse(**build_kwargs(
                include_temperature=not omit_temp_now,
                include_cache_points=cache_now,
            ))
        except Exception as exc2:  # noqa: BLE001
            raise _wrap_boto_error(exc2) from exc2



# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (pure; unit-testable without boto3)
# ──────────────────────────────────────────────────────────────────────────────

def _blocks_to_wire(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map our normalized content blocks to Converse wire blocks."""
    out: list[dict[str, Any]] = []
    for b in content:
        btype = b.get("type")
        if btype == "text":
            out.append({"text": b.get("text", "")})
        elif btype == "tool_use":
            out.append({
                "toolUse": {
                    "toolUseId": b["tool_use_id"],
                    "name": b["name"],
                    "input": b.get("input", {}),
                }
            })
        elif btype == "tool_result":
            out.append({
                "toolResult": {
                    "toolUseId": b["tool_use_id"],
                    "content": [{"text": b.get("text", "")}],
                    "status": b.get("status", "success"),
                }
            })
        else:
            # Already-wire block (defensive passthrough).
            out.append(b)
    return out


def _parse_converse_response(resp: dict[str, Any], latency_ms: float) -> LLMResponse:
    """Map a Converse API response to our normalized :class:`LLMResponse`."""
    output_msg = (resp.get("output") or {}).get("message") or {}
    wire_content = output_msg.get("content") or []

    texts: list[str] = []
    tool_uses: list[ToolUse] = []
    norm_blocks: list[dict[str, Any]] = []
    for block in wire_content:
        if "text" in block:
            texts.append(block["text"])
            norm_blocks.append({"type": "text", "text": block["text"]})
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_uses.append(ToolUse(
                tool_use_id=tu.get("toolUseId", ""),
                name=tu.get("name", ""),
                input=tu.get("input", {}) or {},
            ))
            norm_blocks.append({
                "type": "tool_use",
                "tool_use_id": tu.get("toolUseId", ""),
                "name": tu.get("name", ""),
                "input": tu.get("input", {}) or {},
            })

    usage = resp.get("usage") or {}
    assistant_message = Message(role="assistant", content=norm_blocks) if norm_blocks else None
    return LLMResponse(
        text="\n".join(texts),
        tool_uses=tool_uses,
        stop_reason=resp.get("stopReason"),
        input_tokens=int(usage.get("inputTokens", 0) or 0),
        output_tokens=int(usage.get("outputTokens", 0) or 0),
        # Bedrock prompt-caching usage (only present when caching is enabled for the
        # model/request; absent → 0). Keys per the Converse usage object.
        cache_read_tokens=int(usage.get("cacheReadInputTokens", 0) or 0),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0) or 0),
        latency_ms=latency_ms,
        raw=resp,
        assistant_message=assistant_message,
    )


def _wrap_boto_error(exc: Exception) -> ProviderError:
    """Classify a boto3 exception as retryable (throttling/5xx/transport) or not."""
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code", "")
    retryable = code in _RETRYABLE_CODES
    if not retryable:
        # Transport faults arrive as BotoCoreError subclasses with no `.response`,
        # so there is no error Code to match — classify them by type name instead.
        # Walk the MRO so subclasses of the listed types are caught too.
        for klass in type(exc).__mro__:
            if klass.__name__ in _RETRYABLE_BOTOCORE_TYPES:
                retryable = True
                break
    return ProviderError(f"{code or type(exc).__name__}: {exc}", retryable=retryable)
