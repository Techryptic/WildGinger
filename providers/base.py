# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Provider-blind LLM interface + shared message/tool types.

The orchestrator and every stage talk only to :class:`LLMProvider`. The concrete
provider is selected by :func:`build_provider`, with no change above this layer.

These types are intentionally a small, normalized superset of what Bedrock's
Converse API exposes (text, tool-use, token usage, stop reason), so mapping to
another vendor is mechanical.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass
class Message:
    """One conversation turn. ``content`` is a list of content blocks in the
    normalized shape the provider understands (text / tool_use / tool_result)."""
    role: Role
    content: list[dict[str, Any]]

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(role="user", content=[{"type": "text", "text": text}])

    @classmethod
    def assistant_text(cls, text: str) -> "Message":
        return cls(role="assistant", content=[{"type": "text", "text": text}])


@dataclass
class ToolSpec:
    """Declaration of a tool the model may call."""
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the tool's arguments


@dataclass
class ToolUse:
    """A tool call requested by the model."""
    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized model response."""
    text: str
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt-caching token counts (populated only when the provider/model reports
    # them, e.g. Bedrock prompt caching or the GPT gateway's cached_tokens). Default
    # 0 so callers never crash when caching is off/unsupported.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None
    # The assistant message exactly as the provider returned it, so the caller can
    # append it to the running history before adding tool_result blocks.
    assistant_message: Message | None = None
    # What the PROVIDER says this call cost, in USD, when it reports one (OpenRouter
    # does, via `usage: {include: true}`). None means "not reported" — most providers.
    #
    # This is recorded for reconciliation only. The cost ledger and the hard kill-switch
    # deliberately keep running on the prices in config.yaml: the dry-run estimate has to
    # project a cost BEFORE any call exists, preflight refuses to start when a model is
    # unpriced, and a ceiling that depended on a field some providers omit would be a
    # ceiling that silently stops working. Having both numbers in the audit log is what
    # makes a configured price checkable against reality.
    reported_cost_usd: float | None = None
    # A reasoning model's THINKING, when the endpoint exposes it separately (GLM sends
    # `message.reasoning`; vLLM and DeepSeek send `message.reasoning_content`).
    #
    # It is kept strictly apart from ``text`` and MUST NOT be merged into it. The
    # findings parser scans for the last JSON value in the model's answer, and a model's
    # thinking routinely contains a DRAFT findings array that it then revises or talks
    # itself out of — folding reasoning into the answer would let that draft be reported
    # as a real finding. This field exists so a reply that spent its whole token budget
    # thinking can be DIAGNOSED (see the preflight probe) instead of looking like
    # silence; nothing in the analysis path reads it.
    reasoning_text: str | None = None

    @property
    def wants_tool(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_uses)


class ProviderError(RuntimeError):
    """Wraps a provider/API failure so callers can branch on retryability."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(abc.ABC):
    """Abstract, provider-blind LLM interface."""

    #: Identifier used in audit logs / cost ledger.
    model_id: str

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """One model turn. Returns text + any tool calls + token usage."""
        raise NotImplementedError


class EndpointNotAllowed(RuntimeError):
    """A provider was pointed at an endpoint the governance allowlist does not permit.

    Raised by :func:`build_provider`, which is the single choke point every provider is
    constructed through. ``Settings`` already refuses such a config at load; this is the
    defence-in-depth copy, so a programmatic caller that builds a provider directly
    cannot bypass the egress control either.
    """


def _check_endpoint_allowed(base_url: str | None, allowed: list[str] | None) -> None:
    """Raise :class:`EndpointNotAllowed` unless ``base_url`` is on the allowlist.

    Defaults/environment values must already be resolved. None means there is no
    HTTP API endpoint to check; a missing required endpoint is refused by the provider.
    """
    url = (base_url or "").strip()
    if not url:
        return
    from ..settings import endpoint_transport, model_endpoint_allowed

    prefixes = [p.strip() for p in (allowed or []) if p and p.strip()]
    try:
        transport = endpoint_transport(url)
        permitted = model_endpoint_allowed(url, prefixes)
    except ValueError as exc:
        raise EndpointNotAllowed(str(exc)) from exc
    if transport in ("public", "hostname"):
        raise EndpointNotAllowed(
            "plaintext http:// model endpoints require loopback or a non-public literal IP; "
            "use https:// for public addresses and hostnames")
    if not permitted:
        raise EndpointNotAllowed(
            f"model endpoint '{url}' is not permitted by "
            f"governance.allowed_model_endpoints "
            f"({prefixes or 'empty — no custom endpoints allowed'}). Add its prefix to "
            f"that list in config.yaml to authorize sending code there.")


def build_provider(provider: str, *, model_id: str, profile: str | None = "default",
                   region: str = "us-east-1", base_url: str | None = None,
                   send_temperature: bool = True,
                   prompt_caching: bool = False,
                   max_concurrency: int = 12,
                   api_key_env: str | None = None,
                   tool_mode: str = "native",
                   token_param: str = "max_tokens",
                   request_timeout_seconds: int = 300,
                   max_concurrent_requests: int | None = None,
                   routing: Any = None,
                   allowed_endpoints: list[str] | None = None) -> LLMProvider:
    """Factory: map a provider name to a concrete implementation.

    Supported:
      * ``bedrock``       — AWS Bedrock Runtime ``Converse``, native tool use;
      * ``openai_gpt``    — an explicitly configured OpenAI gateway (deployment in the URL,
        ReAct text protocol for tools);
      * ``openai_compat`` — ANY OpenAI-compatible ``/chat/completions`` endpoint: a local
        model (Ollama / LM Studio / vLLM / llama.cpp) or a hosted one. Native tool
        calling, with a ReAct fallback for endpoints that have no tool API;
      * ``openrouter``    — ``openai_compat`` plus OpenRouter's routing/attribution
        extras and its reported per-call cost.

    Imports are local so the package imports cleanly even when a given SDK/lib isn't
    installed (boto3 in particular is optional).

    ``provider`` must be one of the CANONICAL names above. Friendly spellings
    (``ollama``, ``vllm``, ``local``, ``gpt``, …) are accepted in ``config.yaml`` and
    normalized to these by ``settings.ModelProfile``, so the mapping lives in exactly one
    place — two hand-maintained alias lists would eventually disagree about what
    ``local`` means.

    ``send_temperature`` / ``prompt_caching`` / ``tool_mode`` / ``token_param`` /
    ``api_key_env`` / ``request_timeout_seconds`` / ``max_concurrent_requests`` /
    ``routing`` come from the model profile in config.yaml (see
    :class:`settings.ModelProfile`). ``max_concurrency`` is how many threads will share
    this instance, used to size the HTTP connection pool.

    ``allowed_endpoints`` is ``governance.allowed_model_endpoints``: every effective
    HTTP API URL must be on it, including default/environment destinations.
    """
    from ..settings import resolve_model_base_url

    base_url = resolve_model_base_url(provider, base_url)
    endpoint = base_url
    if endpoint and provider in ("openai_gpt", "gpt", "openai", "azure"):
        endpoint = endpoint.format(deployment=model_id)
    _check_endpoint_allowed(endpoint, allowed_endpoints)
    if provider == "bedrock":
        from .bedrock import BedrockProvider
        return BedrockProvider(model_id=model_id, profile=profile, region=region,
                               send_temperature=send_temperature,
                               prompt_caching=prompt_caching,
                               max_concurrency=max_concurrency)
    if provider in ("openai_gpt", "gpt", "openai", "azure"):
        # The legacy spellings stay accepted here: this branch predates the settings-level
        # normalization and may have direct callers.
        from .openai_gpt import OpenAIGPTProvider
        return OpenAIGPTProvider(model_id=model_id, base_url=base_url,
                                 send_temperature=send_temperature)
    if provider in ("openai_compat", "openrouter"):
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            model_id=model_id, base_url=base_url, api_key_env=api_key_env,
            send_temperature=send_temperature, tool_mode=tool_mode,
            token_param=token_param, timeout_seconds=request_timeout_seconds,
            max_concurrent_requests=max_concurrent_requests,
            routing=routing,
            openrouter=provider == "openrouter",
        )
    raise ValueError(
        f"unknown provider '{provider}'. Supported: bedrock, openai_gpt, "
        f"openai_compat, openrouter")
