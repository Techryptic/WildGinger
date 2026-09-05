# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared scan context + agent factory.

`ScanContext` bundles the live objects every stage needs (settings, db, audit,
navigator, sanitizer, provider cache) so stages stay small. `make_agent` builds a
role-routed :class:`Agent` with the right model, budget, tools, and the privileged
system prompt — and wires the injection guard into tool results.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import injection_guard
from .agent import Agent, Tool
from .audit import Audit
from .db import Database
from .indexing import Index
from .navigation import Navigator
from .prompts import build_system_prompt
from .providers.base import LLMProvider, build_provider
from .sanitizer import Sanitizer
from .settings import Settings


@dataclass
class ScanContext:
    settings: Settings
    root: Path
    db: Database
    audit: Audit
    index: Index
    sanitizer: Sanitizer
    system_prompt: str
    _provider_cache: dict[str, LLMProvider] = field(default_factory=dict)
    # Guards the cache's check-then-set. `make_agent` runs per cell and per judge per
    # finding, from every hunter thread, so an unguarded `if key not in cache` let two
    # threads each BUILD a provider for the same model. Each duplicate carries its own
    # boto3 Session + client + credential chain + connection pool, and its own copy of
    # the self-healing flags — so the "temperature is unsupported" lesson had to be
    # re-learned per duplicate, and the double-checked lock inside the provider's own
    # lazy client build was defeated by having two providers to begin with.
    _provider_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── provider routing ───────────────────────────────────────────────────────
    def provider_for_role(self, role: str) -> LLMProvider:
        rc, mp = self.settings.model_for_role(role)
        key = _provider_cache_key(mp)
        hit = self._provider_cache.get(key)
        if hit is not None:
            return hit
        with self._provider_lock:
            # Re-check inside the lock: another thread may have built it while we waited.
            hit = self._provider_cache.get(key)
            if hit is None:
                hit = build_provider(
                    mp.provider, model_id=mp.model_id,
                    profile=self.settings.aws.profile, region=self.settings.aws.region,
                    base_url=getattr(mp, "base_url", None),
                    # Per-model quirks/features from config.yaml.
                    send_temperature=getattr(mp, "send_temperature", True),
                    prompt_caching=getattr(mp, "prompt_caching", False),
                    # Hunters and (during validation) judges share one instance.
                    max_concurrency=max(1, self.settings.concurrency.hunters),
                    # OpenAI-compatible / OpenRouter settings (ignored by bedrock).
                    api_key_env=getattr(mp, "api_key_env", None),
                    tool_mode=getattr(mp, "tool_mode", "native"),
                    token_param=getattr(mp, "token_param", "max_tokens"),
                    request_timeout_seconds=getattr(mp, "request_timeout_seconds", 300),
                    max_concurrent_requests=getattr(mp, "max_concurrent_requests", None),
                    routing=getattr(mp, "routing", None),
                    allowed_endpoints=list(
                        getattr(self.settings.governance, "allowed_model_endpoints", [])
                        or []),
                )
                self._provider_cache[key] = hit
        return hit


    # ── navigator (read-only tools) ──────────────────────────────────────────────
    def make_navigator(self, agent_id: str, *, with_tools: bool = True) -> Navigator:
        def _sanitize(text: str, file_rel: str) -> str:
            # REDACTION ONLY. The injection reminder moved to `_guard`, applied after
            # truncation — chaining it here meant the note was appended before _truncate
            # and therefore cut off on exactly the largest files.
            return self.sanitizer.sanitize(text, file_rel)

        def _guard(text: str, file_rel: str, line_offset: int, cite_line: bool) -> str:
            return injection_guard.guard_tool_result(
                text, audit=self.audit, agent_id=agent_id, file_rel=file_rel,
                line_offset=line_offset, cite_line=cite_line)

        def _on_sanitizer_error(file_rel: str, exc: BaseException) -> None:
            # A redaction failure is a security-relevant anomaly, not a shrug: the
            # navigator now withholds the content instead of shipping it raw, and this
            # makes the fact visible in the 0600 audit log.
            self.audit.anomaly("sanitizer_failed", agent_id=agent_id,
                               file_rel=file_rel, error=f"{type(exc).__name__}: {exc}")
            print(f"⚠ Redaction failed for {file_rel} ({type(exc).__name__}); the "
                  f"content was WITHHELD from the model rather than sent unredacted.")

        def _record_read(file_rel: str, start: int, end: int) -> None:
            fid = self._file_id(file_rel)
            if fid is not None:
                self.db.record_read(agent_id, fid, start, end)

        return Navigator(
            root=self.root,
            index=self.index,
            max_result_tokens=self.settings.tools.max_result_tokens,
            list_default_depth=self.settings.tools.list_default_depth,
            sanitize=_sanitize,
            guard=_guard,
            record_read=_record_read,
            # Same exclusions discovery uses, so the agent's tools don't walk (and pay
            # to read) node_modules/venv/dist that the scan itself skipped.
            exclude_dirs=tuple(self.settings.scope.exclude_dirs),
            max_file_bytes=self.settings.scope.max_file_kb * 1024,
            on_sanitizer_error=_on_sanitizer_error,
        )

    def _file_id(self, file_rel: str) -> Optional[int]:
        def _q(cur):
            cur.execute("SELECT id FROM files WHERE path = ?;", (file_rel,))
            row = cur.fetchone()
            return row["id"] if row else None
        return self.db.read(_q)

    # ── agent factory ────────────────────────────────────────────────────────────
    def make_agent(self, role: str, agent_id: str, *, with_tools: bool = True,
                   lease: tuple[int, str] | None = None) -> Agent:
        """Build an agent for ``role``.

        ``lease`` is ``(cell_id, lease_owner)`` for a hunt. When given, every completed
        turn renews the cell's lease through the ``on_checkpoint`` hook — which existed
        and fired on every turn but was never wired to anything. Without it a hunt with a
        generous turn budget reliably outlived its 600s claim, and any other claimant
        could then take the cell and pay for the same work again.
        """
        rc, mp = self.settings.model_for_role(role)
        provider = self.provider_for_role(role)
        tools: list[Tool] = self.make_navigator(agent_id).tools() if with_tools else []

        on_checkpoint = None
        if lease is not None:
            cell_id, lease_owner = lease

            def on_checkpoint(_messages) -> None:  # noqa: F811 — intentional closure
                try:
                    if not self.db.renew_lease(cell_id, lease_owner):
                        # Lost the claim: record it, but keep working. The terminal
                        # transition is owner-guarded, so this hunt simply will not be
                        # the one that commits — it must not crash mid-conversation.
                        self.audit.event("lease_renewal_lost", cell_id=cell_id,
                                         agent_id=agent_id)
                except Exception as exc:  # noqa: BLE001 — bookkeeping, never fatal
                    self.audit.event("lease_renewal_failed", cell_id=cell_id,
                                     error=str(exc))

        return Agent(
            provider=provider,
            role=role,
            system=self.system_prompt,
            tools=tools,
            max_tokens=mp.max_tokens,
            temperature=rc.temperature,
            max_turns=rc.max_turns,
            max_resume_attempts=self.settings.resilience.max_resume_attempts,
            backoff_cap_seconds=self.settings.resilience.backoff_cap_seconds,
            max_retry_seconds=self.settings.resilience.max_retry_seconds,
            agent_id=agent_id,
            audit=self.audit,
            db=self.db,
            on_checkpoint=on_checkpoint,
        )


def build_system_prompt_for(settings: Settings) -> str:
    return build_system_prompt(settings.governance.engagement_context_file)


# Constructor arguments that change how a provider BEHAVES. All of them belong in the
# cache key.
#
# The key used to be `provider:model_id` alone, while the instance also depends on
# base_url, send_temperature and prompt_caching — so two model profiles sharing a
# provider+model_id but differing in those (a judge with caching off, a hunter with it
# on) silently shared whichever instance was built first: expected cache pricing never
# happened, and a `send_temperature: true` provider reused for a deployment that rejects
# temperature cost one guaranteed-failing request per process.
#
# With OpenAI-compatible endpoints this stops being a corner case: the same `model_id`
# (e.g. "qwen2.5-coder:32b") legitimately appears twice pointing at different hosts,
# different keys or different tool modes, and collapsing those into one instance would
# send requests to the wrong endpoint entirely.
_CACHE_KEY_FIELDS = (
    "provider", "model_id", "base_url", "send_temperature", "prompt_caching",
    "api_key_env", "tool_mode", "token_param", "request_timeout_seconds",
    "max_concurrent_requests",
)


def _provider_cache_key(mp) -> str:
    parts = [str(getattr(mp, name, None)) for name in _CACHE_KEY_FIELDS]
    routing = getattr(mp, "routing", None)
    if routing is not None:
        # Routing preferences change which upstream serves the request, so two profiles
        # that differ only there must not share an instance.
        parts.append(json.dumps(routing.to_body(), sort_keys=True)
                     if hasattr(routing, "to_body") else str(routing))
    return "|".join(parts)
