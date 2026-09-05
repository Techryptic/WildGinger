# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration - the single source of truth.

Edit ``config.yaml``; every module imports the validated :class:`Settings`
object from here. Pydantic validates types/ranges at load time so a typo in the
YAML fails fast with a clear message instead of surfacing as a mysterious runtime
error mid-scan.

A ``config_hash`` is computed over the *effective* config so each run is
reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, get_args
from urllib.parse import SplitResult, unquote, urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Base for every config model: an UNKNOWN key is a hard error.

    Without ``extra="forbid"`` pydantic silently drops keys it doesn't recognize,
    so a typo like ``concurency:`` (missing 'r') would validate cleanly and then do
    nothing at runtime — the exact opposite of the "a typo here fails fast with a
    clear error" promise at the top of config.yaml. Forbidding extras makes that
    promise real: the misspelled key is reported by name, with its location.

    ``validate_assignment`` extends the same guarantee to values set AFTER load,
    which is how every CLI override arrives (``settings.concurrency.hunters =
    args.hunters``). Without it, ``--hunters -5`` sailed past the ``ge=1`` bound and
    reached ThreadPoolExecutor.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class OpenRouterRouting(StrictModel):
    """OpenRouter's ``provider`` routing block (openrouter only).

    OpenRouter is a BROKER: one ``model_id`` can be served by several upstream
    providers, chosen per request. That has two consequences this project cares about:

      * **Reproducibility.** The same model can be answered by different upstreams
        between runs (different quantization, different tool support), which quietly
        breaks the "same repo + same config ⇒ comparable results" property the
        ledger relies on. Pin with ``order`` + ``allow_fallbacks: false``.
      * **Confidentiality.** Some upstreams log or train on prompts by default. This
        harness sends proprietary SOURCE CODE, so ``data_collection`` defaults to
        ``deny`` here rather than to OpenRouter's own default.

    ``require_parameters`` matters whenever tools are in play: without it OpenRouter
    may route to an upstream that silently DROPS the ``tools`` array, and the hunt then
    scores zero thoroughness for a reason nothing reports.
    """
    order: list[str] = Field(default_factory=list)   # e.g. ["anthropic", "amazon-bedrock"]
    allow_fallbacks: bool = True
    require_parameters: bool = True
    data_collection: Literal["deny", "allow"] = "deny"
    only: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)
    sort: Literal["price", "throughput", "latency"] | None = None

    def to_body(self) -> dict[str, Any]:
        """The ``provider`` object to embed in the request body (omitting empties)."""
        out: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "data_collection": self.data_collection,
        }
        if self.order:
            out["order"] = list(self.order)
        if self.only:
            out["only"] = list(self.only)
        if self.ignore:
            out["ignore"] = list(self.ignore)
        if self.sort:
            out["sort"] = self.sort
        return out


class AWSConfig(StrictModel):
    # None = use botocore's DEFAULT credential chain (environment variables, an EC2/ECS
    # instance role, SSO, or a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK). A named
    # profile is only one of the ways to authenticate, and forcing one made key-based
    # auth impossible: boto3.Session(profile_name="default") raises ProfileNotFound on a
    # machine that has credentials in the environment but no ~/.aws/credentials file.
    profile: str | None = "default"
    region: str = "us-east-1"


# Every provider the factory knows how to build (providers/base.py:build_provider).
# Kept here so a typo is caught at CONFIG LOAD with a suggestion, rather than surfacing
# as a ValueError from the factory after discovery, indexing and seeding have run.
KNOWN_PROVIDERS: tuple[str, ...] = (
    "bedrock",          # AWS Bedrock Runtime, Converse API
    "openai_gpt",       # explicitly configured OpenAI gateway (deployment-in-URL)
    "openai_compat",    # ANY OpenAI-compatible /chat/completions endpoint (local or hosted)
    "openrouter",       # openai_compat + OpenRouter routing/attribution extras
)
# Accepted spellings that map onto the canonical names above.
_PROVIDER_ALIASES: dict[str, str] = {
    "gpt": "openai_gpt", "openai": "openai_gpt", "azure": "openai_gpt",
    "openai_compatible": "openai_compat", "oai_compat": "openai_compat",
    "local": "openai_compat", "ollama": "openai_compat", "vllm": "openai_compat",
    "lmstudio": "openai_compat", "llamacpp": "openai_compat",
    "open_router": "openrouter", "openrouter.ai": "openrouter",
}


class ModelProfile(StrictModel):
    provider: str = "bedrock"          # see KNOWN_PROVIDERS
    model_id: str                      # Bedrock ARN · Apigee deployment · OpenAI-compat model name
    max_tokens: int = Field(4000, gt=0)
    # Endpoint. For `openai_gpt` this may contain {deployment}; when omitted it uses
    # GPT_BASE_URL, with no built-in gateway destination.
    # For `openai_compat`/`openrouter` it is the API root (…/v1) and `/chat/completions`
    # is appended. Any value here must be permitted by
    # `governance.allowed_model_endpoints` — see Settings._model_endpoints_are_allowed.
    base_url: str | None = None
    description: str = ""               # human note: what this model is for
    # Pricing in USD per 1,000,000 tokens (from the AWS/Azure price sheet). These
    # power the real cost tracking + hard kill-switch. Leave 0 only if unknown
    # (then that model's spend is under-counted — avoid for the kill-switch).
    price_per_1m_input: float = Field(0.0, ge=0.0)
    price_per_1m_output: float = Field(0.0, ge=0.0)
    # Some deployments reject an explicit `temperature` and only accept their default
    # (o-series / GPT-5 reasoning models). Set false to omit it from every request.
    # Leave true and the provider will also self-heal on the first rejection.
    send_temperature: bool = True
    # Bedrock prompt caching: insert cachePoint markers after the system prompt +
    # tool config so repeated prefixes are billed at the cheaper cache-read rate.
    # Only supported on some models, so it's opt-in per model.
    prompt_caching: bool = False
    # Price per 1M CACHE tokens. Cache reads are typically ~10% of the input price
    # and cache writes ~125%. Left 0 → the input price is used as a safe fallback
    # (never cheaper than reality, so the kill-switch can't under-count).
    price_per_1m_cache_read: float = Field(0.0, ge=0.0)
    price_per_1m_cache_write: float = Field(0.0, ge=0.0)

    # ── OpenAI-compatible / OpenRouter settings (ignored by bedrock) ───────────
    # NAME of the environment variable holding the API key — never the key itself.
    # A secret in config.yaml would be committed to source control and fed into
    # config_hash(). None = send no Authorization header at all, which is what most
    # local servers (Ollama / LM Studio / llama.cpp) expect.
    api_key_env: str | None = None
    # How this model receives the navigation tools:
    #   native — OpenAI `tools` + `tool_calls` (preferred: reliable, allows parallel calls)
    #   react  — the `ACTION: name({...})` text protocol, for endpoints with no tool API
    #   none   — no tools at all. See Settings._tool_mode_is_usable: a hunt with no tools
    #            cannot reach quality.thoroughness_min, so this is refused for that role.
    tool_mode: Literal["native", "react", "none"] = "native"
    # Which field caps the response length. OpenAI/Azure moved to
    # `max_completion_tokens`; Ollama, llama.cpp and older vLLM only accept `max_tokens`.
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # Per-request HTTP timeout. Local generation of several thousand tokens on CPU can
    # take many minutes, and a timeout mid-generation wastes the whole turn.
    request_timeout_seconds: int = Field(300, gt=0)
    # Cap on this model's concurrent in-flight requests, independent of
    # concurrency.hunters. One local GPU cannot serve 12 hunters at once; None = no cap.
    max_concurrent_requests: int | None = Field(None, gt=0)
    # 'free' DECLARES that this endpoint bills nothing (a model running on your own
    # hardware). It exempts the model from the unpriced-model refusal in preflight —
    # which exists because a $0 price makes the cost kill-switch unable to fire — while
    # keeping that fact explicit, printed and audit-logged instead of hidden behind a
    # fake price. Only budget.max_time_minutes bounds a free model.
    billing: Literal["metered", "free"] = "metered"
    # OpenRouter-only routing preferences (see OpenRouterRouting).
    routing: "OpenRouterRouting | None" = None

    @field_validator("provider")
    @classmethod
    def _provider_is_known(cls, provider: str) -> str:
        """Reject an unknown provider AT LOAD, with a suggestion.

        This used to be a bare ``str``: ``provider: openrouter_`` validated cleanly and
        then raised out of ``build_provider`` — after discovery, the full file read,
        indexing, attack-surface analysis and ledger seeding had already run.
        """
        raw = (provider or "").strip()
        canonical = _PROVIDER_ALIASES.get(raw.lower(), raw)
        if canonical not in KNOWN_PROVIDERS:
            hint = _suggest_key(raw.lower(), list(KNOWN_PROVIDERS) + list(_PROVIDER_ALIASES))
            suffix = f" Did you mean '{hint}'?" if hint else ""
            raise ValueError(
                f"unknown provider '{provider}'. Supported: "
                f"{', '.join(KNOWN_PROVIDERS)}.{suffix}")
        return canonical

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env_is_a_name_not_a_secret(cls, value: str | None) -> str | None:
        """Catch a key pasted into the field that wants a variable NAME.

        ``config.yaml`` is committed to source control and hashed into
        ``config_hash()``; a secret here would leak into both. Environment variable
        names are short and have no punctuation, so the test is cheap and unambiguous.
        """
        if value is None:
            return None
        name = value.strip()
        if not name:
            return None
        looks_like_secret = (
            len(name) > 64
            or name.lower().startswith(("sk-", "sk_", "or-", "bearer "))
            or any(ch in name for ch in ".:/ ")
        )
        if looks_like_secret:
            raise ValueError(
                "api_key_env must be the NAME of an environment variable (e.g. "
                "OPENROUTER_API_KEY), not the key itself. config.yaml is committed and "
                "hashed into config_hash, so a secret here would leak.")
        return name





class RoleConfig(StrictModel):
    model: str  # key into Settings.models
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_turns: int = Field(100, gt=0)


class BudgetConfig(StrictModel):
    max_time_minutes: int = Field(120, gt=0)
    # gt=0, not ge=0. Zero used to be accepted and then silently DISABLED both cost
    # ceilings (the loop check reads `cost >= cap > 0`, and the mid-call kill-switch
    # reads `cap > 0`), so `--max-cost 0` — the obvious way to say "spend nothing" —
    # ran completely unbounded. If you want to spend nothing, don't start a scan.
    max_cost_usd: float = Field(50.0, gt=0.0)
    max_passes: int = Field(10, gt=0)
    convergence_passes: int = Field(2, gt=0)


class ConcurrencyConfig(StrictModel):
    # Upper bound is generous; the REAL ceiling is your Bedrock/Apigee rate limits.
    # Going too high just causes throttling + retries (slower). Mixing providers
    # (Bedrock + GPT) is the better way to raise effective throughput.
    hunters: int = Field(6, gt=0, le=1000)



class ValidationConfig(StrictModel):
    """Multi-judge quorum. Each candidate finding is judged by several
    DIFFERENT models; a finding is confirmed only if >= quorum judges agree."""
    judges: list[str] = Field(default_factory=list)  # role names, e.g. [judge1, judge2, judge3]
    quorum: int = Field(2, ge=1)                      # how many must agree to confirm

    @field_validator("quorum")
    @classmethod
    def _quorum_is_reachable(cls, quorum: int, info: Any) -> int:
        """A quorum larger than the panel can NEVER be met.

        Without this check, ``quorum: 4`` with three judges silently left every single
        finding stuck at status 'new' forever — no confirmations, no false positives,
        and no error telling you why.
        """
        judges = info.data.get("judges") or []
        if judges and quorum > len(judges):
            raise ValueError(
                f"validation.quorum is {quorum} but only {len(judges)} judge(s) are "
                f"configured ({', '.join(judges)}) — no finding could ever reach "
                f"quorum. Set quorum <= {len(judges)}."
            )
        # ...and a quorum that is not a MAJORITY inverts the panel. `validate.py` tests
        # `confirmed >= quorum` before `rejected >= quorum`, so with 5 judges and
        # quorum 2 a finding that 3 judges REJECTED is reported "confirmed" with
        # boosted confidence. `quorum: 1` (previously legal) makes a single judge
        # decisive, which is the opposite of a cross-model vote.
        if judges and quorum * 2 <= len(judges):
            raise ValueError(
                f"validation.quorum is {quorum} with {len(judges)} judges, which is not "
                f"a majority: a finding rejected by most of the panel would still be "
                f"reported as confirmed. Set quorum >= {len(judges) // 2 + 1}."
            )
        return quorum

    @field_validator("judges")
    @classmethod
    def _judges_are_distinct(cls, judges: list[str]) -> list[str]:
        """Duplicate judge roles are not independent opinions.

        ``judges: [judge1, judge1]`` passed every check and satisfied a 2-of-2 quorum
        with ONE model — so correlated single-model errors were reported as independent
        cross-model agreement, at double the cost, with colliding agent ids.
        """
        dupes = sorted({j for j in judges if judges.count(j) > 1})
        if dupes:
            raise ValueError(
                f"validation.judges lists {dupes} more than once. A quorum needs "
                f"DIFFERENT judges; repeating one just double-charges for a single "
                f"model's opinion."
            )
        return judges
    # DEPRECATED alias. This had no reader anywhere: it described exactly what
    # `reachability.unreachable_handling` controls, so two keys claimed one behaviour and
    # toggling this one silently did nothing. Kept (and now actually consulted, see
    # Settings.effective_unreachable_handling) so existing configs don't break, but
    # `reachability.unreachable_handling` is the real switch.
    require_reachable: bool = True
    parallel: bool = True                             # validate findings concurrently (much faster)
    # Skip the (expensive) multi-judge quorum for findings below this severity.
    skip_validation_below: Literal["none", "low", "medium", "high"] = "low"



class DedupeConfig(StrictModel):
    """Noise control BEFORE the expensive validation step."""
    merge_same_location: bool = True   # collapse "20 findings, same endpoint, same attack class"
    # AI semantic dedupe: after the heuristic merge and BEFORE the judges, ask a
    # model to group findings that are the SAME root cause phrased differently or
    # spread across chunks/files, so we never spend judge tokens on duplicates.
    semantic: bool = False
    semantic_role: str = "dedupe"      # role (→ model) used for the semantic pass


class IndexingConfig(StrictModel):
    """Symbol/call-graph backend. 'auto' prefers tree-sitter, falls back to regex."""
    backend: Literal["auto", "tree_sitter", "regex"] = "auto"


class PreflightConfig(StrictModel):
    """Make one cheap call per configured model before scanning, to fail fast on a
    bad ARN / gateway error instead of after seeding thousands of cells."""
    enabled: bool = True


class ReachabilityConfig(StrictModel):

    """How to treat findings that aren't reachable from an external entry point."""
    enabled: bool = True
    # 'informational' keeps them in the report under a separate section (not gating);
    # 'hide' drops them; 'report' treats them like any other finding.
    unreachable_handling: Literal["informational", "hide", "report"] = "informational"



class ResilienceConfig(StrictModel):
    max_resume_attempts: int = Field(20, ge=0)
    backoff_cap_seconds: int = Field(300, gt=0)
    # Wall-clock ceiling on transport backoff for a single turn. The attempt count
    # alone left retries unbounded in TIME (20 × up to 300s = 68 min on one cell,
    # well past the 600s cell lease). Keep this under the lease.
    max_retry_seconds: float = Field(480.0, gt=0.0)


class ScopeConfig(StrictModel):
    path: str = "./target_repo"
    mode: Literal["full", "incremental"] = "full"
    artifact_categories: list[str] = Field(default_factory=lambda: ["code", "iac", "config"])
    include_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "java", "go", "c", "cpp"]
    )
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [".git", "node_modules", "target", "venv", "dist", "build"]
    )
    max_file_kb: int = Field(800, gt=0)
    # Scan unrecognized but TEXT files via the generic ("unknown") path so coverage
    # is complete. Images/binaries are still excluded (sniff + denylist).
    scan_unknown_extensions: bool = True
    binary_extensions: list[str] = Field(
        default_factory=lambda: [
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
            ".pdf", ".zip", ".gz", ".tar", ".tgz", ".7z", ".rar", ".bz2", ".xz",
            ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".class", ".jar",
            ".woff", ".woff2", ".ttf", ".otf", ".eot",
            ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".flac", ".ogg",
            ".pyc", ".pyo", ".wasm", ".db", ".sqlite", ".pdf",
        ]
    )
    chunk_lines: int = Field(150, gt=0)
    chunk_overlap_lines: int = Field(20, ge=0)


    @field_validator("chunk_overlap_lines")
    @classmethod
    def _overlap_lt_chunk(cls, v: int, info: Any) -> int:
        chunk = info.data.get("chunk_lines", 150)
        if v >= chunk:
            raise ValueError(f"chunk_overlap_lines ({v}) must be < chunk_lines ({chunk})")
        return v


class ToolsConfig(StrictModel):
    max_result_tokens: int = Field(4000, gt=0)
    list_default_depth: int = Field(1, ge=1)


# The canonical attack-class vocabulary. ``"all"`` is NOT a class — it is the
# open-ended sentinel (see coverage.OPEN_ENDED_CLASS) and is only legal inside
# ``hunting.attack_classes``, never inside a group.
KNOWN_ATTACK_CLASSES: tuple[str, ...] = (
    "injection", "authz", "authn", "crypto", "deserialization",
    "ssrf", "path_traversal", "xss", "secrets", "logic", "misconfig", "csrf",
)


class HuntingConfig(StrictModel):
    attack_classes: list[str] = Field(
        default_factory=lambda: [*KNOWN_ATTACK_CLASSES, "all"]
    )
    # Directed classes bundled K-per-cell instead of one cell per class: full
    # called-out coverage at (number of groups)x cost instead of (number of
    # classes)x. When set, this REPLACES attack_classes (set only one of them).
    attack_class_groups: list[list[str]] = Field(default_factory=list)
    open_ended: bool = True

    @field_validator("attack_class_groups")
    @classmethod
    def _groups_are_valid(cls, groups: list[list[str]]) -> list[list[str]]:
        """Every group must be a non-empty set of DISTINCT, KNOWN classes.

        A class in two groups is paid for twice per chunk; an unknown class is a
        typo that would silently hunt nothing; ``all`` is the open-ended sentinel
        and cannot share a directed cell. All three used to be accepted quietly.
        """
        known = set(KNOWN_ATTACK_CLASSES)
        seen: dict[str, int] = {}
        for i, group in enumerate(groups, 1):
            if not group:
                raise ValueError(
                    f"hunting.attack_class_groups group #{i} is empty — drop it or "
                    f"add classes to it.")
            for c in group:
                if c == "all":
                    raise ValueError(
                        f"hunting.attack_class_groups group #{i} contains 'all' — the "
                        f"open-ended sweep cannot share a directed cell. Use "
                        f"hunting.attack_classes for 'all'.")
                if c not in known:
                    raise ValueError(
                        f"hunting.attack_class_groups group #{i} lists unknown class "
                        f"'{c}'. Known classes: {', '.join(KNOWN_ATTACK_CLASSES)}.")
                if c in seen:
                    raise ValueError(
                        f"hunting.attack_class_groups lists '{c}' in both group "
                        f"#{seen[c]} and group #{i} — you would pay for it twice per "
                        f"chunk. Keep it in one group.")
                seen[c] = i
        return groups

    @model_validator(mode="after")
    def _groups_xor_flat_classes(self) -> "HuntingConfig":
        """attack_classes and attack_class_groups are two ways to say the same
        thing; silently preferring one would hide a conflict of intent."""
        explicit = self.model_fields_set
        if "attack_classes" in explicit and "attack_class_groups" in explicit:
            raise ValueError(
                "hunting.attack_classes and hunting.attack_class_groups are both set "
                "— they are two ways to choose the per-chunk cells. Delete one of "
                "them (groups replace the flat list).")
        return self


class SeedingConfig(StrictModel):
    mode: Literal["llm_only", "hybrid", "triage_only"] = "llm_only"
    fortify_sarif: str | None = None
    generic_sarif: list[str] = Field(default_factory=list)
    blackduck_report: str | None = None


class RulesConfig(StrictModel):
    enabled: bool = True
    dir: str = "./rules"
    org_patterns: str | None = None


class SanitizationConfig(StrictModel):
    mode: Literal["redact", "block", "off"] = "redact"
    entropy_threshold: float = Field(4.0, ge=0.0)
    quarantine_extensions: list[str] = Field(
        default_factory=lambda: [".env", ".pem", ".key", ".p12"]
    )


class QualityConfig(StrictModel):
    # NOTE on resolution: _score_thoroughness can only return {0.0, 0.4, 0.8, 1.0}, so
    # every value in (0.4, 0.8] — including this default — behaves identically. Treat it
    # as a 4-position switch, not a continuous dial.
    thoroughness_min: float = Field(0.7, ge=0.0, le=1.0)
    # A cell that scores below `thoroughness_min` is re-queued. After this many attempts
    # WITHOUT the score improving, stop re-paying for it: the model has read the code and
    # reported nothing, repeatedly, and asking again costs money to get the same answer.
    #
    # This is not just a cost knob. `coverage_pct` counts only ANALYZED cells and the pass
    # loop only converges when coverage reaches 1.0 — so a single permanently-re-queued
    # cell made convergence impossible, burned every one of `budget.max_passes` sweeps,
    # and then reported EXIT_COVERAGE_INCOMPLETE (a CI failure) for a scan that had in
    # fact looked at everything and found nothing.
    #
    # Cells that never made a single successful tool call (score 0.0) are NEVER settled
    # this way — an agent that read nothing is a genuine coverage gap and must keep
    # showing up as one.
    max_shallow_attempts: int = Field(2, ge=1)
    hedge_penalty: bool = True
    language_aware_fp: bool = True


class DataProtectionConfig(StrictModel):
    restrict_permissions: bool = True


class GovernanceConfig(StrictModel):

    engagement_context_file: str | None = None
    data_protection: DataProtectionConfig = Field(default_factory=DataProtectionConfig)
    # EGRESS ALLOWLIST: match scheme, host, port and a complete path boundary.
    # Includes OpenRouter's default URL and gateways configured via GPT_BASE_URL.
    # Empty (the default) permits no HTTP API endpoints. Bedrock uses its AWS SDK.
    allowed_model_endpoints: list[str] = Field(default_factory=list)


class BaselineConfig(StrictModel):
    file: str | None = None
    mode: Literal["new_only", "all"] = "new_only"


class SuppressionsConfig(StrictModel):
    file: str | None = None


class OutputConfig(StrictModel):
    # 'markdown' writes report.md AND executive_summary.md. Add 'exec_summary' to get
    # the summary WITHOUT the full markdown report (e.g. a JSON-first CI that still
    # wants a human-readable digest).
    formats: list[Literal["json", "sarif", "markdown", "exec_summary"]] = Field(
        default_factory=lambda: ["json", "sarif", "markdown"]
    )
    severity_threshold: Literal["low", "medium", "high", "critical"] = "medium"
    fail_on_incomplete_coverage: bool = True
    out_dir: str = "./wildginger_output"


class PoCConfig(StrictModel):
    generate: bool = True
    # Attach a deterministic MALICIOUS sample request (curl) to reachable endpoint
    # findings — built from the parsed route/params, not an LLM. See endpoints.py.
    include_sample_request: bool = True
    # LLM fallback: when the DETERMINISTIC call-graph walk can't tie a finding to an
    # HTTP route (dynamic routing, framework magic, deep call chains), ask a model to
    # find the route in the code. Its answer must cite a real file:line that we then
    # VERIFY against source — an unverified/guessed route is discarded (never shipped),
    # falling back to the generic sketch. So a red teamer gets method/path/headers/body
    # for far more findings, with no fabricated endpoints.
    llm_route_resolution: bool = True
    route_resolver_role: str = "trace"   # role (→ model) used for the LLM fallback




class MLflowConfig(StrictModel):
    """MLflow experiment tracking + tracing for every LLM call (Bedrock + Azure/GPT).
    A safe no-op when ``enabled`` is false or if mlflow isn't importable."""
    enabled: bool = False
    tracking_uri: str | None = None          # None = MLflow default (./mlruns)
    experiment: str = "WildGinger"  # default experiment (CLI --experiment overrides)
    experiment_per_target: bool = False      # give each scanned app its own experiment
    run_name: str | None = None
    tracing: bool = True                     # emit spans → Traces/Overview/Sessions
    log_io: bool = False                     # opt in to prompt+response text in spans
    log_prompts: bool = True                 # register stage templates in the Prompt Registry



class MonitoringConfig(StrictModel):
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class VersioningConfig(StrictModel):
    prompts_version: str = "v1"
    taxonomy_version: str = "v1"



# ──────────────────────────────────────────────────────────────────────────────
# Top-level settings
# ──────────────────────────────────────────────────────────────────────────────

class Settings(StrictModel):
    # Which file this config was loaded from, for the run's record. A PRIVATE attribute,
    # deliberately not a field: `config_hash()` hashes the model dump, so a path in there
    # would make two byte-identical configs at different paths hash differently — a rename
    # would then read as a config change and refuse a `--resume`. See load_settings.
    _config_source: str | None = PrivateAttr(default=None)

    aws: AWSConfig = Field(default_factory=AWSConfig)
    models: dict[str, ModelProfile]
    roles: dict[str, RoleConfig]
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reachability: ReachabilityConfig = Field(default_factory=ReachabilityConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)

    scope: ScopeConfig = Field(default_factory=ScopeConfig)

    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    hunting: HuntingConfig = Field(default_factory=HuntingConfig)
    seeding: SeedingConfig = Field(default_factory=SeedingConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    sanitization: SanitizationConfig = Field(default_factory=SanitizationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)

    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    suppressions: SuppressionsConfig = Field(default_factory=SuppressionsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    poc: PoCConfig = Field(default_factory=PoCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


    # ── cross-field validation ────────────────────────────────────────────────
    @field_validator("roles")
    @classmethod
    def _roles_reference_known_models(cls, roles: dict[str, RoleConfig], info: Any):
        models = info.data.get("models") or {}
        for role_name, role in roles.items():
            if role.model not in models:
                raise ValueError(
                    f"role '{role_name}' references unknown model '{role.model}'. "
                    f"Known models: {sorted(models)}"
                )
        return roles

    @model_validator(mode="after")
    def _referenced_roles_exist(self) -> "Settings":
        """Every configured role REFERENCE must name a real role.

        A typo in one of these used to disable a whole stage silently at runtime —
        e.g. a misspelled ``dedupe.semantic_role`` skipped the entire AI dedupe pass,
        which looks identical in the output to "found no duplicates". Same class of
        bug as an unknown config key, so it fails the same way: at load, by name.

        Two conditions must BOTH hold before we reject a reference: the feature is
        enabled, AND the role name was set explicitly in the config. A DEFAULT role
        name is allowed to be absent — those features degrade gracefully at runtime,
        and failing on them would break every minimal config (e.g. ``poc`` defaults to
        ``route_resolver_role: trace``, which a hunt-only config legitimately lacks).
        What we're catching is a TYPO: something the user wrote that means nothing.
        """
        referenced: list[tuple[str, str, bool]] = [
            ("dedupe.semantic_role", self.dedupe.semantic_role,
             self.dedupe.semantic and "semantic_role" in self.dedupe.model_fields_set),
            ("poc.route_resolver_role", self.poc.route_resolver_role,
             self.poc.llm_route_resolution
             and "route_resolver_role" in self.poc.model_fields_set),
        ]
        for field_path, role, should_check in referenced:
            if should_check and role and role not in self.roles:
                raise ValueError(
                    f"{field_path} = '{role}' is not a configured role. "
                    f"Known roles: {sorted(self.roles)}"
                )
        # Judges are validated here too rather than silently dropped at runtime.
        unknown_judges = [j for j in self.validation.judges if j not in self.roles]
        if unknown_judges:
            raise ValueError(
                f"validation.judges references unknown role(s) {unknown_judges}. "
                f"Known roles: {sorted(self.roles)}"
            )
        return self

    @model_validator(mode="after")
    def _model_endpoints_are_allowed(self) -> "Settings":
        """Every effective HTTP API endpoint must be on the egress allowlist.

        This is the load-time half of the egress control (``build_provider`` enforces it
        again at construction, as defence in depth). Failing here means a config that
        would talk to an unapproved destination cannot even reach ``validate``,
        ``dry-run`` or preflight — no discovery, no file reads, no model call.

        Two rules:

        1. Resolve default/environment URLs before matching scheme, host, port and
           path boundaries against ``governance.allowed_model_endpoints``. The
           default list is EMPTY, so every HTTP API destination must be written down.
        2. Plaintext ``http://`` is permitted only where the bytes do not cross the
           public internet — loopback, or an address that is not publicly routable (a LAN
           GPU box, a Tailscale peer). Demanding TLS from a self-hosted inference server
           with no domain and no CA would just push people into disabling the allowlist or
           faking certificates, which is worse than the risk it removes. A publicly
           routable plaintext destination IS refused, and a plaintext *hostname* is too,
           because we do not resolve DNS while validating config. Permitted plaintext is
           warned about at preflight and recorded in the audit log — see
           :func:`endpoint_transport`.
        """
        allowed = [p.strip() for p in self.governance.allowed_model_endpoints if p.strip()]
        for name, mp in self.models.items():
            base_url = resolve_model_base_url(mp.provider, mp.base_url)
            url = base_url or ""
            if url and mp.provider == "openai_gpt":
                url = url.format(deployment=mp.model_id)
            if not url:
                continue
            transport = endpoint_transport(url)
            if transport == "public":
                raise ValueError(
                    f"models.{name}.base_url '{url}' sends plaintext http:// to a "
                    f"publicly routable address. The request body contains your source "
                    f"code, so that is refused. Use https://, or an address on your own "
                    f"machine or private network (loopback, 10.x, 172.16-31.x, "
                    f"192.168.x, or a Tailscale/CGNAT 100.64.x address).")
            if transport == "hostname":
                raise ValueError(
                    f"models.{name}.base_url '{url}' sends plaintext http:// to a "
                    f"hostname. Whether that is safe depends on what DNS answers, and "
                    f"this check deliberately does not resolve names while validating "
                    f"config. Use https://, or the literal IP address of the machine "
                    f"(e.g. http://192.168.50.10:8000/v1).")
            if not model_endpoint_allowed(url, allowed):
                raise ValueError(
                    f"models.{name}.base_url '{url}' is not permitted by "
                    f"governance.allowed_model_endpoints "
                    f"({allowed or 'empty — no custom endpoints allowed'}).\n"
                    f"  This is the egress allowlist: every destination this tool may "
                    f"send code to has to be listed there explicitly.\n"
                    f"  To allow it, add a prefix such as "
                    f"'{_url_prefix_hint(url)}' to governance.allowed_model_endpoints.")
            # Keep the approved default/environment destination in the effective config
            # for hashing, audit output, and construction even if the environment changes.
            mp.base_url = base_url
        return self

    @model_validator(mode="after")
    def _tool_mode_is_usable(self) -> "Settings":
        """No TOOL-USING role may be given a model that cannot use tools.

        Six roles are built with ``with_tools=True``, and each is instructed to navigate:

          * ``hunt`` — the strongest case, because it cannot merely degrade:
            ``quality.thoroughness_min`` defaults to 0.7 and ``hunt._score_thoroughness``
            awards at most 0.4 to a cell that made no successful tool calls, so NO cell
            could ever be marked ANALYZED. Every cell would be re-hunted until the
            shallow-plateau rule gave up on it, producing a scan that is 100% "thin
            coverage" and finds almost nothing.
          * ``validate`` and every role in ``validation.judges`` — ``VALIDATE_PROMPT``
            tells the judge to "investigate the cited code independently with your
            read-only tools". Without them a judge votes on the hunter's claim text
            alone, which is exactly the rubber-stamping the adversarial panel exists to
            prevent — and it votes with full confidence while doing it.
          * ``recon``, ``trace`` and the PoC route resolver — each maps or verifies
            something in the source; with no tools they answer from the prompt alone.

        ``dedupe`` is deliberately excluded: it is built with
        ``with_tools=False`` and reasons purely over text we hand it, so
        ``tool_mode: none`` is legitimate (and marginally cheaper) there.
        """
        tool_using: list[str] = ["hunt", "recon", "trace", "validate"]
        tool_using.extend(self.validation.judges)
        if self.poc.llm_route_resolution and self.poc.route_resolver_role:
            tool_using.append(self.poc.route_resolver_role)
        for role_name in dict.fromkeys(tool_using):     # de-dup, keep order
            rc = self.roles.get(role_name)
            if rc is None:
                continue
            mp = self.models.get(rc.model)
            if mp is None or mp.tool_mode != "none":
                continue
            if role_name == "hunt":
                raise ValueError(
                    f"roles.hunt uses model '{rc.model}' with tool_mode: none, but the "
                    f"hunter has to navigate the code with tools: a cell with no "
                    f"successful tool calls scores at most 0.4 thoroughness and "
                    f"quality.thoroughness_min is {self.quality.thoroughness_min}, so no "
                    f"cell could ever be analyzed. Use tool_mode: native (or react for "
                    f"an endpoint with no tool API).")
            raise ValueError(
                f"roles.{role_name} uses model '{rc.model}' with tool_mode: none, but "
                f"that role is given read-only navigation tools and its prompt instructs "
                f"it to use them (a judge is told to verify the cited code itself). "
                f"Without tools it would answer from the prompt text alone while "
                f"reporting full confidence. Use tool_mode: native, or react for an "
                f"endpoint with no tool API. Only the dedupe role runs "
                f"tool-free.")
        return self

    def duplicate_judge_models(self) -> list[tuple[str, str, str]]:
        """Judge roles that resolve to the SAME model: ``[(role_a, role_b, model)]``.

        Two roles pointing at one ``models:`` entry satisfies a "cross-model" quorum
        with a single model, so correlated errors from that one model look like
        independent agreement. Reported rather than raised: running one model twice at
        different temperatures is a legitimate (if weaker) setup, and a deployment with
        access to a single model should not be locked out of quorum entirely. Duplicate
        judge *role names*, which add literally nothing, ARE rejected outright.
        """
        seen: dict[str, str] = {}
        clashes: list[tuple[str, str, str]] = []
        for j in self.validation.judges:
            rc = self.roles.get(j)
            if rc is None:
                continue
            if rc.model in seen:
                clashes.append((seen[rc.model], j, rc.model))
            else:
                seen[rc.model] = j
        return clashes

    def effective_unreachable_handling(self) -> str:
        """How unreachable findings are handled, honouring the deprecated alias.

        ``reachability.unreachable_handling`` wins when it was set explicitly. Otherwise
        ``validation.require_reachable: false`` means "don't demote", i.e. ``report``.
        """
        rc = self.reachability
        if "unreachable_handling" in rc.model_fields_set:
            return rc.unreachable_handling
        if not self.validation.require_reachable:
            return "report"
        return rc.unreachable_handling

    def unpriced_role_models(self) -> list[str]:
        """Models a role can invoke that have NO price configured.

        The hard cost ceiling silently cannot fire for these: ``estimate_cost`` returns
        $0.00, so ``cost >= max_cost_usd`` is never true and ``budget.max_cost_usd`` —
        documented as a "HARD spend stop" — becomes unbounded spending. ``max_cost_usd``
        is ``gt=0``, so there is no "ceiling disabled" configuration in which that is
        harmless.

        ``billing: free`` models are EXCLUDED, because for them $0.00 is the truth rather
        than missing data — a model on your own hardware bills nothing, so there is no
        spend for a ceiling to cap. That exemption is deliberately an explicit,
        per-model declaration: the alternative was users setting
        ``price_per_1m_input: 0.000001`` to get past this check, which re-opens the hole
        for a genuinely metered model. Preflight prints and audit-logs every free model
        so the run's record still shows that the cost ceiling did not apply to it (see
        :meth:`free_billing_models`).

        Reported rather than raised, so ``validate`` / ``dry-run`` and minimal test
        configs still load; ``scan`` refuses in preflight, before spending anything.
        """
        return sorted({
            rc.model for rc in self.roles.values()
            if rc.model in self.models
            and self.models[rc.model].billing != "free"
            and self.models[rc.model].price_per_1m_input <= 0.0
            and self.models[rc.model].price_per_1m_output <= 0.0
        })

    def free_billing_models(self) -> list[str]:
        """Role models declared ``billing: free`` — the cost ceiling cannot bound them.

        Surfaced by ``validate`` and preflight so "no cost ceiling applies here" is a
        stated fact of the run rather than a silent property of the config. For these
        models only ``budget.max_time_minutes`` limits the run.
        """
        return sorted({
            rc.model for rc in self.roles.values()
            if rc.model in self.models and self.models[rc.model].billing == "free"
        })


    # ── helpers ─────────────────────────────────────────────────────────────
    def model_for_role(self, role: str) -> tuple[RoleConfig, ModelProfile]:
        """Resolve a logical role (e.g. 'hunt') to its RoleConfig + ModelProfile."""
        if role not in self.roles:
            raise KeyError(f"unknown role '{role}'. Known roles: {sorted(self.roles)}")
        rc = self.roles[role]
        return rc, self.models[rc.model]

    def config_hash(self) -> str:
        """Stable hash of the effective config for reproducibility/audit."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def config_source(self) -> str:
        """The config file this was loaded from ("<in-memory>" if constructed directly).

        Reported alongside ``config_hash`` wherever a run is described. The hash alone
        cannot answer "which config did this run use?" once there is more than one file
        (config.yaml vs config_api.yaml), and that is exactly the question someone asks
        when two reports disagree.
        """
        return self._config_source or "<in-memory>"


def _suggest_key(unknown: str, candidates: list[str]) -> str | None:
    """Closest known key to ``unknown`` (for 'did you mean' on a typo)."""
    import difflib
    matches = difflib.get_close_matches(unknown, candidates, n=1, cutoff=0.7)
    return matches[0] if matches else None


# Hostnames that always mean "this machine", so no DNS or address parsing is needed.
_LOOPBACK_NAMES = ("localhost", "localhost.localdomain")

# How a model endpoint's traffic is protected. Returned by :func:`endpoint_transport`
# and consumed in three places that must agree: the config validator (which refuses
# some of these), `scan validate`'s endpoint table, and preflight's warning + audit
# event.
EndpointTransport = Literal["https", "loopback", "private", "public", "hostname"]


def resolve_model_base_url(provider: str, base_url: str | None) -> str | None:
    """Resolve a default/environment endpoint before applying egress controls."""
    if provider in ("openai_gpt", "gpt", "openai", "azure") and base_url is None:
        base_url = os.environ.get("GPT_BASE_URL")
    resolved = base_url.strip() if base_url is not None else None
    if provider == "openrouter" and not resolved:
        return "https://openrouter.ai/api/v1"
    return resolved


def _parse_model_endpoint(url: str) -> SplitResult:
    """Parse an unambiguous HTTP API URL; credentials belong in headers, not URLs."""
    raw = url.strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port  # Access validates malformed and out-of-range ports.
    except ValueError:
        raise ValueError("model endpoint must be a valid http:// or https:// URL") from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.netloc.endswith(":") or port == 0):
        raise ValueError("model endpoint must have an http:// or https:// scheme, host and valid port")
    if parsed.username is not None or parsed.password is not None or "?" in raw or "#" in raw:
        raise ValueError("model endpoint URLs must not contain userinfo, query strings or fragments; "
                         "configure credentials separately")
    path = unquote(parsed.path)
    if (any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in raw)
            or "\\" in raw or "\\" in path
            or any(part in (".", "..") for part in path.split("/"))):
        raise ValueError("model endpoint URL contains ambiguous whitespace, separators or dot segments")
    return parsed


def model_endpoint_allowed(url: str, allowed: list[str]) -> bool:
    """Match an exact origin and a path subtree, never a textual authority prefix."""
    endpoint = _parse_model_endpoint(url)
    origin = (endpoint.scheme, endpoint.hostname,
              endpoint.port or (443 if endpoint.scheme == "https" else 80))
    path = unquote(endpoint.path).rstrip("/")
    for prefix in allowed:
        entry = _parse_model_endpoint(prefix)
        entry_origin = (entry.scheme, entry.hostname,
                        entry.port or (443 if entry.scheme == "https" else 80))
        entry_path = unquote(entry.path).rstrip("/")
        if origin == entry_origin and (path == entry_path or path.startswith(entry_path + "/")):
            return True
    return False


def endpoint_transport(url: str) -> EndpointTransport:
    """Classify how ``url``'s traffic is protected, for the egress rules.

    ``https`` is encrypted. Non-HTTP schemes and credential-bearing URLs are refused. For
    plaintext ``http://`` the question is *where the bytes go*, because the request body
    is the customer's source code:

      * ``loopback``  — 127.0.0.0/8, ::1, ``localhost``, 0.0.0.0. Never leaves the host.
      * ``private``   — any other address that is **not publicly routable**: RFC1918
        (10/8, 172.16/12, 192.168/16), link-local, IPv6 ULA (fd00::/7) and CGNAT
        (100.64/10). Permitted, but warned about and recorded — the bytes do cross a
        network, just not the internet.
      * ``public``    — a publicly routable literal address. Refused.
      * ``hostname``  — a name rather than a literal. Refused for plaintext, because
        resolving DNS at config-load time would make the decision depend on whatever the
        resolver answers at that moment (and on whoever controls it).

    The test is ``not ip.is_global``, not ``ip.is_private``. That distinction is load
    bearing: Python reports CGNAT 100.64.0.0/10 — which is **Tailscale's** range, a very
    normal way to reach a GPU box — as ``is_private=False``, so an ``is_private`` rule
    would refuse it; while it reports the *public* documentation block 203.0.113.0/24 as
    ``is_private=True``. "Is this address publicly routable" is the property we actually
    care about, and ``is_global`` answers exactly that for both IPv4 and IPv6.
    """
    import ipaddress

    parsed = _parse_model_endpoint(url)
    if parsed.scheme == "https":
        return "https"
    host = parsed.hostname
    if host in _LOOPBACK_NAMES:
        return "loopback"
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return "hostname"
    if addr.is_loopback or addr.is_unspecified:
        return "loopback"
    return "public" if addr.is_global else "private"


def _url_prefix_hint(url: str) -> str:
    """A scheme+host prefix suggestion for the allowlist error message."""
    parsed = _parse_model_endpoint(url)
    return f"{parsed.scheme}://{parsed.netloc}/"



def _model_at(loc: tuple) -> type[BaseModel] | None:
    """Resolve a pydantic error ``loc`` path to the model that owns it.

    Handles mapping fields: ``models`` is ``dict[str, ModelProfile]``, so in the
    path ``('models', 'opus48', 'price_per_1m_input')`` the ``'opus48'`` segment is
    a user-chosen key, not a field name, and must be skipped.
    """
    model: type[BaseModel] | None = Settings
    skip_next_key = False
    for part in loc:
        if model is None:
            return None
        if skip_next_key:
            # This segment is a mapping KEY; the model is already the value model.
            skip_next_key = False
            continue
        fields = getattr(model, "model_fields", None) or {}
        if part not in fields:
            return None
        ann = fields[part].annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            model = ann
            continue
        # dict[str, SomeModel] → descend to the value model, skipping the key.
        args = [a for a in get_args(ann) if isinstance(a, type) and issubclass(a, BaseModel)]
        if args:
            model = args[0]
            skip_next_key = True
            continue
        return None
    return model


def _format_validation_error(exc: ValidationError, path: Path) -> str:
    """Turn pydantic's error list into an actionable message.

    Unknown keys (the common typo case) get a 'did you mean' hint naming the
    closest valid key at that exact location in the config tree.
    """
    lines = [f"invalid config: {path}"]
    errors = exc.errors()
    unknown_keys = [e for e in errors if e["type"] == "extra_forbidden"]
    for err in errors:
        loc = ".".join(str(x) for x in err["loc"])
        if err["type"] == "extra_forbidden":
            # Resolve the PARENT model so we can list the keys valid at this spot.
            model = _model_at(tuple(err["loc"][:-1]))
            valid = sorted(getattr(model, "model_fields", {}) or {}) if model else []
            hint = _suggest_key(str(err["loc"][-1]), valid) if valid else None
            msg = f"  unknown key '{loc}'"
            if hint:
                msg += f" — did you mean '{hint}'?"
            elif valid:
                msg += f" — valid keys here: {', '.join(valid)}"
            lines.append(msg)
        elif unknown_keys:
            # An unknown key makes its whole sub-model fail to build, which can
            # cascade into confusing secondary errors (e.g. "role references unknown
            # model" because `models` never validated). Mark them as such instead of
            # sending the user chasing a bug that isn't there.
            lines.append(f"  {loc}: {err['msg']}  (likely a knock-on of the unknown key above)")
        else:
            lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load + validate a config file into a typed :class:`Settings`.

    ``config_path`` None means "work it out": see :func:`resolve_config_path` for the
    precedence. The old default was the relative string ``"config.yaml"``, which only
    resolved when the process happened to start in the package directory.

    Raises a clear error if the file is missing or fails validation. Because every
    config model forbids unknown keys (:class:`StrictModel`), a typo is reported by
    name with a 'did you mean' suggestion rather than being silently ignored.

    The loaded file's path is remembered on the returned object (``config_source``) so a
    run's record can name WHICH config produced it. It is a private attribute, not a
    field, on purpose: ``config_hash()`` hashes the model dump, and two byte-identical
    configs at different paths must hash the same — otherwise renaming a file would look
    like a config change and refuse a resume.
    """
    p = Path(config_path) if config_path is not None else resolve_config_path()
    if not p.exists():
        raise FileNotFoundError(_config_not_found_message(p, source="load_settings"))
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        # ValueError so the CLI's existing handler prints it without a traceback.
        raise ValueError(_format_validation_error(exc, p)) from exc
    settings._config_source = str(p)
    return settings


# Environment variable holding a default config path, so a session that works against an
# alternate config (e.g. config_local.yaml) does not have to repeat --config on every
# command. Lower precedence than an explicit --config, higher than any search path.
CONFIG_ENV_VAR = "WILDGINGER_CONFIG"

# The config shipped inside the package, used as the last resort so the tool works when
# invoked from a directory that is not the package's own.
PACKAGED_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    """Decide which config file to load, in a documented precedence order.

    1. ``explicit`` — the ``--config`` argument;
    2. ``$WILDGINGER_CONFIG``;
    3. ``./config.yaml`` in the current working directory;
    4. the ``config.yaml`` shipped inside the package.

    An explicitly named file that does not exist is a HARD error — it is never quietly
    replaced by one of the fallbacks. Silently scanning with a different config than the
    one you named is the worst possible outcome here: every model, price, threshold and
    egress rule could differ, and the report would look perfectly normal.

    The last resort exists because the old default was the relative string
    ``"config.yaml"``, which only resolved when the process happened to be started from
    the package directory — running the CLI from a repo root failed with "config file not
    found" even though the shipped config was right there.
    """
    import os

    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(_config_not_found_message(p, source="--config"))
        return p

    from_env = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if from_env:
        p = Path(from_env).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                _config_not_found_message(p, source=f"${CONFIG_ENV_VAR}"))
        return p

    cwd_config = Path("config.yaml")
    if cwd_config.exists():
        return cwd_config
    if PACKAGED_CONFIG.exists():
        return PACKAGED_CONFIG
    raise FileNotFoundError(_config_not_found_message(cwd_config, source="default"))


def _config_not_found_message(missing: Path, *, source: str) -> str:
    """A not-found message that says where we looked and what is actually there.

    Juggling several configs (config.yaml, config_local.yaml) makes a
    typo in the filename the most likely failure, so list the candidates rather than
    making the user go looking.
    """
    lines = [f"config file not found: {missing} (from {source})"]
    nearby: list[str] = []
    seen: set[Path] = set()
    for directory in (missing.parent if str(missing.parent) else Path("."),
                      Path.cwd(), PACKAGED_CONFIG.parent):
        try:
            for candidate in sorted(Path(directory).glob("config*.y*ml")):
                # De-dup on the RESOLVED path: the same file reached via a relative and an
                # absolute directory would otherwise be listed twice, which makes the
                # message look broken and the suggestion less trustworthy.
                real = candidate.resolve()
                if real in seen:
                    continue
                seen.add(real)
                nearby.append(str(candidate))
        except OSError:
            continue
    if nearby:
        lines.append("  config files I can see:")
        lines.extend(f"    {n}" for n in nearby[:10])
        hint = _suggest_key(missing.name, [Path(n).name for n in nearby])
        if hint and hint != missing.name:
            lines.append(f"  did you mean '{hint}'?")
    lines.append(f"  precedence: --config > ${CONFIG_ENV_VAR} > ./config.yaml > "
                 f"{PACKAGED_CONFIG}")
    return "\n".join(lines)
