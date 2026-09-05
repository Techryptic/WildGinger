# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Agentic tool-use loop with session persistence + robust parsing.

One :class:`Agent` drives a single conversation: it calls the provider, and when
the model requests a tool (``stopReason == "tool_use"``) it runs the registered
handler, appends the result, and loops — until the model stops, a turn budget is
hit, or the budget manager says stop.

Key responsibilities (mirroring lessons from the Anthropic reference harness):
  1. **Tool dispatch** — map a model tool call to a Python handler; never execute
     shell/code. Handlers are pure-Python (navigation tools live in navigation.py).
  2. **Session persistence** — the running message list is exposed so the caller
     can checkpoint/resume mid-conversation after a transient failure.
  3. **Resume-on-error** — transient provider errors retry with exponential
     backoff (cap configurable); partial history is preserved, never lost.
  4. **Robust output parsing** — :func:`find_tagged_block` scans *backwards* for
     the last valid structured block, because agents emit the block then chatty
     prose.
  5. **Tool-output safety** — handlers return already-truncated, already-sanitized
     text (enforced by the navigation/sanitizer layers); the agent records audit
     events for every tool call.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .providers.base import LLMProvider, LLMResponse, Message, ProviderError, ToolSpec

# A tool handler takes the model-supplied args dict and returns (result_text,
# truncated_flag). It must be pure-Python and side-effect-free w.r.t. the source
# tree (read-only). Raising is allowed; the agent converts it to a tool error.
ToolHandler = Callable[[dict[str, Any]], "ToolResult"]


@dataclass
class ToolResult:
    text: str
    truncated: bool = False
    is_error: bool = False


@dataclass
class Tool:
    """A navigation tool: its schema (for the model) + its Python handler."""
    spec: ToolSpec
    handler: ToolHandler


class _TurnFailed(Exception):
    """Internal: one conversation turn could not be completed.

    Carries the running ``resume_count`` and the ``stop_reason`` the caller should
    report, so the retry bookkeeping stays inside :meth:`Agent._turn_with_retry`
    instead of leaking into the main loop.
    """

    def __init__(self, cause: BaseException, *, resume_count: int, stop_reason: str):
        super().__init__(str(cause))
        self.cause = cause
        self.resume_count = resume_count
        self.stop_reason = stop_reason


@dataclass
class AgentResult:
    """Outcome of one agent run."""
    final_text: str
    messages: list[Message]
    stop_reason: str | None
    turns: int
    tool_calls: int
    # How many of ``tool_calls`` came back as an ERROR (unknown tool, path-jail
    # rejection, binary/oversized file, bad regex, handler exception). Thoroughness
    # scoring counts *successful* navigation, and counting failures as evidence of
    # having "looked" let a cell where every single tool call failed score 1.0 and be
    # committed ANALYZED with zero reads and zero findings.
    tool_errors: int = 0
    resume_count: int = 0
    error: Optional[str] = None

    @property
    def successful_tool_calls(self) -> int:
        return max(0, self.tool_calls - self.tool_errors)

    def tagged(self, tag: str) -> str | None:
        """The LAST ``<tag>...</tag>`` anywhere in the conversation.

        ``final_text`` is only the most recent non-empty turn, and both providers happily
        return text AND a tool_use in one message — so a hunter that emitted its
        ``<findings>`` block alongside a tool call and then signed off with a sentence had
        the block overwritten and unrecoverable. ``run_hunt_cell`` then reported
        "unparseable findings block", discarded findings whose citations had ALREADY been
        verified, and re-hunted the cell at full cost on the next pass. The block is right
        there in the transcript, so look there before giving up.
        """
        hit = find_tagged_block(self.final_text, tag)
        if hit is not None:
            return hit
        for msg in reversed(self.messages or []):
            text = _assistant_text(msg)
            if not text:
                continue
            hit = find_tagged_block(text, tag)
            if hit is not None:
                return hit
        return None


@dataclass
class Agent:
    """Drives one tool-use conversation against a provider."""
    provider: LLMProvider
    role: str
    system: str
    tools: list[Tool] = field(default_factory=list)
    max_tokens: int = 4000
    temperature: float = 0.2
    max_turns: int = 100
    max_resume_attempts: int = 20
    backoff_cap_seconds: int = 300
    # Total wall-clock seconds this agent may spend SLEEPING in transport backoff for
    # one turn. Attempt-bounded retries alone were unbounded in time: 20 attempts of
    # min(2**k, 300) sums to 4110s (68 min) on a single cell, which also blows straight
    # through the 600s cell lease — so another hunter reclaims the cell and both pay
    # for it. Whichever limit is reached first ends the retry loop.
    max_retry_seconds: float = 480.0
    agent_id: str = "agent"
    audit: Any = None  # optional Audit instance
    db: Any = None     # optional Database (for llm_calls mirror)
    # Hook called after each successful provider turn with the current message
    # list, so the caller can checkpoint for resume. Signature: (messages) -> None
    on_checkpoint: Optional[Callable[[list[Message]], None]] = None

    def __post_init__(self) -> None:
        self._by_name: dict[str, Tool] = {t.spec.name: t for t in self.tools}
        self._specs: list[ToolSpec] = [t.spec for t in self.tools]

    # ── public ─────────────────────────────────────────────────────────────────
    def run(self, prompt: str, *, history: list[Message] | None = None) -> AgentResult:
        """Run the loop starting from ``prompt`` (or resuming ``history``)."""
        messages: list[Message] = list(history) if history else [Message.user_text(prompt)]
        turns = 0
        tool_calls = 0
        tool_errors = 0
        resume_count = 0
        last_text = ""

        while turns < self.max_turns:
            turns += 1
            # Transport retries happen INSIDE this call and never consume a turn — a
            # retry is not a conversation step. See _turn_with_retry.
            try:
                resp, resume_count = self._turn_with_retry(messages, resume_count)
            except _TurnFailed as failure:
                resume_count = failure.resume_count
                # Distinguish "the transport kept failing" from "the model gave a bad
                # answer": both used to surface as a generic error, which reads as a
                # quality problem and hides a throttle storm.
                return AgentResult(
                    final_text=last_text, messages=messages,
                    stop_reason=failure.stop_reason,
                    turns=turns, tool_calls=tool_calls, tool_errors=tool_errors,
                    resume_count=resume_count,
                    error=str(failure.cause),
                )

            if resp.text:
                last_text = resp.text

            # Append the assistant message (incl. any tool_use blocks) to history.
            if resp.assistant_message is not None:
                messages.append(resp.assistant_message)

            if self.on_checkpoint:
                self.on_checkpoint(messages)

            if not resp.wants_tool:
                return AgentResult(
                    final_text=last_text, messages=messages, stop_reason=resp.stop_reason,
                    turns=turns, tool_calls=tool_calls, tool_errors=tool_errors,
                    resume_count=resume_count,
                )

            # Run each requested tool and append a single user message of results.
            result_blocks: list[dict[str, Any]] = []
            for tu in resp.tool_uses:
                tool_calls += 1
                block = self._dispatch_tool(tu.tool_use_id, tu.name, tu.input)
                if block.get("status") == "error":
                    tool_errors += 1
                result_blocks.append(block)
            messages.append(Message(role="user", content=result_blocks))

        # Turn budget exhausted.
        return AgentResult(
            final_text=last_text, messages=messages, stop_reason="max_turns",
            turns=turns, tool_calls=tool_calls, tool_errors=tool_errors,
            resume_count=resume_count,
        )

    # ── internals ──────────────────────────────────────────────────────────────
    def _turn_with_retry(self, messages: list[Message],
                         resume_count: int) -> tuple[LLMResponse, int]:
        """One conversation turn, retrying transport failures internally.

        Retries live HERE, not in :meth:`run`'s loop, because a retry is a transport
        event — not a conversation step. Previously each retry consumed one unit of
        ``max_turns``, so a throttle storm could eat 20 of a judge's 60 turns (and 20
        of ``dedupe``'s 40) before the model had said anything, then report
        ``max_turns`` as though the model had rambled.

        ``BudgetExceeded`` is deliberately NOT caught: the cost kill-switch must
        propagate instantly, never be retried.

        Returns ``(response, updated_resume_count)``. Raises :class:`_TurnFailed` when
        the error isn't retryable, the resume budget is spent, or the retry TIME budget
        (``max_retry_seconds``) is spent — whichever comes first.
        """
        started = time.monotonic()
        while True:
            try:
                return self._turn(messages), resume_count
            except ProviderError as exc:
                if not exc.retryable:
                    raise _TurnFailed(exc, resume_count=resume_count,
                                      stop_reason="error") from exc
                # Measure ACTUAL elapsed wall-clock, not the sum of intended sleeps:
                # the budget exists to bound how long a cell can be wedged (it must stay
                # under the cell lease), and the provider's own blocking time counts
                # toward that just as much as our backoff does.
                elapsed = time.monotonic() - started
                if resume_count >= self.max_resume_attempts or elapsed >= self.max_retry_seconds:
                    # Out of retries on a *transient* failure: name it as such so the
                    # CLI doesn't blame the model for a transport problem.
                    if self.audit:
                        self.audit.event("agent_retries_exhausted",
                                         agent_id=self.agent_id,
                                         attempts=resume_count,
                                         elapsed_seconds=round(elapsed, 1),
                                         error=str(exc))
                    raise _TurnFailed(exc, resume_count=resume_count,
                                      stop_reason="retries_exhausted") from exc
                resume_count += 1
                # Full jitter (AWS's recommended form). Without it, N hunters throttled
                # by the same quota all slept for exactly 2**k and retried in lockstep,
                # re-throttling each other — the classic thundering herd.
                ceiling = min(2 ** resume_count, self.backoff_cap_seconds)
                backoff = random.uniform(0.0, float(ceiling))
                # Never sleep past the remaining time budget.
                backoff = min(backoff, max(0.0, self.max_retry_seconds - elapsed))
                if self.audit:
                    self.audit.event("agent_resume", agent_id=self.agent_id,
                                     attempt=resume_count, backoff_s=round(backoff, 1),
                                     error=str(exc))
                time.sleep(backoff)
                # Retry the same turn; history is intact and no turn was consumed.

    def _turn(self, messages: list[Message]) -> LLMResponse:
        # PRE-CALL BUDGET GATE. The post-call check can only notice the ceiling once a
        # request has already been billed, so every call in flight when the ceiling is
        # crossed is paid for in full. Checking here turns N in-flight calls into N
        # not-started calls. BudgetExceeded is deliberately not a ProviderError, so
        # _turn_with_retry propagates it instantly instead of retrying it.
        if self.audit is not None:
            check = getattr(self.audit, "check_ceiling", None)
            if check is not None:
                check()
        resp = self.provider.generate(
            system=self.system,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=self._specs or None,
        )
        if self.audit:
            # Pass prompt (last user text) + response to enrich the MLflow trace
            # span (prompt-in / response-out). These are ignored by the JSONL/DB
            # audit and only used for tracing when MLflow is enabled.
            self.audit.llm_call(
                role=self.role,
                model_id=self.provider.model_id,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cache_read_tokens=getattr(resp, "cache_read_tokens", 0),
                cache_write_tokens=getattr(resp, "cache_write_tokens", 0),
                latency_ms=resp.latency_ms,
                stop_reason=resp.stop_reason,
                status="ok",
                db=self.db,
                system=self.system,
                prompt=_last_user_text(messages),
                response=resp.text,
                # Present only for providers that report what they charged (OpenRouter).
                # Logged for reconciliation; the ledger still runs on config prices.
                reported_cost_usd=getattr(resp, "reported_cost_usd", None),
            )
        return resp

    def _dispatch_tool(self, tool_use_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._by_name.get(name)
        if tool is None:
            return _tool_result_block(tool_use_id, f"[error] unknown tool '{name}'", is_error=True)
        try:
            result = tool.handler(args)
        except Exception as exc:  # noqa: BLE001 — surface as a tool error, never crash the loop
            if self.audit:
                self.audit.event("tool_error", agent_id=self.agent_id, tool=name, error=str(exc))
            return _tool_result_block(tool_use_id, f"[error] {type(exc).__name__}: {exc}", is_error=True)
        if self.audit:
            self.audit.tool_call(
                agent_id=self.agent_id, tool=name, args=args,
                result_bytes=len(result.text.encode("utf-8")), truncated=result.truncated,
            )
        return _tool_result_block(tool_use_id, result.text, is_error=result.is_error)


def _tool_result_block(tool_use_id: str, text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "text": text,
        "status": "error" if is_error else "success",
    }


def _assistant_text(msg: Any) -> str:
    """Concatenated text blocks of an assistant message (tolerates dict/dataclass)."""
    role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
    if role != "assistant":
        return ""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _last_user_text(messages: list[Message]) -> str:
    """Best-effort: the most recent user text block, for trace 'prompt' capture."""
    for m in reversed(messages):
        if m.role != "user":
            continue
        parts = [b.get("text", "") for b in m.content if b.get("type") == "text"]
        if parts:
            return "\n".join(parts)
        # tool_result-only turn: summarize so the span still has an input.
        obs = [b.get("text", "") for b in m.content if b.get("type") == "tool_result"]
        if obs:
            return "[tool results]\n" + "\n".join(obs)
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Robust output parsing: scan BACKWARDS for the last valid block.
# ──────────────────────────────────────────────────────────────────────────────

def find_tagged_block(text: str, tag: str) -> str | None:
    """Return the content of the LAST ``<tag>...</tag>`` in ``text``.

    Agents emit the structured tag, then often a chatty trailing message. Taking
    the first/naive match risks grabbing an earlier draft; we want the final one.
    DOTALL so multi-line blocks work.
    """
    if not text:
        return None
    matches = list(re.finditer(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def find_last_json(text: str) -> Any | None:
    """Best-effort: extract the last JSON object/array embedded in ``text``.

    Handles ```json fenced blocks and bare objects. Returns the parsed value or
    None. Used as a fallback when a stage expects JSON but the model wrapped it in
    prose.
    """
    if not text:
        return None
    # Prefer fenced ```json blocks (take the last).
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = list(reversed(fences)) if fences else []
    # Also try the whole text and the largest brace/bracket span.
    candidates.append(text)
    for cand in candidates:
        parsed = _try_parse_json_span(cand)
        if parsed is not None:
            return parsed
    return None


def _try_parse_json_span(text: str) -> Any | None:
    """Parse ``text``, or the LAST balanced JSON value embedded in it.

    The old implementation took ``text.find(open)`` .. ``text.rfind(close)`` — the
    WIDEST span, not the last value. That silently lost the whole payload whenever the
    model added prose containing a bracket, or emitted a draft and then a final block:

      ``'[{"a":1}]\n\nSee reference [1].'``      → None   (the trailing ``]`` widened the span)
      ``'draft: [{"a":1}] final: [{"b":2}]'``   → None
      ``'{"verdict":"NO"} ... {"verdict":"YES"}'`` → None   (a judge silently abstained)

    Now we scan RIGHT to LEFT for closing delimiters and, for each, walk back to its
    balanced opener (string- and escape-aware), returning the first span that parses.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Collect candidates right-to-left, then RANK them. "Rightmost that parses" alone is
    # not enough: prose like "See reference [1]" is itself valid JSON, so a literal
    # last-value rule would return `[1]` and quietly discard the real payload. Ranking by
    # plausibility of shape first, position second, gets both cases right.
    candidates: list[tuple[int, Any]] = []
    for end in range(len(text) - 1, -1, -1):
        if text[end] not in "}]":
            continue
        start = _matching_open(text, end)
        if start is None:
            continue
        try:
            candidates.append((start, json.loads(text[start:end + 1])))
        except (json.JSONDecodeError, ValueError):
            continue
        if len(candidates) >= _MAX_JSON_CANDIDATES:
            break
    if not candidates:
        return None
    # Highest rank wins; ties go to the RIGHTMOST (a model that revises itself means the
    # later block).
    best_start, best = max(candidates, key=lambda c: (_json_shape_rank(c[1]), c[0]))
    return best


# Every stage here expects either a list of objects (findings, dedupe groups) or a single
# object (a judge verdict, a reachability answer). Anything else is far more likely to be
# an accident of the surrounding prose than the payload.
def _json_shape_rank(value: Any) -> int:
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        return 3
    if isinstance(value, dict) and value:
        return 2
    if isinstance(value, list) and value and all(isinstance(x, list) for x in value):
        return 2       # dedupe returns a list of index lists
    if isinstance(value, (list, dict)):
        return 1
    return 0


# Bound the right-to-left scan so a huge reply cannot make this quadratic.
_MAX_JSON_CANDIDATES = 12


def _matching_open(text: str, close_idx: int) -> int | None:
    """Index of the delimiter that opens the value closing at ``close_idx``.

    Walks backwards tracking depth, and ignores delimiters inside string literals
    (honouring backslash escapes) so a brace in a quoted code snippet — which findings
    are full of — cannot throw the match off.
    """
    close_ch = text[close_idx]
    open_ch = "{" if close_ch == "}" else "["
    depth = 0
    i = close_idx
    while i >= 0:
        ch = text[i]
        if ch in "\"'":
            # Skip back over a string literal ending at i.
            quote = ch
            i -= 1
            while i >= 0:
                if text[i] == quote and not _is_escaped(text, i):
                    break
                i -= 1
            i -= 1
            continue
        if ch == close_ch:
            depth += 1
        elif ch == open_ch:
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    return None


def _is_escaped(text: str, idx: int) -> bool:
    """True if ``text[idx]`` is preceded by an odd number of backslashes."""
    n = 0
    j = idx - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 1
