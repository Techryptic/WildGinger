# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared surface for the two OpenAI-style providers (``openai_gpt``, ``openai_compat``).

Holds the pieces that MUST behave identically in both, because a heuristic that exists
twice is a heuristic that eventually disagrees with itself:

  * the **ReAct tool protocol** — tools implemented in the prompt, for endpoints with no
    tool-calling API;
  * the **"this model rejects `temperature`" detector**, which drives a one-shot self-heal
    in both providers.

(Bedrock keeps its own parameter-rejection detector: its wording comes from a different
API and the two sets of marker phrases are genuinely not the same.)

WHY THE ReAct PROTOCOL EXISTS. Some endpoints have no tool-calling API at all: the
internal Apigee gateway (``openai_gpt``) is a bare chat-completions surface, and plenty of
local servers (llama.cpp builds, older vLLM, small models behind Ollama) either ignore a
``tools`` array or 400 on it. For those we implement tools in the PROMPT: the model is
told to answer with a single line

    ACTION: tool_name({"arg": value})

which we parse back into a normalized ``tool_use`` block, so the SAME agent loop that
drives Bedrock's native tool use also lets those models navigate code.

This module had already been through several rounds of hardening inside
``openai_gpt.py``; duplicating it into a second provider would have duplicated the bug
history with it. The two regressions that cost real money:

  * a greedy ``(\\{.*\\})`` DOTALL regex swallowed prose up to the LAST ``}`` anywhere in
    the reply, so any text containing a brace corrupted the tool arguments into an
    unparseable blob that then silently became ``{}``;
  * the keyword itself was matched too strictly, so ``**ACTION:**`` / a leading list
    marker / lowercase / a missing colon made the whole line unparseable. The ACTION was
    then returned as a FINAL ANSWER: the cell produced no findings and no tool calls,
    scored 0 thoroughness, and was re-hunted (re-paid) on the next pass.

``openai_gpt`` re-exports the names its own tests import, so those keep working.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base import ToolSpec

# Locates the START of an ACTION directive:  ACTION: read_file({...})
# Only the tool name + opening brace are matched here; the JSON argument object is
# extracted by BRACE BALANCING (_extract_json_object) rather than a regex.
#
# Tolerant of the decorations models actually emit around the keyword: markdown
# bolding (`**ACTION:**`), a leading list marker or heading, lowercase, and a missing
# colon.
_ACTION_START_RE = re.compile(
    r"(?im)^[\s>*\-#]*\**\s*ACTION\**\s*:?\s*\**\s*([A-Za-z_]\w*)\s*\(\s*(?=\{)")


def _extract_json_object(text: str, start: int) -> tuple[str | None, int]:
    """Extract one balanced ``{...}`` JSON object from ``text`` starting at ``start``.

    Returns ``(json_text, end_index)`` or ``(None, start)`` if the braces never
    balance. String literals (and their escapes) are tracked so a ``}`` inside a
    quoted value doesn't terminate the object early.
    """
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
    return None, start


def _parse_action(text: str) -> tuple[str, dict[str, Any], bool] | None:
    """Parse the FIRST ``ACTION: name({...})`` directive in ``text``.

    Returns ``(tool_name, args, parse_ok)`` or None when there is no ACTION line.
    ``parse_ok`` is False when the argument object was present but malformed, so the
    caller can surface a real error instead of silently running the tool with ``{}``.
    """
    m = _ACTION_START_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    raw, _end = _extract_json_object(text, m.end())
    if raw is None:
        return name, {}, False
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return name, {}, False
    if not isinstance(args, dict):
        return name, {}, False
    return name, args, True


def _tools_instructions(tools: list[ToolSpec]) -> str:
    """Describe the available tools + the ReAct ACTION protocol for the model."""
    lines = [
        "You have READ-ONLY tools to explore the code. To use one, reply with a "
        "SINGLE line exactly in this form and nothing else:",
        "  ACTION: <tool_name>({\"arg\": value, ...})",
        "After you receive the tool result, continue. When you are DONE and ready to "
        "give your final answer, reply normally WITHOUT an ACTION line.",
        "",
        "Available tools:",
    ]
    for t in tools:
        try:
            schema = json.dumps(t.input_schema.get("properties", {}))
        except Exception:  # noqa: BLE001
            schema = "{}"
        lines.append(f"  - {t.name}: {t.description} args={schema}")
    return "\n".join(lines)


def malformed_action_hint(name: str) -> str:
    """The message handed back to a model whose ACTION arguments would not parse.

    Running the tool with ``{}`` produced a confusing "'path' is required" error the
    model could not act on, so tell it plainly what went wrong and let it retry.
    """
    return (f"[harness] Could not parse the arguments of your "
            f"ACTION: {name}(...) call. Reply with a SINGLE line: "
            f'ACTION: {name}({{"arg": "value"}}) — valid JSON, no prose '
            f"on that line.")


# ── "this model rejects an explicit temperature" ───────────────────────────────
# Phrases the OpenAI-family APIs use when a deployment refuses `temperature` (o-series and
# GPT-5-class reasoning models accept only their default; some local servers reject it
# too). Matched NARROWLY — the parameter name alone is not enough — so an unrelated
# failure whose message merely mentions temperature cannot silently drop the parameter for
# a whole run.
_TEMPERATURE_REJECTION_MARKERS = (
    "unsupported_value",
    "does not support",
    "unsupported parameter",
    "not supported with this model",
    "unsupported_parameter",
)


def _is_temperature_rejection(exc_or_message: Any) -> bool:
    """True if this error specifically means "this model won't accept temperature".

    Accepts an exception or a string, because the two providers reached this check from
    different places (one had a ``ProviderError`` in hand, the other a formatted message)
    and having one function serve both is the point of putting it here.
    """
    low = str(exc_or_message or "").lower()
    if "temperature" not in low:
        return False
    return any(marker in low for marker in _TEMPERATURE_REJECTION_MARKERS)
