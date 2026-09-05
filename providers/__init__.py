# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic provider layer.

`base` defines the provider-blind interface + shared types; `bedrock` is the only
implementation today. A future Azure/OpenAI provider drops in here with zero
changes to orchestration or stages.
"""
from __future__ import annotations

from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    ToolUse,
    build_provider,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolSpec",
    "ToolUse",
    "build_provider",
]
