# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m WildGinger``."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
