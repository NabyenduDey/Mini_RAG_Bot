"""Normalize LLM text for Telegram: tighter spacing, no runaway blank lines."""

from __future__ import annotations

import re

_MULTI_BLANK = re.compile(r"\n{3,}")


def polish_reply(text: str) -> str:
    """Make bot answers easier to read in chat (clear, compact)."""
    s = (text or "").strip()
    if not s:
        return s
    s = _MULTI_BLANK.sub("\n\n", s)
    return s.strip()
