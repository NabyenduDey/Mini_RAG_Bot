"""
Shim: starts the Telegram bot via the same entrypoint as `python -m rag.app`.

Usage (from mini_rag_telegram_bot/):
  python -m rag.Tel_bot

Or from rag/:
  python Tel_bot.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_module("rag.app", run_name="__main__")
