"""Settings loaded from the environment (and optional .env file)."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = folder that contains `rag/`, `knowledge/`, `requirements.txt`, etc.
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load BASE_DIR/.env into os.environ (idempotent; safe if file missing)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass


_load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

_PLACEHOLDER_TOKENS = frozenset(
    {
        "your_token_from_botfather",
        "paste_your_token_here",
        "changeme",
        "xxx",
        "token",
        "here",
    }
)


def _telegram_token_is_placeholder(token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return True
    if t in _PLACEHOLDER_TOKENS:
        return True
    if "your_token" in t or "from_botfather" in t:
        return True
    return False


def _telegram_token_is_plausible(token: str) -> bool:
    """Rough shape check for Bot API tokens (digits:secret)."""
    t = token.strip()
    if ":" not in t:
        return False
    left, _, right = t.partition(":")
    if not left.isdigit() or len(left) < 8:
        return False
    if len(right) < 25:
        return False
    return True


def require_telegram_token() -> None:
    """Refresh token from env/.env; fail fast if missing, placeholder, or malformed."""
    global TELEGRAM_BOT_TOKEN
    _load_dotenv()
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_path = BASE_DIR / ".env"
    example_path = BASE_DIR / ".env.example"

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN.\n"
            "  • Open Telegram, talk to @BotFather → /newbot (or use an existing bot) → copy the API token.\n"
            f"  • Create or edit: {env_path}\n"
            "      TELEGRAM_BOT_TOKEN=123456789:AAH…your_real_token…\n"
            f"  • Tip: cp .env.example .env   then replace the placeholder (run from {BASE_DIR}).\n"
            f"  • Or export in the shell: export TELEGRAM_BOT_TOKEN='…'\n"
            f"  (.env.example template: {example_path})"
        )

    if _telegram_token_is_placeholder(TELEGRAM_BOT_TOKEN):
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is still a placeholder (e.g. your_token_from_botfather).\n"
            f"  Edit {env_path} and paste the real token from @BotFather after the = sign.\n"
            "  One line, no spaces, no quotes: TELEGRAM_BOT_TOKEN=123456789:AAH…"
        )

    if not _telegram_token_is_plausible(TELEGRAM_BOT_TOKEN):
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN does not look like a valid Bot API token.\n"
            "  Expected shape: <numeric_bot_id>:<secret> (copy exactly from @BotFather).\n"
            f"  Fix {env_path} — common mistakes: extra spaces, missing colon, or truncated copy/paste."
        )


KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", DATA_DIR / "rag.db")).resolve()

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "480"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "90"))
TOP_K = int(os.environ.get("TOP_K", "5"))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")

# Image → text: local Hugging Face (or optional Tesseract)
# blip_vqa | blip_caption | llava | clip_interrogator | tesseract
# Aliases: blip → blip_vqa; clip | clip-interrogator → clip_interrogator
_raw_ib = os.environ.get("IMAGE_TEXT_BACKEND", "blip_vqa").strip().lower()
if _raw_ib == "blip":
    IMAGE_TEXT_BACKEND = "blip_vqa"
elif _raw_ib in ("clip", "clip-interrogator"):
    IMAGE_TEXT_BACKEND = "clip_interrogator"
else:
    IMAGE_TEXT_BACKEND = _raw_ib

IMAGE_TEXT_MODEL = os.environ.get("IMAGE_TEXT_MODEL", "").strip()
HF_IMAGE_DEVICE = os.environ.get("HF_IMAGE_DEVICE", "auto").strip()
HF_IMAGE_MAX_NEW_TOKENS = int(os.environ.get("HF_IMAGE_MAX_NEW_TOKENS", "256"))

# Optional: full path to tesseract.exe on Windows (only if IMAGE_TEXT_BACKEND=tesseract)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Rebuild index when knowledge file fingerprints change
AUTO_REINGEST = os.environ.get("AUTO_REINGEST", "true").lower() in (
    "1",
    "true",
    "yes",
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
