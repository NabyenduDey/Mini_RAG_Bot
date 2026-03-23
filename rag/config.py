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
EMBEDDING_BATCH_SIZE = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")))
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "").strip() or None

# Query embedding cache (retrieval only; ingest still uses batch encode). 0 = disabled.
_eq_ttl = os.environ.get("EMBEDDING_QUERY_CACHE_TTL_SEC", "120").strip()
try:
    EMBEDDING_QUERY_CACHE_TTL_SEC = int(_eq_ttl) if _eq_ttl else 120
except ValueError:
    EMBEDDING_QUERY_CACHE_TTL_SEC = 120

_eq_max = os.environ.get("EMBEDDING_QUERY_CACHE_MAX", "48").strip()
try:
    EMBEDDING_QUERY_CACHE_MAX = max(0, int(_eq_max) if _eq_max else 48)
except ValueError:
    EMBEDDING_QUERY_CACHE_MAX = 48

RAG_WARMUP_ON_START = os.environ.get("RAG_WARMUP_ON_START", "true").lower() in (
    "1",
    "true",
    "yes",
)

CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "480"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "90"))
TOP_K = int(os.environ.get("TOP_K", "2"))
# Fetch extra neighbors then drop those much worse than the best match (reduces off-topic policy chunks on recipe queries).
TOP_K_CANDIDATES = int(os.environ.get("TOP_K_CANDIDATES", str(max(TOP_K, 12))))


def _parse_distance_margin() -> float:
    """
    Cosine/L2-style distance band around the best hit. Chunks farther than best + margin are dropped.

    Important: an *empty* value in .env must NOT disable filtering (that lets every doc into CONTEXT).
    Use RETRIEVAL_DISTANCE_MARGIN=0 or off to disable.
    """
    raw = os.environ.get("RETRIEVAL_DISTANCE_MARGIN")
    if raw is None:
        return 0.20
    s = str(raw).strip()
    if not s:
        return 0.20
    if s.lower() in ("0", "off", "false", "none"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.20


RETRIEVAL_DISTANCE_MARGIN = _parse_distance_margin()
# Stop adding chunks after a large jump in distance (removes the next “cluster” of unrelated docs).
_elb = os.environ.get("RETRIEVAL_ELBOW_GAP", "0.13").strip()
try:
    RETRIEVAL_ELBOW_GAP = float(_elb) if _elb else 0.13
except ValueError:
    RETRIEVAL_ELBOW_GAP = 0.13

# Lower temperature reduces made-up “bonus” policy lines when CONTEXT is narrow.
_ot = os.environ.get("OLLAMA_TEMPERATURE", "0.2").strip()
try:
    OLLAMA_TEMPERATURE = float(_ot) if _ot else 0.2
except ValueError:
    OLLAMA_TEMPERATURE = 0.2

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
# Optional Ollama generation caps (smaller / faster replies). See ollama.com docs for options.
OLLAMA_NUM_PREDICT = os.environ.get("OLLAMA_NUM_PREDICT", "").strip()
OLLAMA_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX", "").strip()
OLLAMA_HTTP_TIMEOUT = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "180"))

# Image → text: local Hugging Face (or optional Tesseract)
# blip2 (recommended for screenshot / UI text) | llava | clip_interrogator |
# blip_vqa | blip_caption | tesseract
# Aliases: blip → blip2; clip | clip-interrogator → clip_interrogator
_raw_ib = os.environ.get("IMAGE_TEXT_BACKEND", "blip2").strip().lower()
if _raw_ib == "blip":
    IMAGE_TEXT_BACKEND = "blip2"
elif _raw_ib in ("clip", "clip-interrogator"):
    IMAGE_TEXT_BACKEND = "clip_interrogator"
elif _raw_ib in ("blip-2", "blip_2"):
    IMAGE_TEXT_BACKEND = "blip2"
else:
    IMAGE_TEXT_BACKEND = _raw_ib

IMAGE_TEXT_MODEL = os.environ.get("IMAGE_TEXT_MODEL", "").strip()
HF_IMAGE_DEVICE = os.environ.get("HF_IMAGE_DEVICE", "auto").strip()
HF_IMAGE_MAX_NEW_TOKENS = int(os.environ.get("HF_IMAGE_MAX_NEW_TOKENS", "256"))
# BLIP-2: beam search + upscaling small screenshots improves reading UI / search-bar text.
BLIP2_NUM_BEAMS = max(1, int(os.environ.get("BLIP2_NUM_BEAMS", "5")))
BLIP2_UPSCALE_MIN_EDGE = max(0, int(os.environ.get("BLIP2_UPSCALE_MIN_EDGE", "640")))

# Optional: full path to tesseract.exe on Windows (only if IMAGE_TEXT_BACKEND=tesseract)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Rebuild index when knowledge file fingerprints change
AUTO_REINGEST = os.environ.get("AUTO_REINGEST", "true").lower() in (
    "1",
    "true",
    "yes",
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
