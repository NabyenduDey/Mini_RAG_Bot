"""
Step 3 of RAG: HTTP client to the local Ollama `/api/chat` endpoint (keep-alive, shared client).
"""

from __future__ import annotations

import logging
import threading

import httpx

from . import config

logger = logging.getLogger(__name__)

_http_lock = threading.Lock()
_http_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is not None:
        return _http_client
    with _http_lock:
        if _http_client is None:
            timeout = httpx.Timeout(config.OLLAMA_HTTP_TIMEOUT)
            _http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
    return _http_client


async def close_client() -> None:
    global _http_client
    with _http_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


async def complete_chat(*, system: str, user: str) -> str:
    """Non-streaming chat completion; returns assistant text."""
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    opts: dict = {"temperature": config.OLLAMA_TEMPERATURE}
    if config.OLLAMA_NUM_PREDICT:
        opts["num_predict"] = int(config.OLLAMA_NUM_PREDICT)
    if config.OLLAMA_NUM_CTX:
        opts["num_ctx"] = int(config.OLLAMA_NUM_CTX)

    payload: dict = {
        "model": config.OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": opts,
    }

    client = await get_client()
    r = await client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


def ollama_connect_error_message(exc: httpx.ConnectError) -> str:
    logger.warning("Ollama unreachable at %s (%s)", config.OLLAMA_BASE_URL, exc)
    base = config.OLLAMA_BASE_URL
    model = config.OLLAMA_CHAT_MODEL
    return (
        "Cannot connect to Ollama (nothing is listening at that address).\n\n"
        f"Configured URL: {base}\n"
        f"Model: {model}\n\n"
        f"Check (on the machine running this bot): curl -sS {base}/api/version\n\n"
        "127.0.0.1 is this machine only. Run the bot where Ollama runs, or set "
        "OLLAMA_BASE_URL to a reachable host (SSH tunnel, LAN IP, or Ollama on the server).\n\n"
        f"If Ollama is local: start the Ollama app, then run: ollama pull {model}"
    )
