"""
End-to-end answer path in three simple steps:

1. Collect question text — what the user typed plus text from the image (local HF model or Tesseract).
2. Look up similar passages in the local document index (vector search).
3. Call the local chat model (Ollama) with those passages and answer.

See HOW_IT_WORKS.md in the project folder for the same story without code jargon.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from . import config
from .embeddings import encode_texts
from .image_text import extract_text_from_image
from .prompts import SYSTEM_RAG, build_user_prompt
from .vector_store import _connect, init_schema, search

logger = logging.getLogger(__name__)


def _find_similar_chunks(search_text: str) -> list[tuple[int, str, str, float]]:
    """Step 2 — vector search over ingested knowledge (runs in a worker thread from async code)."""
    conn = _connect(config.SQLITE_PATH)
    try:
        init_schema(conn)
        q = search_text.strip() or "help"
        q_emb = encode_texts(config.EMBEDDING_MODEL, [q])[0]
        return search(conn, q_emb, config.TOP_K)
    finally:
        conn.close()


async def _call_ollama_chat(system: str, user: str) -> str:
    """Step 3 — ask the local LLM to produce the final reply."""
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()


async def answer_user_message(
    *,
    typed_text: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """
    Main entry: typed_text is from /ask or the photo caption; image_bytes is optional.
    """
    typed = (typed_text or "").strip()

    # Step 1 — text from image via local HF model (or Tesseract), in a worker thread
    from_image = ""
    if image_bytes:
        from_image = await asyncio.to_thread(extract_text_from_image, image_bytes)

    if not typed and not from_image:
        return (
            "I need either some typed text (use /ask or a caption) or readable text in the image. "
            "Try a clearer screenshot or add a short caption with your question."
        )

    search_query = "\n".join(x for x in (typed, from_image) if x).strip()

    try:
        rows = await asyncio.to_thread(_find_similar_chunks, search_query)
    except Exception as e:
        logger.exception("Retrieval failed")
        return f"Could not search the knowledge base: {e}"

    context = "\n\n---\n\n".join(f"[{doc}]\n{chunk}" for _, doc, chunk, _ in rows)

    user_prompt = build_user_prompt(
        typed_message=typed,
        text_from_image=from_image,
        context=context,
    )

    try:
        return await _call_ollama_chat(SYSTEM_RAG, user_prompt)
    except httpx.ConnectError as e:
        logger.warning(
            "Ollama unreachable at %s (%s)", config.OLLAMA_BASE_URL, e
        )
        return (
            "Cannot connect to Ollama (nothing is listening at that address).\n\n"
            f"Configured URL: {config.OLLAMA_BASE_URL}\n"
            f"Model: {config.OLLAMA_CHAT_MODEL}\n\n"
            "Important: 127.0.0.1 is always THIS machine — the one running the Python bot.\n"
            "If you installed Ollama on your Mac but run the bot on a remote server (SSH, DSW, "
            "Docker elsewhere), the server cannot see your Mac’s Ollama. Either run the bot on "
            "the same Mac as Ollama, or install Ollama on the server and point OLLAMA_BASE_URL there.\n\n"
            "Same machine as Ollama (e.g. Mac):\n"
            "  1) Open the Ollama app (menu bar) so the API is up — see https://ollama.com/download/mac\n"
            f"  2) Terminal: ollama pull {config.OLLAMA_CHAT_MODEL}\n"
            f"  3) On the machine that runs THIS bot, test: curl -sS {config.OLLAMA_BASE_URL}/api/version\n\n"
            "Remote bot + Ollama on another PC: set OLLAMA_BASE_URL in .env to a reachable host "
            "(SSH tunnel, LAN IP, or run Ollama on the server)."
        )
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out at %s", config.OLLAMA_BASE_URL)
        return (
            "Ollama did not respond in time. Try again, use a smaller/faster model, "
            "or increase load time on the Ollama host."
        )
    except httpx.HTTPStatusError as e:
        return (
            f"The chat model returned an error ({e.response.status_code}). "
            "Is Ollama running, and did you run `ollama pull` for your model? "
            f"Details: {e.response.text[:400]}"
        )
    except Exception as e:
        logger.exception("LLM chat failed")
        return f"Could not get an answer from the chat model: {e}"
