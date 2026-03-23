"""
RAG orchestration — data flow:

  Telegram (caption + optional photo bytes)
      → vision: extract_text_from_image (if bytes)
      → retrieval.build_search_query → find_similar_chunks → chunks_to_context
      → prompts.build_user_prompt + SYSTEM_RAG
      → ollama_chat.complete_chat
      → reply_format.polish_reply
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from . import config
from .image_text import extract_text_from_image
from .ollama_chat import (
    close_client,
    complete_chat,
    ollama_connect_error_message,
)
from .prompts import SYSTEM_RAG, build_user_prompt
from .reply_format import polish_reply
from .retrieval import build_search_query, chunks_to_context, find_similar_chunks
from .user_messages import NEED_QUESTION, PHOTO_CAPTION_REQUIRED

logger = logging.getLogger(__name__)


async def answer_user_message(
    *,
    typed_text: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """
    Entry: `typed_text` = /ask or photo caption; `image_bytes` set when user sends a photo.

    Flow: validate → optional vision → retrieve → prompt → Ollama → format.
    """
    typed = (typed_text or "").strip()

    if image_bytes and not typed:
        return PHOTO_CAPTION_REQUIRED

    from_image = ""
    if image_bytes:
        from_image = (await asyncio.to_thread(extract_text_from_image, image_bytes)).strip()

    if not typed and not from_image:
        return NEED_QUESTION

    search_query = build_search_query(
        typed_message=typed, text_from_image=from_image
    ).strip()
    if not search_query:
        return (
            "Could not build a search query. If you sent a photo, try a clearer screenshot "
            "or a more specific caption."
        )

    try:
        rows = await asyncio.to_thread(find_similar_chunks, search_query)
    except Exception as e:
        logger.exception("Retrieval failed")
        return f"Could not search the knowledge base: {e}"

    context = chunks_to_context(rows)
    user_prompt = build_user_prompt(
        typed_message=typed,
        text_from_image=from_image,
        context=context,
        image_attached=bool(image_bytes),
    )

    try:
        raw = await complete_chat(system=SYSTEM_RAG, user=user_prompt)
        return polish_reply(raw)
    except httpx.ConnectError as e:
        return ollama_connect_error_message(e)
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out at %s", config.OLLAMA_BASE_URL)
        return (
            "Ollama did not respond in time. Try again, use a smaller/faster model, "
            "or increase OLLAMA_HTTP_TIMEOUT in .env."
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


async def close_ollama_http_client() -> None:
    """Hook for Application.post_shutdown."""
    await close_client()
