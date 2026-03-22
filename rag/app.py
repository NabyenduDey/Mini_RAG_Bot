#!/usr/bin/env python3
"""
Telegram bot entry point (inside the rag package).

Run from project root (mini_rag_telegram_bot/):
  python -m rag.app

User flow:
  • Text question → /ask <question>
  • Photo / screenshot → mandatory one-line caption; vision model reads on-screen text
"""

from __future__ import annotations

import io
import logging
import socket

import httpx

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import config
from . import user_messages
from .embeddings import encode_texts
from .ingest import ingest_if_needed
from .pipeline import answer_user_message, close_ollama_http_client

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mini_rag_bot")


def _log_ollama_startup_check() -> None:
    """Warn if Ollama is not reachable from this process (same host as 127.0.0.1)."""
    url = f"{config.OLLAMA_BASE_URL}/api/version"
    try:
        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
        ver = r.json().get("version", "?")
        logger.info("Ollama reachable at %s (version %s)", config.OLLAMA_BASE_URL, ver)
    except Exception as e:
        host = socket.gethostname()
        logger.warning(
            "Ollama not reachable at %s from this host (%s): %s",
            config.OLLAMA_BASE_URL,
            host,
            e,
        )
        logger.warning(
            "The bot process must reach Ollama at the URL in OLLAMA_BASE_URL. "
            "127.0.0.1 is only the machine running Python — not your Mac if the bot runs on a server. "
            "Fix: run the bot on the same machine as Ollama, install Ollama on this host, or set "
            "OLLAMA_BASE_URL to a reachable address (e.g. SSH tunnel or LAN IP)."
        )


def _split_telegram_text(s: str, limit: int = 4000) -> list[str]:
    if len(s) <= limit:
        return [s]
    parts: list[str] = []
    while s:
        parts.append(s[:limit])
        s = s[limit:]
    return parts


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(user_messages.START_INTRO)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(user_messages.HELP_TEXT)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    q = " ".join(context.args).strip() if context.args else ""
    if not q:
        await update.message.reply_text("Usage: /ask your question here")
        return
    await _reply_rag(update, typed_text=q, image_bytes=None)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    caption = (update.message.caption or "").strip()
    if not caption:
        await update.message.reply_text(user_messages.PHOTO_CAPTION_REQUIRED)
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    buffer = io.BytesIO()
    await file.download_to_memory(out=buffer)
    await _reply_rag(
        update, typed_text=caption, image_bytes=buffer.getvalue()
    )


async def _reply_rag(
    update: Update,
    *,
    typed_text: str | None,
    image_bytes: bytes | None,
) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.chat.send_action(ChatAction.TYPING)
    try:
        answer = await answer_user_message(
            typed_text=typed_text,
            image_bytes=image_bytes,
        )
    except Exception as e:
        logger.exception("answer_user_message")
        answer = f"Something went wrong: {e}"
    for chunk in _split_telegram_text(answer or "(no reply)"):
        await msg.reply_text(chunk)


async def _post_init(application: Application) -> None:
    # Polling cannot run while a webhook is set; clears 409-style conflicts with webhook mode.
    await application.bot.delete_webhook(drop_pending_updates=False)
    logger.info("Telegram webhook cleared (safe for long polling).")


async def _post_shutdown(application: Application) -> None:
    await close_ollama_http_client()


def main() -> None:
    changed, status = ingest_if_needed()
    logger.info("Knowledge index: %s — %s", changed, status)
    config.require_telegram_token()
    _log_ollama_startup_check()

    if config.RAG_WARMUP_ON_START:
        logger.info("Warming up embedding model (first query will be faster)…")
        try:
            encode_texts(config.EMBEDDING_MODEL, ["warmup"])
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.warning("Embedding warmup failed (first /ask may be slow): %s", e)

    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))

    logger.info("Bot is running (waiting for Telegram messages)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
