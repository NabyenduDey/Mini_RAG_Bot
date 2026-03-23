"""
mini_rag_telegram_bot.rag — local RAG over Markdown knowledge + Telegram.

Flow (answers): ``app`` → ``flow.answer_user_message`` → optional ``image_text``,
``retrieval`` (cached query embeddings + sqlite-vec), ``ollama_chat``, ``reply_format``.

Flow (index): ``ingest`` → ``embeddings.encode_texts`` (batch) → ``vector_store``.
"""
