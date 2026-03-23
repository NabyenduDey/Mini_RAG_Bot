"""
Public RAG entry points (thin re-exports).

Data flow: `app` → `flow.answer_user_message` → `retrieval` / `image_text` / `ollama_chat`.
Ingest and indexing stay in `ingest` + `vector_store`; embeddings batch path in `embeddings`.
"""

from __future__ import annotations

from .flow import answer_user_message, close_ollama_http_client

__all__ = ["answer_user_message", "close_ollama_http_client"]
