"""
Step 2 of RAG: embed the search string (with optional query-vector cache) and run sqlite-vec KNN + filters.
"""

from __future__ import annotations

import threading

from . import config
from .embedding_cache import embed_query_vector
from .vector_store import _connect, init_schema, search

_reader_local = threading.local()


def reader_connection():
    """One SQLite connection per worker thread (Telegram thread pool)."""
    conn = getattr(_reader_local, "conn", None)
    if conn is None:
        conn = _connect(config.SQLITE_PATH)
        init_schema(conn)
        _reader_local.conn = conn
    return conn


def build_search_query(*, typed_message: str, text_from_image: str) -> str:
    """Merge caption / /ask text with vision transcript for embedding."""
    typed = (typed_message or "").strip()
    from_img = (text_from_image or "").strip()
    if from_img and typed:
        return f"{from_img}\n\nAdditional question from the user (photo caption): {typed}"
    if from_img:
        return from_img
    return typed


def narrow_rows(
    rows: list[tuple[int, str, str, float]],
) -> list[tuple[int, str, str, float]]:
    """Distance band + elbow so unrelated docs (e.g. policy vs recipe) rarely reach the LLM."""
    if not rows:
        return rows
    margin = config.RETRIEVAL_DISTANCE_MARGIN
    if margin <= 0:
        return rows[: config.TOP_K]

    best_d = rows[0][3]
    band = [r for r in rows if r[3] <= best_d + margin]
    if not band:
        band = [rows[0]]

    elbow = config.RETRIEVAL_ELBOW_GAP
    out: list[tuple[int, str, str, float]] = [band[0]]
    for i in range(1, len(band)):
        prev_d, d = band[i - 1][3], band[i][3]
        if d - prev_d > elbow:
            break
        out.append(band[i])
        if len(out) >= config.TOP_K:
            break

    return out[: config.TOP_K]


def find_similar_chunks(search_text: str) -> list[tuple[int, str, str, float]]:
    """Vector search over ingested knowledge (call from a worker thread)."""
    conn = reader_connection()
    q = search_text.strip() or "help"
    q_emb = embed_query_vector(config.EMBEDDING_MODEL, q)
    candidates = search(conn, q_emb, config.TOP_K_CANDIDATES)
    return narrow_rows(candidates)


def chunks_to_context(
    rows: list[tuple[int, str, str, float]],
) -> str:
    """Format retrieved rows for the LLM user prompt."""
    return "\n\n---\n\n".join(f"[{doc}]\n{chunk}" for _, doc, chunk, _ in rows)
