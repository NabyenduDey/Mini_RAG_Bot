"""Load knowledge files, chunk, embed, and write sqlite-vec index."""

from __future__ import annotations

from pathlib import Path

from . import config
from .chunking import chunk_markdown_file
from .embeddings import encode_texts
from .vector_store import (
    chunk_count,
    clear_all,
    get_stored_fingerprint,
    init_schema,
    insert_chunks,
    knowledge_fingerprint,
    set_fingerprint,
    _connect,
)


def list_knowledge_files(knowledge_dir: Path) -> list[Path]:
    if not knowledge_dir.is_dir():
        return []
    md = sorted(knowledge_dir.glob("*.md"))
    txt = sorted(knowledge_dir.glob("*.txt"))
    return md + txt


def ingest_if_needed() -> tuple[bool, str]:
    """
    Returns (did_reingest, status_message).
    """
    kb = config.KNOWLEDGE_DIR
    files = list_knowledge_files(kb)
    if not files:
        return False, f"No .md or .txt files found under {kb}"

    fp = knowledge_fingerprint(kb)
    conn = _connect(config.SQLITE_PATH)
    try:
        init_schema(conn)
        prev = get_stored_fingerprint(conn)
        n = chunk_count(conn)
        need = config.AUTO_REINGEST and (prev != fp or n == 0)
        if not need:
            return False, f"Index up to date ({n} chunks, fingerprint ok)."

        clear_all(conn)
        all_chunks: list[tuple[str, str]] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            parts = chunk_markdown_file(
                text, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP
            )
            rel = path.name
            for p in parts:
                all_chunks.append((rel, p))

        if not all_chunks:
            set_fingerprint(conn, fp)
            return True, "Knowledge files produced zero chunks."

        texts = [t for _, t in all_chunks]
        embs = encode_texts(config.EMBEDDING_MODEL, texts)

        # Insert per document group to keep chunk_index meaningful
        by_doc: dict[str, list[str]] = {}
        for doc, t in all_chunks:
            by_doc.setdefault(doc, []).append(t)

        offset = 0
        for doc, doc_chunks in by_doc.items():
            n_doc = len(doc_chunks)
            slice_emb = embs[offset : offset + n_doc]
            offset += n_doc
            insert_chunks(conn, doc, doc_chunks, slice_emb)

        set_fingerprint(conn, fp)
        return True, f"Ingested {len(all_chunks)} chunks from {len(files)} file(s)."
    finally:
        conn.close()
