"""sqlite-vec backed vector index + chunk metadata."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import sqlite_vec

from .config import EMBEDDING_DIM


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        );
        """
    )
    dim = EMBEDDING_DIM
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}]);"
    )
    conn.commit()


def clear_all(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM vec_chunks;")
    conn.execute("DELETE FROM chunks;")
    conn.commit()


def knowledge_fingerprint(knowledge_dir: Path) -> str:
    paths = sorted(knowledge_dir.glob("*.md")) + sorted(knowledge_dir.glob("*.txt"))
    h = hashlib.sha256()
    for p in paths:
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()


def get_stored_fingerprint(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'kb_fingerprint'").fetchone()
    return row[0] if row else None


def set_fingerprint(conn: sqlite3.Connection, fp: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('kb_fingerprint', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (fp,),
    )
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    doc_path: str,
    chunk_texts: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    assert len(chunk_texts) == len(embeddings)
    from sqlite_vec import serialize_float32

    with conn:
        for i, (t, row_emb) in enumerate(zip(chunk_texts, embeddings)):
            cur = conn.execute(
                "INSERT INTO chunks(doc_path, chunk_index, text) VALUES (?, ?, ?)",
                (doc_path, i, t),
            )
            chunk_id = cur.lastrowid
            blob = serialize_float32(row_emb.astype(np.float32))
            conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )


def search(
    conn: sqlite3.Connection, query_embedding: np.ndarray, k: int
) -> list[tuple[int, str, str, float]]:
    from sqlite_vec import serialize_float32

    q = serialize_float32(query_embedding.astype(np.float32))
    # vec0 KNN with JOIN: LIMIT on the outer query is not accepted by the planner;
    # run KNN on vec_chunks alone with k = ?, then join chunk text (sqlite-vec #96 / #116).
    rows = conn.execute(
        """
        WITH vec_matches AS (
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
              AND k = ?
        )
        SELECT c.id, c.doc_path, c.text, m.distance
        FROM vec_matches m
        JOIN chunks c ON c.id = m.rowid
        ORDER BY m.distance
        """,
        (q, k),
    ).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2]), float(r[3])) for r in rows]


def chunk_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0]) if row else 0
