"""
TTL + LRU cache for single-query embedding vectors (retrieval hot path).

Stdlib only. Disable with EMBEDDING_QUERY_CACHE_TTL_SEC=0 in .env.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

import numpy as np

from . import config
from .embeddings import encode_texts

_lock = threading.Lock()
_store: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()


def embed_query_vector(model_name: str, search_text: str) -> np.ndarray:
    """Return a normalized query embedding; may reuse a recent identical request."""
    text = search_text.strip()
    ttl = config.EMBEDDING_QUERY_CACHE_TTL_SEC
    cap = config.EMBEDDING_QUERY_CACHE_MAX
    if ttl <= 0 or cap <= 0:
        return encode_texts(model_name, [text])[0]

    key = hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()
    now = time.monotonic()

    with _lock:
        hit = _store.get(key)
        if hit is not None:
            vec, ts = hit
            if now - ts < ttl:
                _store.move_to_end(key)
                return vec.copy()
            del _store[key]

        vec = encode_texts(model_name, [text])[0]
        _store[key] = (vec.copy(), now)
        _store.move_to_end(key)
        while len(_store) > cap:
            _store.popitem(last=False)
        return vec.copy()
