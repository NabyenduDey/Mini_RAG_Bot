"""Local embedding model wrapper (Sentence Transformers)."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

_lock = threading.Lock()
_model = None


def get_model(model_name: str):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(model_name)
        return _model


def encode_texts(model_name: str, texts: Sequence[str]) -> np.ndarray:
    model = get_model(model_name)
    emb = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)
