"""Local embedding model wrapper (Sentence Transformers)."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Sequence

import numpy as np

from . import config

_lock = threading.Lock()
_models: dict[str, object] = {}


def get_model(model_name: str):
    with _lock:
        if model_name not in _models:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if config.EMBEDDING_DEVICE:
                kwargs["device"] = config.EMBEDDING_DEVICE
            _models[model_name] = SentenceTransformer(model_name, **kwargs)
        return _models[model_name]


def encode_texts(model_name: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
    model = get_model(model_name)
    try:
        import torch

        ctx = torch.inference_mode()
    except ImportError:
        ctx = nullcontext()
    with ctx:
        emb = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=min(len(texts), config.EMBEDDING_BATCH_SIZE),
        )
    return np.asarray(emb, dtype=np.float32)
