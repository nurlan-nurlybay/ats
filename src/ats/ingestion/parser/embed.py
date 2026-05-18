"""bge-m3 singleton.

Lazy-loaded SentenceTransformer instance shared between the parser's
document embedder and KeyBERT's keyphrase extractor — avoids loading the
~568M-parameter model twice.
"""
from __future__ import annotations

import os
from threading import Lock

from sentence_transformers import SentenceTransformer

from ats.core.config import settings
from ats.core.logger import get_logger

log = get_logger(__name__)

_model: SentenceTransformer | None = None
_lock = Lock()


def get_embedder() -> SentenceTransformer:
    """Return the singleton bge-m3 model. Loads on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        cfg = settings.models.semantic
        # ATS_DEVICE env-var wins over YAML so Docker can force CPU without
        # editing configs/config.yaml (which the host still wants on CUDA).
        device_setting = os.environ.get("ATS_DEVICE") or cfg.device
        device = None if device_setting == "auto" else device_setting
        log.info("bge_m3_load", name=cfg.name, device=device_setting)
        m = SentenceTransformer(cfg.name, device=device)
        m.max_seq_length = cfg.max_seq_length
        _model = m
    return _model


def embed_text(text: str) -> list[float]:
    cfg = settings.models.semantic
    vec = get_embedder().encode(
        text,
        normalize_embeddings=cfg.normalize_embeddings,
        show_progress_bar=False,
    )
    return vec.tolist()


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    cfg = settings.models.semantic
    vecs = get_embedder().encode(
        texts,
        batch_size=batch_size or cfg.batch_size,
        normalize_embeddings=cfg.normalize_embeddings,
        show_progress_bar=False,
    )
    return vecs.tolist()
