import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:

    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = model_name or settings.embedding_model
        self.cache_dir = Path(cache_dir or settings.cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.dimension = settings.embedding_dimension

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        return self._model
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string (with caching)."""
        cache_key = self._cache_key(text)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        vec = self.model.encode(text, normalize_embeddings=True)
        self._save_cache(cache_key, vec)
        return vec

    def embed_batch(
        self, texts: list[str], batch_size: int = None
    ) -> list[np.ndarray]:
        batch_size = batch_size or settings.embedding_batch_size
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []  # (original_idx, text)

        for i, text in enumerate(texts):
            cached = self._load_cache(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, text))

        if to_compute:
            logger.info(f"Computing embeddings for {len(to_compute)} texts (cached: {len(texts) - len(to_compute)})")
            indices, raw_texts = zip(*to_compute)

            all_vecs: list[np.ndarray] = []
            for start in range(0, len(raw_texts), batch_size):
                batch = raw_texts[start : start + batch_size]
                vecs = self.model.encode(
                    list(batch),
                    normalize_embeddings=True,
                    show_progress_bar=len(raw_texts) > 50,
                )
                all_vecs.extend(vecs)

            for idx, vec, text in zip(indices, all_vecs, raw_texts):
                results[idx] = vec
                self._save_cache(self._cache_key(text), vec)

        return results 

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(f"{self.model_name}::{text}".encode()).hexdigest()
        return h

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if path.exists():
            try:
                return np.load(str(path))
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def _save_cache(self, key: str, vec: np.ndarray):
        try:
            np.save(str(self._cache_path(key)), vec)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")