
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import faiss

from config import settings
from processing.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        index_path: str = None,
        metadata_path: str = None,
        dimension: int = None,
    ):
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = Path(metadata_path or settings.metadata_db_path)
        self.dimension = dimension or settings.embedding_dimension

        self._index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine after normalize)
        self._metadata: list[dict] = []   # Parallel array to FAISS vectors
        self._doc_ids: set[str] = set()   # Track unique document IDs

        self._load()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[np.ndarray]):
        """Add chunks and their embeddings to the store."""
        assert len(chunks) == len(embeddings), "Chunks and embeddings must have same length"

        valid_pairs = [
            (c, e) for c, e in zip(chunks, embeddings) if e is not None
        ]
        if not valid_pairs:
            return

        vecs = np.array([e for _, e in valid_pairs], dtype=np.float32)
        faiss.normalize_L2(vecs)

        self._ensure_index()
        self._index.add(vecs)

        for chunk, _ in valid_pairs:
            self._metadata.append(chunk.to_dict())
            self._doc_ids.add(chunk.doc_id)

        self._save()
        logger.info(f"Added {len(valid_pairs)} chunks. Total: {self._index.ntotal}")

    def remove_document(self, doc_id: str):
        if doc_id not in self._doc_ids:
            return

        remaining = [m for m in self._metadata if m["doc_id"] != doc_id]
        logger.info(f"Removing doc {doc_id}: {len(self._metadata) - len(remaining)} chunks removed")

        # Rebuild index
        self._metadata = remaining
        self._doc_ids = {m["doc_id"] for m in remaining}
        self._index = self._new_index()
        self._save()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = None,
        filter_doc_ids: Optional[list[str]] = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = top_k or settings.top_k_chunks
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)

        # Fetch more candidates if filtering
        fetch_k = min(top_k * 10, self._index.ntotal) if (filter_doc_ids or filter_file_names) else top_k
        scores, indices = self._index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = dict(self._metadata[idx])
            meta["score"] = float(score)

            # Apply metadata filters
            if filter_doc_ids and meta["doc_id"] not in filter_doc_ids:
                continue
            if filter_file_names and meta["file_name"] not in filter_file_names:
                continue

            results.append(meta)
            if len(results) >= top_k:
                break

        return results

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for meta in self._metadata:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": meta["file_name"],
                    "source": meta["source"],
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())

    def document_exists(self, doc_id: str) -> bool:
        return doc_id in self._doc_ids

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def total_documents(self) -> int:
        return len(self._doc_ids)

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _load(self):
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path) as f:
                    self._metadata = json.load(f)
                self._doc_ids = {m["doc_id"] for m in self._metadata}
                logger.info(
                    f"Loaded FAISS index: {self._index.ntotal} vectors, "
                    f"{len(self._doc_ids)} documents"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Starting fresh.")

        self._index = self._new_index()
        self._metadata = []
        self._doc_ids = set()

    def _ensure_index(self):
        if self._index is None:
            self._index = self._new_index()

    def _new_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(self.dimension)