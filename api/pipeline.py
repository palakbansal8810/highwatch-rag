import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from connectors.gdrive import GoogleDriveConnector
from processing.parser import DocumentParser
from processing.chunker import TextChunker, Chunk
from embedding.embedder import EmbeddingModel
from search.vector_store import VectorStore
from api.llm import GroqLLM, RAGResponse
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_added: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class RAGPipeline:

    def __init__(self):
        self.drive = GoogleDriveConnector()
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = EmbeddingModel()
        self.store = VectorStore()
        self.llm = GroqLLM()
        self._last_sync_time: Optional[str] = None

    async def sync_drive(
        self,
        folder_id: Optional[str] = None,
        incremental: bool = False,
        force_reindex: bool = False,
    ) -> SyncResult:
        start = datetime.now()
        result = SyncResult()

        if not self.drive.is_authenticated():
            authenticated = self.drive.authenticate()
            if not authenticated:
                raise RuntimeError(
                    "Google Drive not authenticated. Call /auth/login first."
                )

        modified_after = self._last_sync_time if incremental else None

        # Run blocking IO in thread pool
        loop = asyncio.get_event_loop()
        drive_files = await loop.run_in_executor(
            None,
            lambda: self.drive.fetch_all_files(
                folder_id=folder_id,
                modified_after=modified_after,
            ),
        )

        result.files_discovered = len(drive_files)
        logger.info(f"Discovered {len(drive_files)} files from Google Drive")

        for drive_file in drive_files:
            try:
                if not force_reindex and self.store.document_exists(drive_file.file_id):
                    logger.debug(f"Skipping already-indexed: {drive_file.name}")
                    result.files_skipped += 1
                    continue

                # Parse
                parsed = self.parser.parse(
                    content=drive_file.content,
                    file_name=drive_file.name,
                    file_id=drive_file.file_id,
                    mime_type=drive_file.mime_type,
                )

                if not parsed.is_valid:
                    logger.warning(f"Failed to parse {drive_file.name}: {parsed.error}")
                    result.files_failed += 1
                    result.errors.append(f"{drive_file.name}: {parsed.error}")
                    continue

                # Chunk
                chunks: list[Chunk] = self.chunker.chunk_document(parsed)
                if not chunks:
                    logger.warning(f"No chunks produced for {drive_file.name}")
                    result.files_skipped += 1
                    continue

                # Embed (batch)
                texts = [c.chunk_text for c in chunks]
                embeddings = await loop.run_in_executor(
                    None,
                    lambda t=texts: self.embedder.embed_batch(t),
                )

                # Store
                self.store.add_chunks(chunks, embeddings)

                result.files_processed += 1
                result.chunks_added += len(chunks)
                logger.info(f"✓ {drive_file.name} → {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {drive_file.name}: {e}")
                result.files_failed += 1
                result.errors.append(f"{drive_file.name}: {str(e)}")

        self._last_sync_time = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = (datetime.now() - start).total_seconds()

        logger.info(
            f"Sync complete in {result.duration_seconds:.1f}s: "
            f"{result.files_processed} processed, {result.files_skipped} skipped, "
            f"{result.files_failed} failed, {result.chunks_added} chunks added"
        )
        return result

    async def ask(
        self,
        query: str,
        top_k: int = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> RAGResponse:
        if self.store.total_chunks == 0:
            return RAGResponse(
                answer="The knowledge base is empty. Please sync your Google Drive first using POST /sync-drive.",
                sources=[],
                chunks_used=0,
                model=settings.groq_model,
                query=query,
            )

        loop = asyncio.get_event_loop()

        # Embed query
        query_vec = await loop.run_in_executor(
            None, lambda: self.embedder.embed(query)
        )

        # Retrieve
        top_k = top_k or settings.top_k_chunks
        chunks = self.store.search(
            query_vec=query_vec,
            top_k=top_k,
            filter_file_names=filter_file_names,
        )

        logger.info(f"Query: '{query[:80]}' → {len(chunks)} chunks retrieved")

        # Generate
        response = await loop.run_in_executor(
            None, lambda: self.llm.generate_answer(query, chunks)
        )
        return response

    def get_status(self) -> dict:
        return {
            "authenticated": self.drive.is_authenticated(),
            "total_documents": self.store.total_documents,
            "total_chunks": self.store.total_chunks,
            "embedding_model": self.embedder.model_name,
            "llm_model": settings.groq_model,
            "last_sync": self._last_sync_time,
        }