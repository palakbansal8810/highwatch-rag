import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import settings
from processing.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    source: str
    chunk_index: int
    chunk_text: str
    char_start: int
    char_end: int
    section_heading: Optional[str] = None
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "chunk_text": self.chunk_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_heading": self.section_heading,
            "token_count": self.token_count,
        }

HEADING_RE = re.compile(
    r"^(#{1,4}\s.+|[A-Z][A-Z\s]{4,}:|(?:\d+\.)+\d*\s+[A-Z].{4,}|[A-Z].{4,}:)\s*$",
    re.MULTILINE,
)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_length: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_length = min_chunk_length or settings.min_chunk_length

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.is_valid:
            return []

        sections = self._split_into_sections(doc.raw_text)
        chunks: list[Chunk] = []
        offset = 0

        for heading, section_text in sections:
            section_chunks = self._chunk_text(
                text=section_text,
                doc_id=doc.file_id,
                file_name=doc.file_name,
                source=doc.source,
                section_heading=heading,
                base_offset=offset,
                start_index=len(chunks),
            )
            chunks.extend(section_chunks)
            offset += len(section_text)

        logger.info(f"Chunked '{doc.file_name}' into {len(chunks)} chunks")
        return chunks

    def _split_into_sections(self, text: str) -> list[tuple[Optional[str], str]]:
        matches = list(HEADING_RE.finditer(text))

        if not matches:
            return [(None, text)]

        sections: list[tuple[Optional[str], str]] = []
        prev_end = 0
        prev_heading = None

        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append((None, preamble))

        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                sections.append((heading, body))

        return sections if sections else [(None, text)]


    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        file_name: str,
        source: str,
        section_heading: Optional[str],
        base_offset: int,
        start_index: int,
    ) -> list[Chunk]:
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_words: list[str] = []
        current_start = 0
        chunk_idx = start_index

        def flush(words: list[str], start: int) -> Optional[Chunk]:
            chunk_text = " ".join(words).strip()
            if len(chunk_text) < self.min_chunk_length:
                return None
            end = start + len(chunk_text)
            return Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                file_name=file_name,
                source=source,
                chunk_index=len(chunks) + start_index,
                chunk_text=chunk_text,
                char_start=base_offset + start,
                char_end=base_offset + end,
                section_heading=section_heading,
                token_count=len(words),
            )

        pos = 0
        for sentence in sentences:
            words = sentence.split()
            if not words:
                pos += len(sentence) + 1
                continue

            if len(current_words) + len(words) > self.chunk_size and current_words:
                chunk = flush(current_words, current_start)
                if chunk:
                    chunks.append(chunk)
                # Keep overlap
                overlap_words = current_words[-self.chunk_overlap:] if self.chunk_overlap else []
                current_start = pos - sum(len(w) + 1 for w in overlap_words)
                current_words = overlap_words

            current_words.extend(words)
            pos += len(sentence) + 1

        if current_words:
            chunk = flush(current_words, current_start)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        sentences: list[str] = []

        for para in paragraphs:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para.strip())
            sentences.extend(p.strip() for p in parts if p.strip())

        return sentences