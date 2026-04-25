import logging
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise, helpful AI assistant that answers questions strictly based on provided document excerpts.

Rules:
- Answer only using the information in the provided context.
- If the context doesn't contain enough information, say: "The documents don't contain enough information to answer this question."
- Be concise, factual, and specific.
- Reference specific documents when relevant (e.g., "According to policy.pdf...").
- Never fabricate information not present in the context.
- Format your answer clearly with bullet points where appropriate.
"""

USER_PROMPT_TEMPLATE = """Answer the following question using ONLY the document excerpts below.

Question: {query}

Document Excerpts:
{context}

Instructions:
- Answer directly and concisely
- Cite which document each piece of information comes from
- If the answer requires combining info from multiple documents, do so clearly
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    query: str
    context_preview: Optional[str] = None


class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = None,
    ) -> RAGResponse:
        max_tokens = max_tokens or settings.answer_max_tokens

        if not context_chunks:
            return RAGResponse(
                answer="No relevant documents were found to answer your question.",
                sources=[],
                chunks_used=0,
                model=self.model,
                query=query,
            )

        context_parts = []
        sources: list[str] = []
        seen_sources: set[str] = set()

        for i, chunk in enumerate(context_chunks, 1):
            file_name = chunk.get("file_name", "Unknown")
            chunk_text = chunk.get("chunk_text", "")
            score = chunk.get("score", 0.0)
            section = chunk.get("section_heading", "")

            if file_name not in seen_sources:
                sources.append(file_name)
                seen_sources.add(file_name)

            header = f"[Excerpt {i} | Source: {file_name}"
            if section:
                header += f" | Section: {section}"
            header += f" | Relevance: {score:.2f}]"

            context_parts.append(f"{header}\n{chunk_text}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = USER_PROMPT_TEMPLATE.format(query=query, context=context)

        logger.info(f"Generating answer for query: '{query[:80]}...' using {len(context_chunks)} chunks")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1, 
            )

            answer = completion.choices[0].message.content.strip()

            return RAGResponse(
                answer=answer,
                sources=sources,
                chunks_used=len(context_chunks),
                model=self.model,
                query=query,
                context_preview=context[:500] + "..." if len(context) > 500 else context,
            )

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e