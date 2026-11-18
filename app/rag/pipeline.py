"""End-to-end RAG pipeline: embed query -> pgvector top-k -> structured generation."""

import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EmbeddingProvider
from app.rag.generator import AnswerGenerator
from app.rag.retriever import JobRetriever
from app.schemas.rag import RAGQueryResponse

logger = logging.getLogger("app.rag.pipeline")


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        retriever: JobRetriever,
        generator: AnswerGenerator,
        *,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @property
    def prompt_version(self) -> str:
        return self.generator.prompt_version

    def with_prompt_version(self, prompt_version: str) -> "RAGPipeline":
        """Cheap variant sharing the embedder/retriever; used for per-run prompt overrides."""
        if prompt_version == self.prompt_version:
            return self
        return RAGPipeline(
            self.embedder,
            self.retriever,
            self.generator.with_prompt_version(prompt_version),
            top_k=self.top_k,
        )

    async def run(
        self, session: AsyncSession, query: str, top_k: Optional[int] = None
    ) -> RAGQueryResponse:
        started = time.perf_counter()

        query_vector = await self.embedder.embed_query(query)
        chunks = await self.retriever.search(session, query_vector, top_k or self.top_k)
        retrieved_at = time.perf_counter()

        answer = await self.generator.generate(query, chunks)
        finished = time.perf_counter()

        response = RAGQueryResponse(
            query=query,
            answer=answer,
            retrieved_chunks=chunks,
            generator_model=self.generator.model,
            prompt_version=self.prompt_version,
            retrieval_latency_ms=round((retrieved_at - started) * 1000, 2),
            generation_latency_ms=round((finished - retrieved_at) * 1000, 2),
            latency_ms=round((finished - started) * 1000, 2),
        )
        logger.info(
            "rag query ok chunks=%d recs=%d answerable=%s latency_ms=%.0f",
            len(chunks),
            len(answer.recommendations),
            answer.is_answerable,
            response.latency_ms,
        )
        return response
