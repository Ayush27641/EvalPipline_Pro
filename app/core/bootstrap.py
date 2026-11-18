"""Composition root: builds every long-lived component once from Settings.

Used by the FastAPI lifespan, the seed script, the CI gate script and the regression tests,
so all entrypoints wire the stack identically.
"""

import logging
from dataclasses import dataclass

import instructor

from app.core.config import Settings
from app.db.session import Database
from app.evals.judge import LLMJudge
from app.evals.runner import EvalRunner
from app.rag.embeddings import EmbeddingProvider, build_embedding_provider
from app.rag.generator import AnswerGenerator
from app.rag.llm_client import build_instructor_client
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import JobRetriever

logger = logging.getLogger("app.bootstrap")


@dataclass
class AppStack:
    settings: Settings
    db: Database
    embedder: EmbeddingProvider
    llm_client: instructor.AsyncInstructor
    pipeline: RAGPipeline
    judge: LLMJudge
    runner: EvalRunner

    async def close(self) -> None:
        await self.db.dispose()


async def build_app_stack(settings: Settings, *, init_schema: bool = True) -> AppStack:
    db = Database(settings)
    if init_schema:
        await db.init_schema()

    embedder = build_embedding_provider(settings)
    await embedder.load()  # downloads/loads the model once, off the event loop

    llm_client = build_instructor_client(settings)
    generator = AnswerGenerator(
        llm_client,
        model=settings.llm_model,
        prompt_version=settings.prompt_version,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
        enabled=settings.llm_configured,
    )
    pipeline = RAGPipeline(embedder, JobRetriever(), generator, top_k=settings.rag_top_k)
    judge = LLMJudge(
        llm_client,
        model=settings.judge_model,
        max_retries=settings.llm_max_retries,
        enabled=settings.llm_configured,
    )
    runner = EvalRunner(pipeline, judge, settings)

    logger.info(
        "stack ready: embedder=%s generator=%s judge=%s top_k=%d prompt=%s",
        embedder.name,
        settings.llm_model,
        settings.judge_model,
        settings.rag_top_k,
        settings.prompt_version,
    )
    return AppStack(
        settings=settings,
        db=db,
        embedder=embedder,
        llm_client=llm_client,
        pipeline=pipeline,
        judge=judge,
        runner=runner,
    )
