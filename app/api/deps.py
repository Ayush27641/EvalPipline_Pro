"""FastAPI dependencies. Components live on `app.state` (set by the lifespan) so tests can
swap them for fakes with `app.dependency_overrides`."""

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.session import Database
from app.evals.runner import EvalRunner
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import JobRetriever


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Database:
    return request.app.state.db


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


def get_rag_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.rag_pipeline


def get_retriever(request: Request) -> JobRetriever:
    return request.app.state.rag_pipeline.retriever


def get_eval_runner(request: Request) -> EvalRunner:
    return request.app.state.eval_runner
