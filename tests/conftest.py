"""Shared pytest fixtures and fakes (no network, no database for unit tests)."""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import pytest

# Unit tests must not depend on a developer's .env; set safe defaults before the app imports settings.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("EMBEDDING_DIM", "384")

from app.core.config import get_settings  # noqa: E402
from app.schemas.evals import GoldenCaseView  # noqa: E402
from app.schemas.rag import (  # noqa: E402
    JobRecommendation,
    RAGAnswer,
    RAGQueryResponse,
    RetrievedChunk,
)


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- data builders
def make_chunk(job_id: int, title: str = "Backend Engineer", company: str = "Acme", rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        job_id=job_id,
        external_ref=f"JOB-{job_id:03d}",
        title=title,
        company=company,
        location="Bengaluru, India",
        rank=rank,
        similarity=round(0.9 - 0.05 * (rank - 1), 4),
        content=f"{title} at {company}\nLocation: Bengaluru, India\nSalary: ₹30-40 LPA\nSkills: Python",
    )


def make_answer(job_ids: Optional[List[int]] = None, answerable: bool = True) -> RAGAnswer:
    recs = [
        JobRecommendation(
            job_id=j, title="Backend Engineer", company="Acme",
            match_reason="Python backend role in Bengaluru matching the query.",
            key_skills_matched=["Python"],
        )
        for j in (job_ids or [])
    ]
    return RAGAnswer(
        answer="Acme is hiring a Backend Engineer in Bengaluru." if answerable else "No listing matches.",
        recommendations=recs,
        confidence="high" if answerable else "low",
        is_answerable=answerable,
        caveats=[],
    )


def make_response(chunks: Optional[List[RetrievedChunk]] = None, answer: Optional[RAGAnswer] = None) -> RAGQueryResponse:
    chunks = chunks if chunks is not None else [make_chunk(1), make_chunk(2, rank=2)]
    return RAGQueryResponse(
        query="python backend jobs in bengaluru",
        answer=answer or make_answer([c.job_id for c in chunks[:1]]),
        retrieved_chunks=chunks,
        generator_model="fake-generator",
        prompt_version="v1",
        retrieval_latency_ms=12.0,
        generation_latency_ms=88.0,
        latency_ms=100.0,
    )


def make_case(key: str = "GD-001", snippets: Optional[List[str]] = None, category: str = "retrieval") -> GoldenCaseView:
    return GoldenCaseView(
        id=int(key.split("-")[1]),
        case_key=key,
        category=category,
        query="python backend jobs in bengaluru",
        expected_answer="Acme Backend Engineer in Bengaluru.",
        expected_context_snippets=["Acme"] if snippets is None else snippets,
        tags=[],
    )


# --------------------------------------------------------------------------- fakes
class FakeJudge:
    """Returns pre-built verdict objects keyed by response_model class; records prompts."""

    model = "fake-judge"

    def __init__(self, verdicts: Dict[Type, Any]) -> None:
        self._verdicts = verdicts
        self.calls: List[Dict[str, Any]] = []

    async def ask(self, response_model: Type, *, system: str, user: str, validation_context=None):
        self.calls.append({"model": response_model, "system": system, "user": user, "ctx": validation_context})
        verdict = self._verdicts[response_model]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(timezone.utc)
