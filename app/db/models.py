"""SQLAlchemy 2.0 ORM models for EvalPipeline Pro (async, PostgreSQL + pgvector).

Tables
------
job_listings       indexed corpus the RAG pipeline retrieves from (vector(EMBEDDING_DIM))
golden_datasets    curated queries + ground-truth answers + expected evidence snippets
eval_runs          one row per evaluation run (commit, prompt version, averages, PASS/FAIL)
eval_case_results  one row per (run, golden case): scores, retrieved chunks, output, reasoning

Python 3.9 compatible (typing.List/Dict/Optional).
"""

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings

# The vector column width is fixed at import time from EMBEDDING_DIM. Changing the
# embedding model means dropping job_listings and re-seeding (see README).
EMBEDDING_DIM: int = get_settings().embedding_dim


class Base(DeclarativeBase):
    pass


class EvalStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"


# --------------------------------------------------------------------------- job_listings
class JobListing(Base):
    __tablename__ = "job_listings"
    __table_args__ = (
        # HNSW index for approximate nearest-neighbour search with cosine distance (<=>).
        Index(
            "ix_job_listings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_ref: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)  # JOB-001
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    company: Mapped[str] = mapped_column(String(120), nullable=False)
    location: Mapped[str] = mapped_column(String(120), nullable=False)
    employment_type: Mapped[str] = mapped_column(String(40), nullable=False)  # Full-time, Internship
    experience_level: Mapped[str] = mapped_column(String(60), nullable=False)
    salary_range: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    skills: Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    @staticmethod
    def render_document(
        *,
        title: str,
        company: str,
        location: str,
        employment_type: str,
        experience_level: str,
        salary_range: Optional[str],
        skills: List[str],
        description: str,
    ) -> str:
        """Canonical text for a listing.

        Used for BOTH embedding at seed time and as the chunk content shown to the
        generator and the judges, so faithfulness is checked against exactly what
        the LLM saw. The numeric job_id is deliberately excluded (it is noise for
        the embedding); prompts prepend it at render time.
        """
        return (
            f"{title} at {company}\n"
            f"Location: {location} | Type: {employment_type} | Level: {experience_level}\n"
            f"Salary: {salary_range or 'Not disclosed'}\n"
            f"Skills: {', '.join(skills)}\n\n"
            f"{description}"
        )

    def to_document(self) -> str:
        return self.render_document(
            title=self.title,
            company=self.company,
            location=self.location,
            employment_type=self.employment_type,
            experience_level=self.experience_level,
            salary_range=self.salary_range,
            skills=list(self.skills or []),
            description=self.description,
        )


# --------------------------------------------------------------------------- golden_datasets
class GoldenDataset(Base):
    __tablename__ = "golden_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_key: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)  # GD-001
    # retrieval | hallucination | out_of_scope | edge_case | adversarial
    category: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    # Verbatim snippets that MUST appear in the retrieved context for a correct answer.
    # Empty list == the query is unanswerable from the corpus (contextual precision N/A).
    expected_context_snippets: Mapped[List[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    tags: Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --------------------------------------------------------------------------- eval_runs
class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commit_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    generator_model: Mapped[str] = mapped_column(String(120), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(120), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    quality_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    overall_status: Mapped[EvalStatus] = mapped_column(
        Enum(EvalStatus, name="eval_status", native_enum=False, length=8), nullable=False
    )
    # Averages are nullable: a metric can be not-applicable for every case in a run.
    avg_faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_relevancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_contextual_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    passed_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    errored_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    case_results: Mapped[List["EvalCaseResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


# --------------------------------------------------------------------------- eval_case_results
class EvalCaseResult(Base):
    __tablename__ = "eval_case_results"
    __table_args__ = (Index("ix_eval_case_results_run_passed", "run_id", "passed"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    golden_case_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("golden_datasets.id", ondelete="SET NULL"), nullable=True
    )
    # Snapshot of the golden case so history stays readable if the dataset is edited later.
    case_key: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)

    retrieved_chunks: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    actual_output: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    actual_answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relevancy_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contextual_precision_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # {metric_name: {"score", "passed", "applicable", "reasoning", "details", "error"}}
    judge_reasoning: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped[EvalRun] = relationship(back_populates="case_results")
