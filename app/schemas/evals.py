"""Pydantic v2 schemas for the evaluation engine: requests, per-metric results,
per-case reports, run summaries (history rows) and full run reports."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.rag import RAGAnswer, RetrievedChunk

MetricName = Literal["faithfulness", "answer_relevancy", "contextual_precision"]
METRIC_NAMES: List[str] = ["faithfulness", "answer_relevancy", "contextual_precision"]


# --------------------------------------------------------------------------- golden dataset view
class GoldenCaseView(BaseModel):
    """Read-only snapshot of a golden_datasets row (safe to pass between sessions/tasks)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    case_key: str
    category: str
    query: str
    expected_answer: str
    expected_context_snippets: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    @property
    def expects_retrieval(self) -> bool:
        """False for unanswerable/out-of-scope cases -> contextual precision is N/A."""
        return len(self.expected_context_snippets) > 0


# --------------------------------------------------------------------------- requests
class EvalRunRequest(BaseModel):
    commit_hash: Optional[str] = Field(
        None, max_length=64, description="Defaults to GIT_COMMIT_SHA/GITHUB_SHA env or `git rev-parse`"
    )
    prompt_version: Optional[str] = Field(None, max_length=32, description="Defaults to PROMPT_VERSION")
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Defaults to EVAL_QUALITY_THRESHOLD (0.85)"
    )
    max_concurrency: Optional[int] = Field(None, ge=1, le=64, description="asyncio semaphore size")
    case_keys: Optional[List[str]] = Field(None, description="Run only these golden case keys")
    categories: Optional[List[str]] = Field(None, description="Run only these golden categories")


# --------------------------------------------------------------------------- metric result
class MetricResult(BaseModel):
    name: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    passed: Optional[bool] = Field(None, description="None when not applicable")
    applicable: bool = True
    reasoning: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def not_applicable(cls, name: str, reason: str) -> "MetricResult":
        return cls(name=name, score=None, passed=None, applicable=False, reasoning=reason)

    @classmethod
    def failed(cls, name: str, error: str) -> "MetricResult":
        return cls(name=name, score=None, passed=False, applicable=True, error=error)


# --------------------------------------------------------------------------- per-case report
class EvalCaseReport(BaseModel):
    case_key: str
    category: str
    golden_case_id: Optional[int] = None
    query: str
    expected_answer: str
    actual_answer: Optional[str] = None
    actual_output: Optional[RAGAnswer] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    faithfulness: Optional[float] = None
    relevancy: Optional[float] = None
    contextual_precision: Optional[float] = None
    latency_ms: Optional[float] = None
    passed: bool = False
    error: Optional[str] = None

    @staticmethod
    def _score(metrics: Dict[str, MetricResult], name: str) -> Optional[float]:
        result = metrics.get(name)
        return result.score if result is not None and result.error is None else None

    @classmethod
    def from_error(cls, case: GoldenCaseView, error: str) -> "EvalCaseReport":
        return cls(
            case_key=case.case_key,
            category=case.category,
            golden_case_id=case.id,
            query=case.query,
            expected_answer=case.expected_answer,
            metrics={name: MetricResult.failed(name, "skipped: RAG pipeline failed") for name in METRIC_NAMES},
            passed=False,
            error=error,
        )

    @classmethod
    def from_results(
        cls,
        case: GoldenCaseView,
        *,
        actual_output: RAGAnswer,
        retrieved_chunks: List[RetrievedChunk],
        latency_ms: float,
        results: List[MetricResult],
    ) -> "EvalCaseReport":
        metrics = {r.name: r for r in results}
        errors = [f"{r.name}: {r.error}" for r in results if r.error]
        passed = not errors and all(r.passed is not False for r in results)
        return cls(
            case_key=case.case_key,
            category=case.category,
            golden_case_id=case.id,
            query=case.query,
            expected_answer=case.expected_answer,
            actual_answer=actual_output.as_text(),
            actual_output=actual_output,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            faithfulness=cls._score(metrics, "faithfulness"),
            relevancy=cls._score(metrics, "answer_relevancy"),
            contextual_precision=cls._score(metrics, "contextual_precision"),
            latency_ms=latency_ms,
            passed=passed,
            error="; ".join(errors) if errors else None,
        )


# --------------------------------------------------------------------------- run summary / report
class EvalRunSummary(BaseModel):
    """One historical run (what `/evals/history` returns and the dashboard plots)."""

    model_config = ConfigDict(from_attributes=True)

    run_id: Optional[int] = None
    commit_hash: str
    prompt_version: str
    generator_model: str
    judge_model: str
    embedding_model: str
    quality_threshold: float
    overall_status: Literal["PASS", "FAIL"]
    avg_faithfulness: Optional[float] = None
    avg_relevancy: Optional[float] = None
    avg_contextual_precision: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_cases: int
    passed_cases: int
    failed_cases: int
    errored_cases: int = 0
    duration_seconds: float = 0.0
    created_at: datetime


class EvalRunReport(EvalRunSummary):
    """Full report: summary + every case with scores, chunks, output and judge reasoning."""

    cases: List[EvalCaseReport] = Field(default_factory=list)

    @property
    def failed_case_reports(self) -> List[EvalCaseReport]:
        return [c for c in self.cases if not c.passed]
