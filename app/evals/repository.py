"""Database access for the evaluation engine: golden cases in, run reports out."""

from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import EvalCaseResult, EvalRun, EvalStatus, GoldenDataset
from app.schemas.evals import (
    EvalCaseReport,
    EvalRunReport,
    EvalRunSummary,
    GoldenCaseView,
    MetricResult,
)
from app.schemas.rag import RAGAnswer, RetrievedChunk


# --------------------------------------------------------------------------- golden dataset
async def load_golden_cases(
    session: AsyncSession,
    *,
    case_keys: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    include_inactive: bool = False,
) -> List[GoldenCaseView]:
    stmt = select(GoldenDataset).order_by(GoldenDataset.id)
    if not include_inactive:
        stmt = stmt.where(GoldenDataset.is_active.is_(True))
    if case_keys:
        stmt = stmt.where(GoldenDataset.case_key.in_(case_keys))
    if categories:
        stmt = stmt.where(GoldenDataset.category.in_(categories))
    rows = (await session.scalars(stmt)).all()
    return [GoldenCaseView.model_validate(r) for r in rows]


async def count_golden_cases(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(GoldenDataset)) or 0)


# --------------------------------------------------------------------------- persistence
def _case_to_row(report: EvalCaseReport) -> EvalCaseResult:
    return EvalCaseResult(
        golden_case_id=report.golden_case_id,
        case_key=report.case_key,
        category=report.category,
        query=report.query,
        expected_answer=report.expected_answer,
        retrieved_chunks=[c.model_dump(mode="json") for c in report.retrieved_chunks],
        actual_output=report.actual_output.model_dump(mode="json") if report.actual_output else None,
        actual_answer_text=report.actual_answer,
        faithfulness_score=report.faithfulness,
        relevancy_score=report.relevancy,
        contextual_precision_score=report.contextual_precision,
        judge_reasoning={name: m.model_dump(mode="json") for name, m in report.metrics.items()},
        latency_ms=report.latency_ms,
        passed=report.passed,
        error=report.error,
    )


async def persist_run(session: AsyncSession, report: EvalRunReport) -> int:
    run = EvalRun(
        commit_hash=report.commit_hash,
        prompt_version=report.prompt_version,
        generator_model=report.generator_model,
        judge_model=report.judge_model,
        embedding_model=report.embedding_model,
        quality_threshold=report.quality_threshold,
        overall_status=EvalStatus(report.overall_status),
        avg_faithfulness=report.avg_faithfulness,
        avg_relevancy=report.avg_relevancy,
        avg_contextual_precision=report.avg_contextual_precision,
        avg_latency_ms=report.avg_latency_ms,
        total_cases=report.total_cases,
        passed_cases=report.passed_cases,
        failed_cases=report.failed_cases,
        errored_cases=report.errored_cases,
        duration_seconds=report.duration_seconds,
        created_at=report.created_at,
        case_results=[_case_to_row(c) for c in report.cases],
    )
    session.add(run)
    await session.commit()
    return run.id


# --------------------------------------------------------------------------- reads
def run_to_summary(run: EvalRun) -> EvalRunSummary:
    return EvalRunSummary(
        run_id=run.id,
        commit_hash=run.commit_hash,
        prompt_version=run.prompt_version,
        generator_model=run.generator_model,
        judge_model=run.judge_model,
        embedding_model=run.embedding_model,
        quality_threshold=run.quality_threshold,
        overall_status=run.overall_status.value,
        avg_faithfulness=run.avg_faithfulness,
        avg_relevancy=run.avg_relevancy,
        avg_contextual_precision=run.avg_contextual_precision,
        avg_latency_ms=run.avg_latency_ms,
        total_cases=run.total_cases,
        passed_cases=run.passed_cases,
        failed_cases=run.failed_cases,
        errored_cases=run.errored_cases,
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
    )


def case_row_to_report(row: EvalCaseResult) -> EvalCaseReport:
    return EvalCaseReport(
        case_key=row.case_key,
        category=row.category,
        golden_case_id=row.golden_case_id,
        query=row.query,
        expected_answer=row.expected_answer,
        actual_answer=row.actual_answer_text,
        actual_output=RAGAnswer.model_validate(row.actual_output) if row.actual_output else None,
        retrieved_chunks=[RetrievedChunk.model_validate(c) for c in row.retrieved_chunks or []],
        metrics={k: MetricResult.model_validate(v) for k, v in (row.judge_reasoning or {}).items()},
        faithfulness=row.faithfulness_score,
        relevancy=row.relevancy_score,
        contextual_precision=row.contextual_precision_score,
        latency_ms=row.latency_ms,
        passed=row.passed,
        error=row.error,
    )


async def list_runs(session: AsyncSession, *, limit: int = 20, offset: int = 0) -> List[EvalRunSummary]:
    stmt = select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc()).limit(limit).offset(offset)
    return [run_to_summary(r) for r in (await session.scalars(stmt)).all()]


async def get_run(session: AsyncSession, run_id: int) -> Optional[EvalRunReport]:
    stmt = select(EvalRun).where(EvalRun.id == run_id).options(selectinload(EvalRun.case_results))
    run = await session.scalar(stmt)
    if run is None:
        return None
    summary = run_to_summary(run)
    cases = sorted((case_row_to_report(c) for c in run.case_results), key=lambda c: c.case_key)
    return EvalRunReport(**summary.model_dump(), cases=cases)
