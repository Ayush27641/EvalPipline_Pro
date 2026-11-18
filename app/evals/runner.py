"""Async evaluation runner.

For every active golden case (optionally filtered) it:
  1. runs the RAG pipeline (own DB session per case — AsyncSession is not concurrency-safe),
  2. scores the response with the three judge metrics concurrently,
  3. builds an EvalCaseReport (errors are captured, never raised, so one bad case cannot
     abort the run),
then aggregates PASS/FAIL and persists the run + case results.

Rate limiting: ONE asyncio.Semaphore bounds every LLM-bound call in the run (generation and
each judge call), so EVAL_MAX_CONCURRENCY == max in-flight provider requests.
"""

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Awaitable, List, Optional, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.evals.judge import Judge
from app.evals.metrics import BaseMetric, build_default_metrics
from app.evals.report import aggregate_run
from app.evals.repository import load_golden_cases, persist_run
from app.rag.pipeline import RAGPipeline
from app.schemas.evals import EvalCaseReport, EvalRunReport, EvalRunRequest, GoldenCaseView, MetricResult

logger = logging.getLogger("app.evals.runner")

T = TypeVar("T")


class NoGoldenCasesError(RuntimeError):
    """No golden cases matched the request (empty table or over-restrictive filters)."""


def resolve_commit_hash(explicit: Optional[str] = None) -> str:
    """Explicit value > CI env vars > local git > 'unknown'. Never raises."""
    if explicit and explicit.strip():
        return explicit.strip()[:64]
    for var in ("GIT_COMMIT_SHA", "GITHUB_SHA", "COMMIT_HASH"):
        value = os.environ.get(var, "").strip()
        if value:
            return value[:64]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


class EvalRunner:
    def __init__(self, pipeline: RAGPipeline, judge: Judge, settings: Settings) -> None:
        self._pipeline = pipeline
        self._judge = judge
        self._settings = settings

    # ------------------------------------------------------------------ public
    async def run(
        self, session_factory: async_sessionmaker[AsyncSession], request: EvalRunRequest
    ) -> EvalRunReport:
        settings = self._settings
        threshold = (
            request.quality_threshold
            if request.quality_threshold is not None
            else settings.eval_quality_threshold
        )
        concurrency = request.max_concurrency or settings.eval_max_concurrency
        pipeline = self._pipeline.with_prompt_version(
            request.prompt_version or self._pipeline.prompt_version
        )
        metrics = build_default_metrics(self._judge, threshold)

        async with session_factory() as session:
            cases = await load_golden_cases(
                session, case_keys=request.case_keys, categories=request.categories
            )
        if not cases:
            raise NoGoldenCasesError(
                "no active golden cases matched; seed the database (python -m scripts.seed_data)"
            )

        logger.info(
            "eval run start cases=%d concurrency=%d threshold=%.2f prompt=%s",
            len(cases),
            concurrency,
            threshold,
            pipeline.prompt_version,
        )
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(concurrency)
        case_reports: List[EvalCaseReport] = list(
            await asyncio.gather(
                *(
                    self._evaluate_case(semaphore, session_factory, pipeline, metrics, case)
                    for case in cases
                )
            )
        )
        duration = time.perf_counter() - started

        summary = aggregate_run(case_reports, threshold)
        report = EvalRunReport(
            commit_hash=resolve_commit_hash(request.commit_hash),
            prompt_version=pipeline.prompt_version,
            generator_model=settings.llm_model,
            judge_model=self._judge.model,
            embedding_model=settings.embedding_model,
            quality_threshold=threshold,
            duration_seconds=round(duration, 2),
            created_at=datetime.now(timezone.utc),
            cases=case_reports,
            **summary,
        )

        async with session_factory() as session:
            report.run_id = await persist_run(session, report)

        logger.info(
            "eval run %s id=%s status=%s F=%s R=%s CP=%s passed=%d/%d errored=%d in %.1fs",
            report.commit_hash,
            report.run_id,
            report.overall_status,
            report.avg_faithfulness,
            report.avg_relevancy,
            report.avg_contextual_precision,
            report.passed_cases,
            report.total_cases,
            report.errored_cases,
            duration,
        )
        return report

    # ------------------------------------------------------------------ internals
    @staticmethod
    async def _limited(semaphore: asyncio.Semaphore, awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    async def _evaluate_case(
        self,
        semaphore: asyncio.Semaphore,
        session_factory: async_sessionmaker[AsyncSession],
        pipeline: RAGPipeline,
        metrics: List[BaseMetric],
        case: GoldenCaseView,
    ) -> EvalCaseReport:
        try:
            async with semaphore:
                async with session_factory() as session:
                    response = await pipeline.run(session, case.query)
        except Exception as exc:  # noqa: BLE001 - one failing case must not abort the run
            logger.exception("case %s: RAG pipeline failed", case.case_key)
            return EvalCaseReport.from_error(
                case, f"rag_pipeline: {type(exc).__name__}: {exc}"[:2000]
            )

        results = await asyncio.gather(
            *(self._run_metric(semaphore, metric, case, response) for metric in metrics)
        )
        report = EvalCaseReport.from_results(
            case,
            actual_output=response.answer,
            retrieved_chunks=response.retrieved_chunks,
            latency_ms=response.latency_ms,
            results=list(results),
        )
        logger.info(
            "case %s %s F=%s R=%s CP=%s latency_ms=%.0f",
            case.case_key,
            "PASS" if report.passed else "FAIL",
            report.faithfulness,
            report.relevancy,
            report.contextual_precision,
            report.latency_ms or 0,
        )
        return report

    async def _run_metric(
        self,
        semaphore: asyncio.Semaphore,
        metric: BaseMetric,
        case: GoldenCaseView,
        response,
    ) -> MetricResult:
        try:
            return await self._limited(semaphore, metric.evaluate(case, response))
        except Exception as exc:  # noqa: BLE001 - captured into the report (fail-closed)
            logger.exception("case %s: metric %s failed", case.case_key, metric.name)
            return MetricResult.failed(metric.name, f"{type(exc).__name__}: {exc}"[:2000])
