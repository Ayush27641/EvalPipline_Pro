"""EvalRunner orchestration with fake pipeline/judge/DB: concurrency bound, error capture."""

import asyncio
from contextlib import asynccontextmanager
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.evals.metrics.answer_relevancy import RelevancyVerdict, StatementVerdict
from app.evals.metrics.contextual_precision import ChunkVerdict, ContextualPrecisionVerdict
from app.evals.metrics.faithfulness import ClaimVerdict, FaithfulnessVerdict
from app.evals.runner import EvalRunner, NoGoldenCasesError
from app.schemas.evals import EvalRunRequest
from tests.conftest import FakeJudge, make_case, make_chunk, make_response

pytestmark = pytest.mark.unit


class FakePipeline:
    prompt_version = "v1"

    def __init__(self, fail_for: List[str] = (), delay: float = 0.01) -> None:
        self.fail_for = set(fail_for)
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    def with_prompt_version(self, version: str) -> "FakePipeline":
        return self

    async def run(self, session, query: str, top_k=None):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if query in self.fail_for:
                raise RuntimeError("provider exploded")
            return make_response([make_chunk(1), make_chunk(2, rank=2)])
        finally:
            self.in_flight -= 1


def _perfect_judge() -> FakeJudge:
    return FakeJudge(
        {
            FaithfulnessVerdict: FaithfulnessVerdict(
                claims=[ClaimVerdict(claim="c", verdict="supported")], summary="ok"
            ),
            RelevancyVerdict: RelevancyVerdict(
                statements=[StatementVerdict(statement="s", verdict="relevant")], summary="ok"
            ),
            ContextualPrecisionVerdict: ContextualPrecisionVerdict(
                verdicts=[ChunkVerdict(chunk_index=0, relevant=True), ChunkVerdict(chunk_index=1, relevant=False)],
                summary="ok",
            ),
        }
    )


@asynccontextmanager
async def _fake_session():
    yield object()


def _session_factory():
    return _fake_session()


def _cases(n: int):
    cases = []
    for i in range(1, n + 1):
        case = make_case(f"GD-{i:03d}")
        case.query = f"query {i}"
        cases.append(case)
    return cases


async def _run(runner: EvalRunner, request: EvalRunRequest, cases):
    with patch("app.evals.runner.load_golden_cases", AsyncMock(return_value=cases)), patch(
        "app.evals.runner.persist_run", AsyncMock(return_value=42)
    ) as persist:
        report = await runner.run(_session_factory, request)
    return report, persist


async def test_runner_passes_and_persists():
    pipeline = FakePipeline()
    runner = EvalRunner(pipeline, _perfect_judge(), Settings(llm_api_key="x"))
    report, persist = await _run(runner, EvalRunRequest(commit_hash="abc"), _cases(4))
    assert report.overall_status == "PASS"
    assert report.run_id == 42 and persist.await_count == 1
    assert report.total_cases == report.passed_cases == 4
    assert report.avg_faithfulness == 1.0 and report.avg_contextual_precision == 1.0
    assert report.commit_hash == "abc" and report.judge_model == "fake-judge"


async def test_runner_respects_concurrency_limit():
    pipeline = FakePipeline(delay=0.02)
    runner = EvalRunner(pipeline, _perfect_judge(), Settings(llm_api_key="x"))
    await _run(runner, EvalRunRequest(max_concurrency=2), _cases(8))
    assert 1 <= pipeline.max_in_flight <= 2


async def test_runner_captures_pipeline_failure_and_fails_closed():
    pipeline = FakePipeline(fail_for=["query 2"])
    runner = EvalRunner(pipeline, _perfect_judge(), Settings(llm_api_key="x"))
    report, _ = await _run(runner, EvalRunRequest(), _cases(3))
    assert report.overall_status == "FAIL"
    assert report.errored_cases == 1 and report.passed_cases == 2
    bad = next(c for c in report.cases if c.case_key == "GD-002")
    assert bad.error.startswith("rag_pipeline: RuntimeError")
    assert all(m.error for m in bad.metrics.values())


async def test_runner_raises_without_golden_cases():
    runner = EvalRunner(FakePipeline(), _perfect_judge(), Settings(llm_api_key="x"))
    with pytest.raises(NoGoldenCasesError):
        await _run(runner, EvalRunRequest(), [])
