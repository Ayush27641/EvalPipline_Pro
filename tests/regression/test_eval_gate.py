"""Live LLM regression gate — run by .github/workflows/eval_gate.yml.

Needs a reachable pgvector database and LLM_API_KEY. Runs the whole golden suite ONCE
(session fixture), writes reports/eval_report.json + reports/eval_summary.md, then asserts
each gate condition separately so CI output names the metric that regressed.

    pytest -m regression -v
"""

import os
from pathlib import Path

import pytest

from app.core.bootstrap import build_app_stack
from app.core.config import get_settings
from app.evals.repository import count_golden_cases
from app.schemas.evals import EvalRunReport, EvalRunRequest
from scripts.run_eval_gate import write_reports
from scripts.seed_data import seed

pytestmark = pytest.mark.regression

REPORT_PATH = Path(os.environ.get("EVAL_REPORT_PATH", "reports/eval_report.json"))


@pytest.fixture(scope="session")
async def eval_report() -> EvalRunReport:
    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("LLM_API_KEY not set — regression gate needs a live judge")  # TODO(apikey)

    stack = await build_app_stack(settings)
    try:
        async with stack.db.session_factory() as session:
            if await count_golden_cases(session) == 0:
                await seed(stack.db, stack.embedder)
        report = await stack.runner.run(stack.db.session_factory, EvalRunRequest())
    finally:
        await stack.close()

    write_reports(report, REPORT_PATH)
    return report


def test_no_case_errored(eval_report: EvalRunReport):
    errored = [(c.case_key, c.error) for c in eval_report.cases if c.error]
    assert not errored, f"infrastructure/judge errors (fail-closed): {errored}"


def test_faithfulness_meets_threshold(eval_report: EvalRunReport):
    assert eval_report.avg_faithfulness is not None
    assert eval_report.avg_faithfulness >= eval_report.quality_threshold, _why(eval_report, "faithfulness")


def test_answer_relevancy_meets_threshold(eval_report: EvalRunReport):
    assert eval_report.avg_relevancy is not None
    assert eval_report.avg_relevancy >= eval_report.quality_threshold, _why(eval_report, "answer_relevancy")


def test_contextual_precision_meets_threshold(eval_report: EvalRunReport):
    if eval_report.avg_contextual_precision is None:
        pytest.skip("no golden case expects retrieval")
    assert eval_report.avg_contextual_precision >= eval_report.quality_threshold, _why(
        eval_report, "contextual_precision"
    )


def test_hallucination_traps_are_faithful(eval_report: EvalRunReport):
    """Adversarial/hallucination cases must individually stay faithful, not just on average."""
    traps = [c for c in eval_report.cases if c.category in ("hallucination", "adversarial")]
    weak = [(c.case_key, c.faithfulness) for c in traps if (c.faithfulness or 0.0) < eval_report.quality_threshold]
    assert not weak, f"hallucination traps below threshold: {weak}"


def test_overall_status_is_pass(eval_report: EvalRunReport):
    assert eval_report.overall_status == "PASS", f"see {REPORT_PATH.with_name('eval_summary.md')}"


def _why(report: EvalRunReport, metric: str) -> str:
    worst = sorted(
        ((c.case_key, c.metrics[metric].score, c.metrics[metric].reasoning) for c in report.cases
         if metric in c.metrics and c.metrics[metric].score is not None),
        key=lambda t: t[1],
    )[:3]
    return f"{metric} regressed; worst cases: {worst}"
