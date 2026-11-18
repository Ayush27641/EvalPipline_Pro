"""PASS/FAIL aggregation, markdown rendering and commit-hash resolution (pure functions)."""

from datetime import datetime, timezone

import pytest

from app.evals.report import aggregate_run, render_markdown_summary
from app.evals.runner import resolve_commit_hash
from app.schemas.evals import EvalCaseReport, EvalRunReport, MetricResult
from tests.conftest import make_answer, make_case, make_chunk

pytestmark = pytest.mark.unit


def _case_report(key: str, f: float, r: float, cp, latency: float = 100.0) -> EvalCaseReport:
    results = [
        MetricResult(name="faithfulness", score=f, passed=f >= 0.85),
        MetricResult(name="answer_relevancy", score=r, passed=r >= 0.85),
        MetricResult.not_applicable("contextual_precision", "n/a")
        if cp is None
        else MetricResult(name="contextual_precision", score=cp, passed=cp >= 0.85),
    ]
    return EvalCaseReport.from_results(
        make_case(key, snippets=[] if cp is None else ["Acme"]),
        actual_output=make_answer([1]),
        retrieved_chunks=[make_chunk(1)],
        latency_ms=latency,
        results=results,
    )


def test_aggregate_pass_when_all_averages_meet_threshold():
    cases = [_case_report("GD-001", 1.0, 0.9, 1.0), _case_report("GD-002", 0.9, 0.9, None, latency=300)]
    summary = aggregate_run(cases, 0.85)
    assert summary["overall_status"] == "PASS"
    assert summary["avg_faithfulness"] == pytest.approx(0.95)
    assert summary["avg_contextual_precision"] == 1.0  # N/A case excluded from the mean
    assert summary["avg_latency_ms"] == 200.0
    assert (summary["total_cases"], summary["passed_cases"], summary["failed_cases"]) == (2, 2, 0)


def test_aggregate_fails_when_one_metric_average_drops():
    cases = [_case_report("GD-001", 1.0, 1.0, 1.0), _case_report("GD-002", 0.5, 1.0, 1.0)]
    summary = aggregate_run(cases, 0.85)
    assert summary["overall_status"] == "FAIL"
    assert summary["avg_faithfulness"] == 0.75
    assert summary["failed_cases"] == 1


def test_aggregate_fails_closed_on_errored_case_even_with_high_scores():
    good = _case_report("GD-001", 1.0, 1.0, 1.0)
    errored = EvalCaseReport.from_error(make_case("GD-002"), "rag_pipeline: timeout")
    summary = aggregate_run([good, errored], 0.85)
    assert summary["overall_status"] == "FAIL"
    assert summary["errored_cases"] == 1
    assert summary["avg_faithfulness"] == 1.0  # errored case does not poison the average


def test_aggregate_fails_with_no_cases():
    assert aggregate_run([], 0.85)["overall_status"] == "FAIL"


def test_case_passes_only_when_all_applicable_metrics_pass():
    assert _case_report("GD-001", 0.9, 0.9, None).passed is True
    assert _case_report("GD-002", 0.9, 0.5, None).passed is False
    assert _case_report("GD-003", 0.9, 0.9, 0.3).passed is False


def test_markdown_summary_lists_failed_cases():
    cases = [_case_report("GD-001", 1.0, 1.0, 1.0), _case_report("GD-002", 0.4, 1.0, 1.0)]
    report = EvalRunReport(
        commit_hash="abc123", prompt_version="v1", generator_model="g", judge_model="j",
        embedding_model="e", created_at=datetime.now(timezone.utc), quality_threshold=0.85,
        cases=cases, **aggregate_run(cases, 0.85),
    )
    md = render_markdown_summary(report)
    assert "**FAIL**" in md and "GD-002" in md and "GD-001 |" not in md
    assert "| Faithfulness | 0.700 | ❌ |" in md


def test_resolve_commit_hash_precedence(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "feedbeef")
    monkeypatch.setenv("GIT_COMMIT_SHA", "cafebabe")
    assert resolve_commit_hash("explicit") == "explicit"
    assert resolve_commit_hash(None) == "cafebabe"  # GIT_COMMIT_SHA beats GITHUB_SHA
    monkeypatch.delenv("GIT_COMMIT_SHA")
    assert resolve_commit_hash("") == "feedbeef"
