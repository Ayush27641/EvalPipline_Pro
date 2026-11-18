"""Pure functions: aggregate per-case results into a run verdict, render summaries.

Kept free of I/O so the PASS/FAIL logic is unit-testable without a database or LLM.

Gate semantics (fail-closed):
  FAIL if there are no cases, if any case errored (pipeline or judge failure), or if any
  metric's average over applicable cases is below the threshold. Otherwise PASS.
"""

from statistics import mean
from typing import Any, Dict, List, Optional

from app.schemas.evals import EvalCaseReport, EvalRunReport

_METRIC_TO_AVG = {
    "faithfulness": "avg_faithfulness",
    "answer_relevancy": "avg_relevancy",
    "contextual_precision": "avg_contextual_precision",
}


def _average(values: List[float]) -> Optional[float]:
    return round(mean(values), 4) if values else None


def aggregate_run(cases: List[EvalCaseReport], threshold: float) -> Dict[str, Any]:
    """Return the summary fields of an EvalRunSummary computed from case reports."""
    averages: Dict[str, Optional[float]] = {}
    for metric, field in _METRIC_TO_AVG.items():
        scores = [
            c.metrics[metric].score
            for c in cases
            if metric in c.metrics
            and c.metrics[metric].error is None
            and c.metrics[metric].score is not None
        ]
        averages[field] = _average([s for s in scores if s is not None])

    latencies = [c.latency_ms for c in cases if c.latency_ms is not None]
    errored = sum(1 for c in cases if c.error)
    passed = sum(1 for c in cases if c.passed)

    below_threshold = [f for f, v in averages.items() if v is not None and v < threshold]
    status = "PASS" if cases and not errored and not below_threshold else "FAIL"

    return {
        "overall_status": status,
        "avg_latency_ms": _average(latencies),
        "total_cases": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "errored_cases": errored,
        **averages,
    }


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown_summary(report: EvalRunReport) -> str:
    """Markdown block for GITHUB_STEP_SUMMARY / CLI output."""
    icon = "✅" if report.overall_status == "PASS" else "❌"
    lines = [
        f"## {icon} Eval gate: **{report.overall_status}**",
        "",
        f"- commit `{report.commit_hash}` · prompt `{report.prompt_version}` · "
        f"generator `{report.generator_model}` · judge `{report.judge_model}`",
        f"- threshold **{report.quality_threshold:.2f}** · cases {report.total_cases} "
        f"(passed {report.passed_cases}, failed {report.failed_cases}, errored {report.errored_cases})"
        f" · {report.duration_seconds:.1f}s",
        "",
        "| Metric | Average | Gate |",
        "|---|---|---|",
    ]
    for label, value in (
        ("Faithfulness", report.avg_faithfulness),
        ("Answer relevancy", report.avg_relevancy),
        ("Contextual precision", report.avg_contextual_precision),
    ):
        gate = "n/a" if value is None else ("✅" if value >= report.quality_threshold else "❌")
        lines.append(f"| {label} | {_fmt(value)} | {gate} |")
    lines.append(f"| Avg latency (ms) | {_fmt(report.avg_latency_ms)} | – |")

    failed = [c for c in report.cases if not c.passed]
    if failed:
        lines += ["", "### Failed / errored cases", "", "| Case | Category | F | R | CP | Why |", "|---|---|---|---|---|---|"]
        for c in failed:
            why = c.error or "; ".join(
                f"{m.name}={m.score:.2f}: {m.reasoning[:120]}"
                for m in c.metrics.values()
                if m.passed is False and m.score is not None
            )
            lines.append(
                f"| {c.case_key} | {c.category} | {_fmt(c.faithfulness)} | {_fmt(c.relevancy)} | "
                f"{_fmt(c.contextual_precision)} | {why.replace('|', '/')[:300]} |"
            )
    return "\n".join(lines) + "\n"
