"""CLI regression gate: run the full evaluation in-process and exit non-zero on FAIL.

    python -m scripts.run_eval_gate [--threshold 0.85] [--commit abc123] [--prompt-version v1]
                                    [--category retrieval ...] [--report reports/eval_report.json]

Exit codes: 0 = PASS, 1 = FAIL (quality below threshold or errored cases), 2 = could not run.
Writes reports/eval_report.json + reports/eval_summary.md and, when GITHUB_STEP_SUMMARY is set,
appends the markdown summary to the GitHub Actions job summary.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

from app.core.bootstrap import build_app_stack
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.evals.report import render_markdown_summary
from app.evals.repository import count_golden_cases
from app.evals.runner import NoGoldenCasesError
from app.schemas.evals import EvalRunReport, EvalRunRequest

logger = logging.getLogger("scripts.eval_gate")


def write_reports(report: EvalRunReport, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    summary = render_markdown_summary(report)
    report_path.with_name("eval_summary.md").write_text(summary, encoding="utf-8")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(summary)
    print(summary)


async def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="EvalPipeline Pro — CI regression gate")
    parser.add_argument("--threshold", type=float, default=None, help="override EVAL_QUALITY_THRESHOLD")
    parser.add_argument("--commit", default=None, help="commit hash to record (default: env/git)")
    parser.add_argument("--prompt-version", default=None, help="override PROMPT_VERSION for this run")
    parser.add_argument("--category", action="append", default=None, help="restrict to a category")
    parser.add_argument("--seed-if-empty", action="store_true", help="seed data when tables are empty")
    parser.add_argument("--report", default="reports/eval_report.json", help="JSON report path")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.llm_configured:
        logger.error("LLM_API_KEY is not set; cannot run the eval gate")  # TODO(apikey)
        return 2

    stack = await build_app_stack(settings)
    try:
        if args.seed_if_empty:
            async with stack.db.session_factory() as session:
                if await count_golden_cases(session) == 0:
                    from scripts.seed_data import seed  # local import: optional dependency on seed data

                    await seed(stack.db, stack.embedder)
        request = EvalRunRequest(
            commit_hash=args.commit,
            prompt_version=args.prompt_version,
            quality_threshold=args.threshold,
            categories=args.category,
        )
        report = await stack.runner.run(stack.db.session_factory, request)
    except NoGoldenCasesError as exc:
        logger.error("%s", exc)
        return 2
    finally:
        await stack.close()

    write_reports(report, Path(args.report))
    return 0 if report.overall_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
