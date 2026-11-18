"""Seed job_listings (30 roles) and golden_datasets (15 cases), embedding listings with the
configured embedding provider.

Usage (inside the api container, or a local venv with .env configured):
    python -m scripts.seed_data            # upsert by external_ref / case_key (idempotent)
    python -m scripts.seed_data --reset    # wipe listings, golden cases AND eval history first

Data lives in scripts/seed_jobs.py and scripts/seed_golden.py.
"""

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict, List, Tuple

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import EvalCaseResult, EvalRun, GoldenDataset, JobListing
from app.db.session import Database
from app.rag.embeddings import EmbeddingProvider, build_embedding_provider
from scripts.seed_golden import GOLDEN_CASES
from scripts.seed_jobs import JOB_LISTINGS

logger = logging.getLogger("scripts.seed")


def _document_for(job: Dict[str, Any]) -> str:
    return JobListing.render_document(
        title=job["title"],
        company=job["company"],
        location=job["location"],
        employment_type=job["employment_type"],
        experience_level=job["experience_level"],
        salary_range=job["salary_range"],
        skills=job["skills"],
        description=job["description"],
    )


async def seed(db: Database, embedder: EmbeddingProvider, *, reset: bool = False) -> Tuple[int, int]:
    """Upsert listings + golden cases. Returns (jobs_written, golden_cases_written)."""
    documents = [_document_for(job) for job in JOB_LISTINGS]
    logger.info("embedding %d job documents with %s", len(documents), embedder.name)
    vectors = await embedder.embed(documents)

    async with db.session_factory() as session:
        if reset:
            logger.warning("--reset: deleting eval history, golden cases and job listings")
            for model in (EvalCaseResult, EvalRun, GoldenDataset, JobListing):
                await session.execute(delete(model))

        existing_jobs = {j.external_ref: j for j in (await session.scalars(select(JobListing))).all()}
        for job, vector in zip(JOB_LISTINGS, vectors):
            row = existing_jobs.get(job["external_ref"])
            if row is None:
                session.add(JobListing(**job, embedding=vector))
            else:
                for field, value in job.items():
                    setattr(row, field, value)
                row.embedding = vector

        existing_cases = {
            c.case_key: c for c in (await session.scalars(select(GoldenDataset))).all()
        }
        for case in GOLDEN_CASES:
            row = existing_cases.get(case["case_key"])
            if row is None:
                session.add(GoldenDataset(**case))
            else:
                for field, value in case.items():
                    setattr(row, field, value)

        await session.commit()

    logger.info("seeded %d job listings and %d golden cases", len(JOB_LISTINGS), len(GOLDEN_CASES))
    return len(JOB_LISTINGS), len(GOLDEN_CASES)


def _validate_dataset() -> List[str]:
    """Cheap consistency checks so a typo in the data fails loudly before touching the DB."""
    problems: List[str] = []
    refs = [j["external_ref"] for j in JOB_LISTINGS]
    if len(refs) != len(set(refs)):
        problems.append("duplicate external_ref in JOB_LISTINGS")
    keys = [c["case_key"] for c in GOLDEN_CASES]
    if len(keys) != len(set(keys)):
        problems.append("duplicate case_key in GOLDEN_CASES")
    corpus = "\n".join(_document_for(j) for j in JOB_LISTINGS)
    for case in GOLDEN_CASES:
        for snippet in case["expected_context_snippets"]:
            if snippet not in corpus:
                problems.append(f"{case['case_key']}: snippet {snippet!r} not found in any listing")
    return problems


async def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Seed EvalPipeline Pro data")
    parser.add_argument("--reset", action="store_true", help="delete existing rows first")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    problems = _validate_dataset()
    if problems:
        for p in problems:
            logger.error(p)
        return 2

    db = Database(settings)
    try:
        await db.init_schema()
        embedder = build_embedding_provider(settings)
        await embedder.load()
        await seed(db, embedder, reset=args.reset)
    finally:
        await db.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
