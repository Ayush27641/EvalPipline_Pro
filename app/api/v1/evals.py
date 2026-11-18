"""/api/v1/evals — run the evaluation suite and read historical results."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_eval_runner, get_session
from app.db.session import Database
from app.evals.repository import get_run, list_runs, load_golden_cases
from app.evals.runner import EvalRunner, NoGoldenCasesError
from app.schemas.evals import EvalRunReport, EvalRunRequest, EvalRunSummary, GoldenCaseView

router = APIRouter(prefix="/evals", tags=["evals"])


@router.post(
    "/run",
    response_model=EvalRunReport,
    summary="Evaluate every golden case concurrently and return a PASS/FAIL report",
    description=(
        "Synchronous: the response arrives when all cases are scored (15 cases ≈ 30-90 s on Groq). "
        "Each case = 1 generation + 3 judge calls, bounded by EVAL_MAX_CONCURRENCY."
    ),
    responses={404: {"description": "No golden cases matched"},
               503: {"description": "LLM_API_KEY not configured"}},
)
async def run_evaluation(
    payload: EvalRunRequest = EvalRunRequest(),
    db: Database = Depends(get_db),
    runner: EvalRunner = Depends(get_eval_runner),
) -> EvalRunReport:
    try:
        return await runner.run(db.session_factory, payload)
    except NoGoldenCasesError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/history", response_model=List[EvalRunSummary], summary="Past runs, newest first")
async def evaluation_history(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> List[EvalRunSummary]:
    return await list_runs(session, limit=limit, offset=offset)


@router.get("/runs/{run_id}", response_model=EvalRunReport, summary="Full report incl. per-case judge reasoning")
async def evaluation_run(run_id: int, session: AsyncSession = Depends(get_session)) -> EvalRunReport:
    report = await get_run(session, run_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"run {run_id} not found")
    return report


@router.get("/golden", response_model=List[GoldenCaseView], summary="Active golden test cases")
async def golden_cases(session: AsyncSession = Depends(get_session)) -> List[GoldenCaseView]:
    return await load_golden_cases(session)
