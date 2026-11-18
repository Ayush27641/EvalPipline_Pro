"""/api/v1/rag — the target RAG service under evaluation."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_rag_pipeline, get_retriever, get_session
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import JobRetriever
from app.schemas.rag import JobListingSummary, RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Semantic search over job listings + grounded structured answer",
    responses={502: {"description": "LLM provider error / schema never satisfied"},
               503: {"description": "LLM_API_KEY not configured"}},
)
async def query_rag(
    payload: RAGQueryRequest,
    session: AsyncSession = Depends(get_session),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGQueryResponse:
    return await pipeline.run(session, payload.query, top_k=payload.top_k)


@router.get("/jobs", response_model=List[JobListingSummary], summary="Browse indexed listings")
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    location: Optional[str] = Query(None, description="case-insensitive substring filter"),
    session: AsyncSession = Depends(get_session),
    retriever: JobRetriever = Depends(get_retriever),
) -> List[JobListingSummary]:
    return await retriever.list_jobs(session, limit=limit, offset=offset, location=location)


@router.get("/jobs/count", summary="Number of indexed listings")
async def count_jobs(
    session: AsyncSession = Depends(get_session),
    retriever: JobRetriever = Depends(get_retriever),
) -> dict:
    return {"count": await retriever.count(session)}
