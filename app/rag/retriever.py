"""pgvector semantic search over job_listings (cosine distance, HNSW-indexed)."""

from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobListing
from app.schemas.rag import JobListingSummary, RetrievedChunk


class JobRetriever:
    async def search(
        self, session: AsyncSession, query_embedding: List[float], top_k: int
    ) -> List[RetrievedChunk]:
        # `<=>` is pgvector's cosine distance operator; similarity = 1 - distance.
        distance = JobListing.embedding.cosine_distance(query_embedding).label("distance")
        stmt = select(JobListing, distance).order_by(distance).limit(top_k)
        rows = (await session.execute(stmt)).all()

        chunks: List[RetrievedChunk] = []
        for rank, (job, dist) in enumerate(rows, start=1):
            chunks.append(
                RetrievedChunk(
                    job_id=job.id,
                    external_ref=job.external_ref,
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    rank=rank,
                    similarity=round(max(-1.0, min(1.0, 1.0 - float(dist))), 4),
                    content=job.to_document(),
                )
            )
        return chunks

    async def count(self, session: AsyncSession) -> int:
        return int(await session.scalar(select(func.count()).select_from(JobListing)) or 0)

    async def list_jobs(
        self, session: AsyncSession, limit: int = 50, offset: int = 0, location: Optional[str] = None
    ) -> List[JobListingSummary]:
        stmt = select(JobListing).order_by(JobListing.id).limit(limit).offset(offset)
        if location:
            stmt = stmt.where(JobListing.location.ilike(f"%{location}%"))
        jobs = (await session.scalars(stmt)).all()
        return [JobListingSummary.model_validate(j) for j in jobs]
