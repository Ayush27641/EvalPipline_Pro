"""Pydantic v2 schemas for the RAG service.

`RAGAnswer` is the schema the LLM is FORCED to produce (via `instructor`). Its
validator uses the validation context supplied by the generator to reject
hallucinated job references; a rejection becomes an automatic re-ask with the
error message, which is how parse/grounding failures are retried.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


# --------------------------------------------------------------------------- requests
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000, examples=["Senior Python roles in Bengaluru"])
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Defaults to RAG_TOP_K")


# --------------------------------------------------------------------------- retrieval
class RetrievedChunk(BaseModel):
    job_id: int
    external_ref: str
    title: str
    company: str
    location: str
    rank: int = Field(..., ge=1)
    similarity: float = Field(..., ge=-1.0, le=1.0, description="cosine similarity (1 - distance)")
    content: str = Field(..., description="Exact text shown to the generator and judges")


class JobListingSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_ref: str
    title: str
    company: str
    location: str
    employment_type: str
    experience_level: str
    salary_range: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- LLM output (strict)
class JobRecommendation(BaseModel):
    job_id: int = Field(..., description="MUST be one of the job_id values in the retrieved listings")
    title: str
    company: str
    match_reason: str = Field(
        ..., min_length=10, description="Why this listing fits, citing facts from the listing only"
    )
    key_skills_matched: List[str] = Field(default_factory=list, max_length=10)


class RAGAnswer(BaseModel):
    """Structured career-matching answer. Every field is grounded in retrieved listings."""

    answer: str = Field(..., min_length=1, description="Conversational answer for the candidate")
    recommendations: List[JobRecommendation] = Field(default_factory=list, max_length=5)
    confidence: Literal["high", "medium", "low"]
    is_answerable: bool = Field(
        ..., description="false when the listings do not contain what the candidate asked for"
    )
    caveats: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Facts the candidate asked for that the listings do NOT state",
    )

    @model_validator(mode="after")
    def _enforce_grounding(self, info: ValidationInfo) -> "RAGAnswer":
        ctx: Dict = info.context or {}
        allowed = ctx.get("allowed_job_ids")
        jobs: Dict[int, Dict[str, str]] = ctx.get("jobs") or {}

        if allowed is not None:
            unknown = sorted({r.job_id for r in self.recommendations} - set(allowed))
            if unknown:
                raise ValueError(
                    f"recommendations reference job_id(s) {unknown} that are NOT in the retrieved "
                    f"listings {sorted(allowed)}. Only recommend retrieved listings; if nothing fits, "
                    "return an empty recommendations list and set is_answerable=false."
                )
        for rec in self.recommendations:
            expected = jobs.get(rec.job_id)
            if expected and rec.company.strip().lower() != expected["company"].strip().lower():
                raise ValueError(
                    f"job_id {rec.job_id} belongs to company {expected['company']!r}, "
                    f"not {rec.company!r}. Copy the company name exactly from the listing."
                )
        if not self.is_answerable and self.recommendations:
            raise ValueError(
                "is_answerable is false but recommendations were given. Either set "
                "is_answerable=true or return an empty recommendations list."
            )
        return self

    def as_text(self) -> str:
        """Flat rendering used by the judges and stored as actual_answer_text."""
        parts: List[str] = [self.answer.strip()]
        for rec in self.recommendations:
            skills = f" Skills matched: {', '.join(rec.key_skills_matched)}." if rec.key_skills_matched else ""
            parts.append(
                f"- Recommended: {rec.title} at {rec.company} (job_id={rec.job_id}). "
                f"{rec.match_reason.strip()}{skills}"
            )
        if self.caveats:
            parts.append("Caveats: " + " ".join(c.strip() for c in self.caveats))
        return "\n".join(parts)


# --------------------------------------------------------------------------- response
class RAGQueryResponse(BaseModel):
    query: str
    answer: RAGAnswer
    retrieved_chunks: List[RetrievedChunk]
    generator_model: str
    prompt_version: str
    retrieval_latency_ms: float = Field(..., ge=0)
    generation_latency_ms: float = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0, description="End-to-end pipeline latency")
