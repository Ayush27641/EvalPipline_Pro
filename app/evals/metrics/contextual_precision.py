"""Contextual Precision: are the relevant chunks ranked above the irrelevant ones?

For each retrieved chunk (in rank order) the judge decides whether it is needed to produce
the EXPECTED answer (ground truth). Score is the weighted cumulative precision used by
deepeval/RAGAS:

    score = (1 / R) * sum_{k : chunk_k relevant} (relevant_in_top_k / k)

where R is the number of relevant chunks. Perfect ranking -> 1.0; relevant chunks buried
below irrelevant ones -> lower; no relevant chunk retrieved -> 0.0.

Not applicable when the golden case has no expected context snippets (unanswerable /
out-of-scope queries): there is nothing the retriever *should* have found.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from app.evals.metrics.base import BaseMetric, render_chunks
from app.schemas.evals import GoldenCaseView, MetricResult
from app.schemas.rag import RAGQueryResponse


class ChunkVerdict(BaseModel):
    chunk_index: int = Field(..., ge=0)
    relevant: bool
    reason: str = ""


class ContextualPrecisionVerdict(BaseModel):
    verdicts: List[ChunkVerdict]
    summary: str

    @model_validator(mode="after")
    def _one_verdict_per_chunk(self, info: ValidationInfo) -> "ContextualPrecisionVerdict":
        expected = (info.context or {}).get("expected_chunk_count")
        if expected is not None:
            indices = sorted(v.chunk_index for v in self.verdicts)
            if indices != list(range(expected)):
                raise ValueError(
                    f"Return exactly one verdict for each chunk_index 0..{expected - 1} "
                    f"(got {indices})."
                )
        return self


def weighted_cumulative_precision(relevance: List[bool]) -> float:
    relevant_so_far = 0
    total = 0.0
    for k, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            relevant_so_far += 1
            total += relevant_so_far / k
    return total / relevant_so_far if relevant_so_far else 0.0


_SYSTEM = """You evaluate a retrieval system for a job-matching assistant.
You get the candidate's QUERY, the EXPECTED ANSWER written by a human reviewer, optional EXPECTED
EVIDENCE snippets (text that must appear in a useful chunk), and the retrieved CONTEXT CHUNKS in rank
order. For EACH chunk decide whether it is relevant: it contains information needed to produce the
expected answer, or it matches one of the expected evidence snippets. A chunk about a different company
or role than the ones the expected answer relies on is irrelevant, even if it is superficially similar.
Return exactly one verdict per chunk, using the chunk_index values given."""

_USER = """QUERY:
{query}

EXPECTED ANSWER (ground truth):
{expected}

EXPECTED EVIDENCE SNIPPETS:
{snippets}

CONTEXT CHUNKS (rank order, best first):
{chunks}"""


class ContextualPrecisionMetric(BaseMetric):
    name = "contextual_precision"

    async def evaluate(self, case: GoldenCaseView, response: RAGQueryResponse) -> MetricResult:
        if not case.expects_retrieval:
            return MetricResult.not_applicable(
                self.name, "query is unanswerable from the corpus; no expected context"
            )
        chunks = response.retrieved_chunks
        if not chunks:
            return self._result(0.0, "nothing was retrieved although relevant listings exist")

        validation_context: Dict[str, Any] = {"expected_chunk_count": len(chunks)}
        verdict = await self.judge.ask(
            ContextualPrecisionVerdict,
            system=_SYSTEM,
            user=_USER.format(
                query=case.query,
                expected=case.expected_answer,
                snippets="\n".join(f"- {s}" for s in case.expected_context_snippets),
                chunks=render_chunks(chunks),
            ),
            context=validation_context,
        )
        ordered = sorted(verdict.verdicts, key=lambda v: v.chunk_index)
        relevance = [v.relevant for v in ordered]
        score = weighted_cumulative_precision(relevance)
        return self._result(
            score,
            verdict.summary,
            relevance_by_rank=relevance,
            relevant_count=sum(relevance),
            verdicts=[v.model_dump() for v in ordered],
        )
