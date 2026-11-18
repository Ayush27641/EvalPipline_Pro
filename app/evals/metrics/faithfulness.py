"""Faithfulness (hallucination check).

The judge splits the answer into atomic claims and labels each one against the retrieved
context only. score = (supported + truthful abstentions) / total claims.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from app.evals.metrics.base import BaseMetric, render_chunks
from app.schemas.evals import GoldenCaseView, MetricResult
from app.schemas.rag import RAGQueryResponse


class ClaimVerdict(BaseModel):
    claim: str = Field(..., description="One atomic factual statement extracted from the answer")
    verdict: Literal["supported", "unsupported", "abstention"]
    evidence: str = Field("", description="Quote from the context that supports/contradicts the claim")


class FaithfulnessVerdict(BaseModel):
    claims: List[ClaimVerdict] = Field(..., min_length=1)
    summary: str = Field(..., description="2-3 sentences explaining the overall judgement")


_SYSTEM = """You are a meticulous fact-checker for a job-matching assistant.
You receive the CONTEXT (job listings the assistant was given) and the assistant's ANSWER.

Step 1: split the ANSWER into atomic factual claims (one fact per claim: a company, a role, a salary,
a location, a requirement, a benefit, a recommendation).
Step 2: label every claim using ONLY the context, never world knowledge:
- supported: stated in, or directly entailed by, the context (paraphrase is fine).
- unsupported: not in the context, contradicts it, or adds specifics (numbers, policies, companies,
  requirements) the context does not contain. Attributing a fact to the wrong listing is unsupported.
- abstention: the claim says information is NOT available / not stated, declines to answer, or asks a
  clarifying question, AND the information is indeed absent from the context. If the answer says
  something is absent but the context contains it, label it unsupported.
Be strict: when in doubt, mark unsupported. If the answer has no factual content at all, return one
abstention claim describing it. Return at least one claim."""

_USER = """CONTEXT (retrieved job listings):
{context}

ANSWER to fact-check:
{answer}"""


class FaithfulnessMetric(BaseMetric):
    name = "faithfulness"

    async def evaluate(self, case: GoldenCaseView, response: RAGQueryResponse) -> MetricResult:
        verdict = await self.judge.ask(
            FaithfulnessVerdict,
            system=_SYSTEM,
            user=_USER.format(
                context=render_chunks(response.retrieved_chunks),
                answer=response.answer.as_text(),
            ),
        )
        total = len(verdict.claims)
        faithful = sum(1 for c in verdict.claims if c.verdict in ("supported", "abstention"))
        unsupported = [c.claim for c in verdict.claims if c.verdict == "unsupported"]
        score = faithful / total if total else 0.0
        return self._result(
            score,
            verdict.summary,
            total_claims=total,
            supported=sum(1 for c in verdict.claims if c.verdict == "supported"),
            abstentions=sum(1 for c in verdict.claims if c.verdict == "abstention"),
            unsupported_claims=unsupported,
            claims=[c.model_dump() for c in verdict.claims],
        )
