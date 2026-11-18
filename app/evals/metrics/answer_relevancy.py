"""Answer Relevancy: does the answer address what the candidate actually asked?

The judge splits the answer into statements and labels each relevant / partial / irrelevant
to the query. score = (relevant + 0.5 * partial) / total. Accuracy is NOT judged here
(that is faithfulness); a well-justified "nothing matches" is fully relevant.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from app.evals.metrics.base import BaseMetric
from app.schemas.evals import GoldenCaseView, MetricResult
from app.schemas.rag import RAGQueryResponse


class StatementVerdict(BaseModel):
    statement: str
    verdict: Literal["relevant", "partial", "irrelevant"]
    reason: str = ""


class RelevancyVerdict(BaseModel):
    statements: List[StatementVerdict] = Field(..., min_length=1)
    summary: str = Field(..., description="2-3 sentences on how well the answer addresses the query")


_SYSTEM = """You evaluate whether a job-matching assistant's ANSWER addresses the candidate's QUERY.
Split the answer into individual statements and label each one:
- relevant: directly helps with the query - a matching role, a detail the candidate asked for,
  a justified statement that nothing matches, a note that a requested detail is not stated, or a
  clarifying question when the query is too vague to act on.
- partial: related but tangential, or useful background the candidate did not ask for.
- irrelevant: off-topic, filler, generic encouragement, or answering a different question.
Do NOT judge factual accuracy here; only whether each statement serves the query.
Return at least one statement."""

_USER = """QUERY:
{query}

ANSWER:
{answer}"""


class AnswerRelevancyMetric(BaseMetric):
    name = "answer_relevancy"

    async def evaluate(self, case: GoldenCaseView, response: RAGQueryResponse) -> MetricResult:
        verdict = await self.judge.ask(
            RelevancyVerdict,
            system=_SYSTEM,
            user=_USER.format(query=case.query, answer=response.answer.as_text()),
        )
        total = len(verdict.statements)
        relevant = sum(1 for s in verdict.statements if s.verdict == "relevant")
        partial = sum(1 for s in verdict.statements if s.verdict == "partial")
        score = (relevant + 0.5 * partial) / total if total else 0.0
        return self._result(
            score,
            verdict.summary,
            total_statements=total,
            relevant=relevant,
            partial=partial,
            irrelevant=total - relevant - partial,
            statements=[s.model_dump() for s in verdict.statements],
        )
