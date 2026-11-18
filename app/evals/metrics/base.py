"""Shared base for LLM-as-a-Judge metrics."""

from abc import ABC, abstractmethod
from typing import ClassVar, List

from app.evals.judge import Judge
from app.schemas.evals import GoldenCaseView, MetricResult
from app.schemas.rag import RAGQueryResponse, RetrievedChunk


def render_chunks(chunks: List[RetrievedChunk], *, label: str = "chunk") -> str:
    if not chunks:
        return "(no context was retrieved)"
    return "\n\n".join(f"[{label}_index={i}]\n{c.content}" for i, c in enumerate(chunks))


class BaseMetric(ABC):
    name: ClassVar[str]

    def __init__(self, judge: Judge, threshold: float) -> None:
        self.judge = judge
        self.threshold = threshold

    def _result(self, score: float, reasoning: str, **details: object) -> MetricResult:
        score = round(max(0.0, min(1.0, score)), 4)
        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            applicable=True,
            reasoning=reasoning,
            details=dict(details),
        )

    @abstractmethod
    async def evaluate(self, case: GoldenCaseView, response: RAGQueryResponse) -> MetricResult: ...
