from typing import List

from app.evals.judge import Judge
from app.evals.metrics.answer_relevancy import AnswerRelevancyMetric
from app.evals.metrics.base import BaseMetric
from app.evals.metrics.contextual_precision import (
    ContextualPrecisionMetric,
    weighted_cumulative_precision,
)
from app.evals.metrics.faithfulness import FaithfulnessMetric


def build_default_metrics(judge: Judge, threshold: float) -> List[BaseMetric]:
    return [
        FaithfulnessMetric(judge, threshold),
        AnswerRelevancyMetric(judge, threshold),
        ContextualPrecisionMetric(judge, threshold),
    ]


__all__ = [
    "AnswerRelevancyMetric",
    "BaseMetric",
    "ContextualPrecisionMetric",
    "FaithfulnessMetric",
    "build_default_metrics",
    "weighted_cumulative_precision",
]
