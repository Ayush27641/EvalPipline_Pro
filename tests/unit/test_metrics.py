"""Metric scoring maths with a fake judge (no network)."""

import pytest

from app.evals.judge import JudgeError
from app.evals.metrics.answer_relevancy import AnswerRelevancyMetric, RelevancyVerdict, StatementVerdict
from app.evals.metrics.contextual_precision import (
    ChunkVerdict,
    ContextualPrecisionMetric,
    ContextualPrecisionVerdict,
    weighted_cumulative_precision,
)
from app.evals.metrics.faithfulness import ClaimVerdict, FaithfulnessMetric, FaithfulnessVerdict
from tests.conftest import FakeJudge, make_case, make_chunk, make_response

pytestmark = pytest.mark.unit
THRESHOLD = 0.85


# --------------------------------------------------------------------------- faithfulness
async def test_faithfulness_counts_supported_and_abstentions():
    verdict = FaithfulnessVerdict(
        claims=[
            ClaimVerdict(claim="Acme hires in Bengaluru", verdict="supported"),
            ClaimVerdict(claim="Salary is not stated", verdict="abstention"),
            ClaimVerdict(claim="Offers visa sponsorship", verdict="unsupported"),
            ClaimVerdict(claim="Pays ₹30-40 LPA", verdict="supported"),
        ],
        summary="One invented benefit.",
    )
    metric = FaithfulnessMetric(FakeJudge({FaithfulnessVerdict: verdict}), THRESHOLD)
    result = await metric.evaluate(make_case(), make_response())
    assert result.score == 0.75
    assert result.passed is False
    assert result.details["unsupported_claims"] == ["Offers visa sponsorship"]


async def test_faithfulness_perfect_score_passes():
    verdict = FaithfulnessVerdict(
        claims=[ClaimVerdict(claim="c", verdict="supported")], summary="ok"
    )
    metric = FaithfulnessMetric(FakeJudge({FaithfulnessVerdict: verdict}), THRESHOLD)
    result = await metric.evaluate(make_case(), make_response())
    assert result.score == 1.0 and result.passed is True


def test_faithfulness_verdict_requires_at_least_one_claim():
    with pytest.raises(Exception):
        FaithfulnessVerdict(claims=[], summary="empty")


# --------------------------------------------------------------------------- relevancy
async def test_relevancy_partial_counts_half():
    verdict = RelevancyVerdict(
        statements=[
            StatementVerdict(statement="a", verdict="relevant"),
            StatementVerdict(statement="b", verdict="partial"),
            StatementVerdict(statement="c", verdict="irrelevant"),
            StatementVerdict(statement="d", verdict="relevant"),
        ],
        summary="mostly on topic",
    )
    metric = AnswerRelevancyMetric(FakeJudge({RelevancyVerdict: verdict}), THRESHOLD)
    result = await metric.evaluate(make_case(), make_response())
    assert result.score == 0.625
    assert result.details["irrelevant"] == 1


# --------------------------------------------------------------------------- contextual precision
@pytest.mark.parametrize(
    "relevance, expected",
    [
        ([True, True, False], 1.0),            # relevant chunks ranked first
        ([False, True], 0.5),                  # one relevant chunk at rank 2 -> 1/2
        ([True, False, True], (1 + 2 / 3) / 2),  # (1/1 + 2/3) / 2
        ([False, False], 0.0),
        ([], 0.0),
    ],
)
def test_weighted_cumulative_precision(relevance, expected):
    assert weighted_cumulative_precision(relevance) == pytest.approx(expected)


async def test_contextual_precision_scores_from_verdicts():
    verdict = ContextualPrecisionVerdict(
        verdicts=[ChunkVerdict(chunk_index=1, relevant=True), ChunkVerdict(chunk_index=0, relevant=False)],
        summary="best chunk ranked second",
    )
    judge = FakeJudge({ContextualPrecisionVerdict: verdict})
    metric = ContextualPrecisionMetric(judge, THRESHOLD)
    result = await metric.evaluate(make_case(), make_response([make_chunk(1), make_chunk(2, rank=2)]))
    assert result.score == 0.5
    assert result.details["relevance_by_rank"] == [False, True]  # sorted by chunk_index
    assert judge.calls[0]["ctx"] == {"expected_chunk_count": 2}


async def test_contextual_precision_not_applicable_for_unanswerable_case():
    judge = FakeJudge({})
    metric = ContextualPrecisionMetric(judge, THRESHOLD)
    result = await metric.evaluate(make_case(snippets=[], category="hallucination"), make_response())
    assert result.applicable is False and result.score is None and result.passed is None
    assert judge.calls == []  # no LLM call spent


async def test_contextual_precision_zero_when_nothing_retrieved():
    metric = ContextualPrecisionMetric(FakeJudge({}), THRESHOLD)
    result = await metric.evaluate(make_case(), make_response(chunks=[]))
    assert result.score == 0.0 and result.passed is False


def test_contextual_precision_verdict_validator_enforces_one_per_chunk():
    bad = {"verdicts": [{"chunk_index": 0, "relevant": True}], "summary": "s"}
    with pytest.raises(Exception, match="exactly one verdict"):
        ContextualPrecisionVerdict.model_validate(bad, context={"expected_chunk_count": 2})


async def test_judge_error_propagates_from_metric():
    metric = FaithfulnessMetric(FakeJudge({FaithfulnessVerdict: JudgeError("boom")}), THRESHOLD)
    with pytest.raises(JudgeError):
        await metric.evaluate(make_case(), make_response())
