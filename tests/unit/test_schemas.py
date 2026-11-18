"""RAGAnswer grounding validator + request schema bounds (no LLM, no DB)."""

import pytest
from pydantic import ValidationError

from app.schemas.evals import EvalRunRequest
from app.schemas.rag import RAGAnswer, RAGQueryRequest
from tests.conftest import make_answer

pytestmark = pytest.mark.unit


def _ctx(*job_ids: int, company: str = "Acme") -> dict:
    return {
        "allowed_job_ids": set(job_ids),
        "jobs": {j: {"title": "Backend Engineer", "company": company} for j in job_ids},
    }


def test_answer_accepts_grounded_recommendation():
    payload = make_answer([1]).model_dump()
    answer = RAGAnswer.model_validate(payload, context=_ctx(1, 2))
    assert answer.recommendations[0].job_id == 1


def test_answer_rejects_hallucinated_job_id():
    payload = make_answer([99]).model_dump()
    with pytest.raises(ValidationError, match="NOT in the retrieved listings"):
        RAGAnswer.model_validate(payload, context=_ctx(1, 2))


def test_answer_rejects_wrong_company_for_job_id():
    payload = make_answer([1]).model_dump()
    with pytest.raises(ValidationError, match="belongs to company"):
        RAGAnswer.model_validate(payload, context=_ctx(1, company="Globex"))


def test_unanswerable_with_recommendations_is_rejected():
    payload = make_answer([1]).model_dump()
    payload["is_answerable"] = False
    with pytest.raises(ValidationError, match="is_answerable is false"):
        RAGAnswer.model_validate(payload, context=_ctx(1))


def test_validator_is_skipped_without_context():
    """Reading historical answers back from the DB has no retrieval context."""
    payload = make_answer([424242]).model_dump()
    assert RAGAnswer.model_validate(payload).recommendations[0].job_id == 424242


def test_as_text_includes_recommendations_and_caveats():
    answer = make_answer([1])
    answer.caveats = ["Salary is not stated."]
    text = answer.as_text()
    assert "job_id=1" in text and "Caveats: Salary is not stated." in text


def test_query_request_bounds():
    with pytest.raises(ValidationError):
        RAGQueryRequest(query="x")
    with pytest.raises(ValidationError):
        RAGQueryRequest(query="valid query", top_k=0)
    assert RAGQueryRequest(query="valid query", top_k=5).top_k == 5


def test_eval_run_request_threshold_bounds():
    with pytest.raises(ValidationError):
        EvalRunRequest(quality_threshold=1.5)
    assert EvalRunRequest().quality_threshold is None
