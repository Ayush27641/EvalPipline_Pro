"""HTTP layer with the pipeline/DB swapped for fakes. The lifespan is NOT run
(TestClient without a `with` block), so no database or model download is needed."""

from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db, get_eval_runner, get_rag_pipeline, get_retriever, get_session
from app.core.config import Settings
from app.main import create_app
from app.rag.llm_client import GenerationError, LLMNotConfiguredError
from tests.conftest import make_response

pytestmark = pytest.mark.unit


class FakePipeline:
    prompt_version = "v1"

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error
        self.calls = []

    async def run(self, session, query, top_k=None):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return make_response()


class FakeRetriever:
    async def count(self, session):
        return 30

    async def list_jobs(self, session, limit=50, offset=0, location=None):
        return []


@pytest.fixture
def client():
    app = create_app(Settings(llm_api_key="test-key", environment="test"))
    pipeline = FakePipeline()
    app.dependency_overrides[get_session] = lambda: None
    app.dependency_overrides[get_db] = lambda: None
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_eval_runner] = lambda: None
    # Plain client (no `with` block) => lifespan never runs => no DB / model download.
    test_client = TestClient(app, raise_server_exceptions=False)
    test_client.pipeline = pipeline
    return test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_configured"] is True


def test_rag_query_returns_structured_answer(client):
    res = client.post("/api/v1/rag/query", json={"query": "python backend jobs in bengaluru", "top_k": 3})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"]["recommendations"][0]["job_id"] == 1
    assert body["retrieved_chunks"][0]["rank"] == 1
    assert client.pipeline.calls == [("python backend jobs in bengaluru", 3)]


def test_rag_query_validates_input(client):
    assert client.post("/api/v1/rag/query", json={"query": "x"}).status_code == 422
    assert client.post("/api/v1/rag/query", json={"query": "ok query", "top_k": 99}).status_code == 422


def test_rag_query_503_when_llm_not_configured(client):
    client.pipeline.error = LLMNotConfiguredError("LLM_API_KEY is not set")
    res = client.post("/api/v1/rag/query", json={"query": "python backend jobs"})
    assert res.status_code == 503 and "LLM_API_KEY" in res.json()["detail"]


def test_rag_query_502_on_generation_error(client):
    client.pipeline.error = GenerationError("schema never satisfied")
    assert client.post("/api/v1/rag/query", json={"query": "python backend jobs"}).status_code == 502


def test_jobs_count(client):
    assert client.get("/api/v1/rag/jobs/count").json() == {"count": 30}
