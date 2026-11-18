"""OpenAI-compatible async client wrapped with `instructor` for strict structured output.

Works for Groq (default) and OpenAI by switching LLM_BASE_URL / LLM_API_KEY only.
"""

import instructor
from openai import AsyncOpenAI

from app.core.config import Settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when an endpoint needs the LLM but LLM_API_KEY is empty."""


class GenerationError(RuntimeError):
    """The provider failed or the output never satisfied the Pydantic schema."""


def build_instructor_client(settings: Settings) -> instructor.AsyncInstructor:
    base_client = AsyncOpenAI(
        api_key=settings.llm_api_key.get_secret_value() or "not-configured",  # TODO(apikey)
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,  # 429 / 5xx / connection errors, exponential backoff
    )
    # Groq is most reliable in JSON mode (schema injected into the system prompt +
    # response_format=json_object). OpenAI gets native tool/function calling.
    mode = instructor.Mode.JSON if settings.llm_provider == "groq" else instructor.Mode.TOOLS
    return instructor.from_openai(base_client, mode=mode)
