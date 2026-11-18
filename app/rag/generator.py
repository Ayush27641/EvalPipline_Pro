"""Structured answer generation with `instructor`.

The LLM must return a `RAGAnswer`. If the JSON is malformed, a field is missing, or
the grounding validator rejects it (unknown job_id, wrong company, recommendations
on an unanswerable query), instructor re-sends the request with the validation
error appended, up to `max_retries` times. Provider-level failures (429/5xx/timeouts)
are retried with backoff inside the OpenAI client.
"""

import logging
from typing import Any, Dict, List

import instructor
from instructor.exceptions import InstructorRetryException
from openai import APIError

from app.rag.llm_client import GenerationError, LLMNotConfiguredError
from app.rag.prompts import get_prompt
from app.schemas.rag import RAGAnswer, RetrievedChunk

logger = logging.getLogger("app.rag.generator")


class AnswerGenerator:
    def __init__(
        self,
        client: instructor.AsyncInstructor,
        *,
        model: str,
        prompt_version: str,
        max_retries: int = 3,
        temperature: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self._client = client
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = get_prompt(prompt_version)  # fail fast on an unknown version
        self._max_retries = max_retries
        self._temperature = temperature
        self._enabled = enabled

    def with_prompt_version(self, prompt_version: str) -> "AnswerGenerator":
        return AnswerGenerator(
            self._client,
            model=self.model,
            prompt_version=prompt_version,
            max_retries=self._max_retries,
            temperature=self._temperature,
            enabled=self._enabled,
        )

    async def generate(self, query: str, chunks: List[RetrievedChunk]) -> RAGAnswer:
        if not self._enabled:
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set; generation is disabled"  # TODO(apikey)
            )

        validation_context: Dict[str, Any] = {
            "allowed_job_ids": {c.job_id for c in chunks},
            "jobs": {c.job_id: {"title": c.title, "company": c.company} for c in chunks},
        }
        try:
            answer: RAGAnswer = await self._client.chat.completions.create(
                model=self.model,
                response_model=RAGAnswer,
                messages=self._prompt.build_messages(query, chunks),
                temperature=self._temperature,
                max_retries=self._max_retries,
                context=validation_context,
            )
        except InstructorRetryException as exc:
            logger.warning("generation failed schema validation after %s attempts", exc.n_attempts)
            raise GenerationError(
                f"LLM output did not satisfy the RAGAnswer schema after {exc.n_attempts} attempts: {exc}"
            ) from exc
        except APIError as exc:
            raise GenerationError(f"LLM provider error ({type(exc).__name__}): {exc}") from exc
        return answer
