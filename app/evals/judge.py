"""LLM-as-a-Judge client: one strict structured call per metric.

The judge model should differ from (and ideally be stronger than) the generator model;
scoring your own output with the same model biases scores upward.
"""

import logging
from typing import Any, Dict, Optional, Protocol, Type, TypeVar

import instructor
from instructor.exceptions import InstructorRetryException
from openai import APIError
from pydantic import BaseModel

from app.rag.llm_client import LLMNotConfiguredError

logger = logging.getLogger("app.evals.judge")

T = TypeVar("T", bound=BaseModel)


class JudgeError(RuntimeError):
    """Judge call failed (provider error or verdict never matched the schema)."""


class Judge(Protocol):
    model: str

    async def ask(
        self,
        response_model: Type[T],
        *,
        system: str,
        user: str,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> T: ...


class LLMJudge:
    def __init__(
        self,
        client: instructor.AsyncInstructor,
        *,
        model: str,
        max_retries: int = 3,
        enabled: bool = True,
    ) -> None:
        self._client = client
        self.model = model
        self._max_retries = max_retries
        self._enabled = enabled

    async def ask(
        self,
        response_model: Type[T],
        *,
        system: str,
        user: str,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> T:
        if not self._enabled:
            raise LLMNotConfiguredError("LLM_API_KEY is not set; judge is disabled")  # TODO(apikey)
        try:
            return await self._client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_retries=self._max_retries,
                context=validation_context,
            )
        except InstructorRetryException as exc:
            raise JudgeError(
                f"{response_model.__name__} never satisfied its schema after {exc.n_attempts} attempts"
            ) from exc
        except APIError as exc:
            raise JudgeError(f"judge provider error ({type(exc).__name__}): {exc}") from exc
