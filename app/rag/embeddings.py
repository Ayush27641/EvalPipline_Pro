"""Embedding providers behind one async interface.

local  -> sentence-transformers (CPU, no API key). Encoding is CPU-bound and
          synchronous, so it runs in the default thread-pool executor to keep the
          event loop free.
openai -> text-embedding-3-small via the OpenAI SDK.  # TODO(apikey)
"""

import asyncio
import functools
import logging
from typing import List, Optional, Protocol

from app.core.config import Settings

logger = logging.getLogger("app.rag.embeddings")


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    async def load(self) -> None: ...

    async def embed(self, texts: List[str]) -> List[List[float]]: ...

    async def embed_query(self, text: str) -> List[float]: ...


class EmbeddingDimensionMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------- local
class LocalEmbeddingProvider:
    """sentence-transformers model, normalised vectors (cosine == dot product)."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        self.name = model_name
        self.dim = dim
        self._batch_size = batch_size
        self._model = None  # type: Optional[object]
        self._lock = asyncio.Lock()

    def _load_sync(self) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        model = SentenceTransformer(self.name, device="cpu")
        actual = model.get_sentence_embedding_dimension()
        if actual != self.dim:
            raise EmbeddingDimensionMismatch(
                f"EMBEDDING_DIM={self.dim} but model {self.name!r} produces {actual}-d vectors. "
                "Fix EMBEDDING_DIM (and re-seed job_listings)."
            )
        self._model = model

    async def load(self) -> None:
        async with self._lock:
            if self._model is None:
                logger.info("loading embedding model %s (cpu)", self.name)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._load_sync)
                logger.info("embedding model ready (dim=%d)", self.dim)

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        assert self._model is not None
        vectors = self._model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [row.astype(float).tolist() for row in vectors]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is None:
            await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(self._encode_sync, texts))

    async def embed_query(self, text: str) -> List[float]:
        return (await self.embed([text]))[0]


# --------------------------------------------------------------------------- openai
class OpenAIEmbeddingProvider:
    def __init__(self, model_name: str, dim: int, api_key: str) -> None:
        self.name = model_name
        self.dim = dim
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_EMBEDDING_API_KEY"  # TODO(apikey)
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)

    async def load(self) -> None:  # nothing to warm up; validate dimension on first call
        return None

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self.name, input=texts, dimensions=self.dim)
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        if vectors and len(vectors[0]) != self.dim:
            raise EmbeddingDimensionMismatch(
                f"EMBEDDING_DIM={self.dim} but {self.name!r} returned {len(vectors[0])}-d vectors"
            )
        return vectors

    async def embed_query(self, text: str) -> List[float]:
        return (await self.embed([text]))[0]


# --------------------------------------------------------------------------- factory
def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            api_key=settings.openai_embedding_api_key.get_secret_value(),
        )
    return LocalEmbeddingProvider(model_name=settings.embedding_model, dim=settings.embedding_dim)
