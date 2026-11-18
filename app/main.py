"""FastAPI application entrypoint for EvalPipeline Pro.

Routes
  GET  /health                 liveness (no dependencies)
  GET  /ready                  readiness (database reachable)
  POST /api/v1/rag/query       target RAG service
  GET  /api/v1/rag/jobs        browse indexed listings
  POST /api/v1/evals/run       run the LLM-as-a-Judge suite -> PASS/FAIL report
  GET  /api/v1/evals/history   past runs (dashboard + regression tracking)
  GET  /api/v1/evals/runs/{id} full report with judge reasoning
  GET  /api/v1/evals/golden    active golden cases
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.bootstrap import build_app_stack
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.evals.judge import JudgeError
from app.rag.llm_client import GenerationError, LLMNotConfiguredError

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "starting %s v%s env=%s llm_provider=%s embedding_provider=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
        settings.llm_provider,
        settings.embedding_provider,
    )
    if not settings.llm_configured:
        logger.warning("LLM_API_KEY is empty — /rag/query and /evals/run will return 503")

    stack = await build_app_stack(settings)
    app.state.stack = stack
    app.state.db = stack.db
    app.state.rag_pipeline = stack.pipeline
    app.state.eval_runner = stack.runner
    try:
        yield
    finally:
        await stack.close()
        logger.info("shutdown complete")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Automated LLM Output Quality & RAG Evaluation Suite",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.include_router(api_v1_router, prefix="/api/v1")

    # ------------------------------------------------------------------ error mapping
    @app.exception_handler(LLMNotConfiguredError)
    async def _llm_not_configured(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "hint": "set LLM_API_KEY in .env"},
        )

    @app.exception_handler(GenerationError)
    async def _generation_error(_: Request, exc: GenerationError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    @app.exception_handler(JudgeError)
    async def _judge_error(_: Request, exc: JudgeError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})

    # ------------------------------------------------------------------ probes
    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "llm_configured": settings.llm_configured,
        }

    @app.get("/ready", tags=["ops"], summary="Readiness probe (checks the database)")
    async def ready(request: Request) -> JSONResponse:
        db = getattr(request.app.state, "db", None)
        db_ok = await db.ping() if db is not None else False
        return JSONResponse(
            status_code=status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready" if db_ok else "degraded", "database": "up" if db_ok else "down"},
        )

    return app


app = create_app()
