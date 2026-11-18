"""Async database lifecycle: engine, session factory, schema bootstrap.

A `Database` instance is created once in the FastAPI lifespan (and by CLI scripts /
tests) instead of module-level globals, so every entrypoint controls its own engine.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings
from app.db.models import Base

logger = logging.getLogger("app.db")


class Database:
    def __init__(self, settings: Settings) -> None:
        # TODO(database): the DSN comes from DATABASE_URL or the POSTGRES_* settings (.env).
        self.engine: AsyncEngine = create_async_engine(
            settings.sqlalchemy_database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            echo=settings.db_echo_sql,
        )
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_schema(self) -> None:
        """Ensure the pgvector extension exists, then create any missing tables/indexes.

        `create_all` is additive only (never alters existing columns), which is enough
        for this project; swap in Alembic if the schema starts evolving in production.
        """
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database schema ready (tables: %s)", ", ".join(Base.metadata.tables))

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - readiness probe must never raise
            logger.warning("database ping failed: %s", exc)
            return False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
