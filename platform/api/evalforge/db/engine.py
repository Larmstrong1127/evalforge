"""Async engine/session factory.

SQLite (aiosqlite) is the zero-setup default; Postgres via DATABASE_URL.
Schema is created with create_all for now; Alembic is introduced with the
first post-release schema change (ADR-006 will record that call).
"""
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from evalforge.config import Settings
from evalforge.db.models import Base


def make_engine(settings: Settings) -> AsyncEngine:
    # File-based SQLite serializes concurrent writes (fine for dev); Postgres
    # is the concurrent-write path.
    return create_async_engine(settings.database_url)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
