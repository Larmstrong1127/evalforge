"""FastAPI dependency: one AsyncSession per request.

This is distinct from the CLI's per-command session (evalforge/cli.py) and
from what a BackgroundTask must do (open its own session via
make_session_factory() directly — see evalforge/api/runs.py — never reuse
this request-scoped session, since it's torn down as part of the request/
response lifecycle and BackgroundTasks run after the response is sent).
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.config import Settings
from evalforge.db.engine import make_engine, make_session_factory

_settings = Settings()
_engine = make_engine(_settings)
_session_factory = make_session_factory(_engine)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
