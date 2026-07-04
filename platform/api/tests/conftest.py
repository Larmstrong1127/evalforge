import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from evalforge.db.models import Base
from evalforge.db.session import get_session
from evalforge.main import app


@pytest.fixture
async def session_factory():
    """The same engine/factory is reused by `session`, `api_client`'s
    get_session override, and (in test_api_runs.py) a monkeypatched stand-in
    for the background task's own session creation — so a run created via
    the test's HTTP client and the "background task" that processes it are
    guaranteed to see the same in-memory database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
async def api_client(session, session_factory):
    """An httpx AsyncClient wired to the FastAPI app, with the real
    get_session dependency overridden to use the test's in-memory session
    instead of opening a real database connection."""

    async def _override_get_session():
        yield session

    app.dependency_overrides[get_session] = _override_get_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()
