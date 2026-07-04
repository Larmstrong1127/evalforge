import asyncio

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from evalforge.db.engine import make_session_factory
from evalforge.db.models import Base, Prompt, PromptVersion, Suite
from evalforge.db.session import get_session
from evalforge.providers import Completion


class FakeProvider:
    name = "fake"

    async def generate(self, model: str, prompt: str) -> Completion:
        return Completion(text=f"answer to {prompt}", input_tokens=10, output_tokens=5)


@pytest.fixture
def patch_fake_provider(monkeypatch, session_factory):
    """Registers 'fake' as a real provider name so POST /runs' candidate
    parsing (provider:model) accepts 'fake:model' without hitting a real API,
    and points the background task's session creation at the SAME in-memory
    engine the test's `session`/`api_client` fixtures use (see conftest.py's
    session_factory fixture docstring for why this matters — a real
    make_engine(settings) call in the background task would otherwise open
    a completely separate, empty :memory: database)."""
    import evalforge.api.runs as runs_module

    def _fake_get_provider(name: str, settings):
        if name != "fake":
            raise KeyError(name)  # matches the real get_provider's dict-lookup behavior
        return FakeProvider()

    monkeypatch.setattr(runs_module, "get_provider", _fake_get_provider)
    monkeypatch.setattr(
        runs_module, "_make_background_session_factory", lambda settings: session_factory
    )


async def _make_suite_with_prompt(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    version = PromptVersion(prompt=prompt, version_number=1, input_text="q1", expected_output="a1")
    session.add_all([suite, prompt, version])
    await session.commit()
    return suite


async def test_create_run_returns_202_and_id(api_client, session, patch_fake_provider):
    suite = await _make_suite_with_prompt(session)
    response = await api_client.post(
        "/api/v1/runs",
        json={"suite_id": str(suite.id), "candidates": ["fake:model-a"], "judges": ["exact_match"]},
    )
    assert response.status_code == 202
    assert "run_id" in response.json()


async def test_create_run_with_missing_suite_returns_404(api_client, patch_fake_provider):
    response = await api_client.post(
        "/api/v1/runs",
        json={
            "suite_id": "00000000-0000-0000-0000-000000000000",
            "candidates": ["fake:model-a"],
        },
    )
    assert response.status_code == 404


async def test_create_run_with_malformed_candidate_returns_400(api_client, session):
    suite = await _make_suite_with_prompt(session)
    response = await api_client.post(
        "/api/v1/runs",
        json={"suite_id": str(suite.id), "candidates": ["not-a-valid-spec"]},
    )
    assert response.status_code == 400


async def test_create_run_with_unknown_provider_returns_400(api_client, session):
    suite = await _make_suite_with_prompt(session)
    response = await api_client.post(
        "/api/v1/runs",
        json={"suite_id": str(suite.id), "candidates": ["totally-unknown-provider:model"]},
    )
    assert response.status_code == 400


async def test_get_run_status_returns_run_fields(api_client, session, patch_fake_provider):
    suite = await _make_suite_with_prompt(session)
    create_resp = await api_client.post(
        "/api/v1/runs",
        json={"suite_id": str(suite.id), "candidates": ["fake:model-a"], "judges": []},
    )
    run_id = create_resp.json()["run_id"]

    response = await api_client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["status"] in ("queued", "running", "completed", "failed")


async def test_get_missing_run_returns_404(api_client):
    response = await api_client.get("/api/v1/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_create_run_row_is_visible_to_background_task_with_separate_engines(
    tmp_path, monkeypatch
):
    """Regression test for a real bug caught via a live E2E run against
    Ollama: BackgroundTasks execute before get_session's post-yield commit
    (both run inside the same ASGI middleware layer that later closes the
    dependency's AsyncExitStack), so a flush()-only Run row was invisible to
    the background task's independently-opened session. Every other test in
    this file monkeypatches _make_background_session_factory to share the
    SAME in-memory engine as the request, which masks this exact bug — this
    test deliberately uses two genuinely separate engines against the same
    file-backed SQLite database, mirroring production."""
    import evalforge.api.runs as runs_module
    from evalforge.main import app

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'regression.db'}"
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", db_url)

    request_engine = create_async_engine(db_url)
    async with request_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    request_factory = make_session_factory(request_engine)

    def _fake_get_provider(name: str, settings):
        if name != "fake":
            raise KeyError(name)
        return FakeProvider()

    monkeypatch.setattr(runs_module, "get_provider", _fake_get_provider)
    # Deliberately NOT monkeypatching _make_background_session_factory: it
    # opens its own real engine via Settings().database_url, which now
    # points at the same file thanks to the env var set above.

    async def _override_get_session():
        async with request_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = _override_get_session
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with request_factory() as session:
                suite = await _make_suite_with_prompt(session)
            response = await client.post(
                "/api/v1/runs",
                json={"suite_id": str(suite.id), "candidates": ["fake:model-a"], "judges": []},
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]

            status_body = {"status": "queued"}
            for _ in range(20):
                status_response = await client.get(f"/api/v1/runs/{run_id}")
                status_body = status_response.json()
                if status_body["status"] != "queued":
                    break
                await asyncio.sleep(0.05)
            assert status_body["status"] == "completed"
    finally:
        app.dependency_overrides.clear()
        await request_engine.dispose()
