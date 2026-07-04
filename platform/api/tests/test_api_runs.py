import pytest

from evalforge.db.models import Prompt, PromptVersion, Suite
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
