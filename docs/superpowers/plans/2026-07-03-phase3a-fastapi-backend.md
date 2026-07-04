# EvalForge Phase 3a: FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing, already-tested runner/DB/CLI logic in a real
FastAPI HTTP API, so a future Next.js dashboard has something to call.

**Architecture:** Thin route handlers over the existing `evalforge.runner`,
`evalforge.db.models`, `evalforge.providers`, `evalforge.judges` modules —
no new business logic. Each request gets its own `AsyncSession` via FastAPI
dependency injection. `POST /runs` returns `202` immediately and hands
execution to a `BackgroundTasks` wrapper that opens its **own** session
(never the request's) before calling the already-tested `execute_run()`.

**Tech Stack:** FastAPI, Pydantic v2 (already a dependency), httpx's
`ASGITransport` for in-process testing (not a real server), uvicorn for
actually running the server.

**Conventions:** Run commands from `platform/api/`. Conventional commits, no
AI attribution. TDD throughout: write the test, watch it fail, implement,
watch it pass.

---

## File structure (locked in by this plan)

```
platform/api/evalforge/
  main.py                  # FastAPI app: CORS, router mounting
  db/
    session.py               # NEW: get_session() dependency
  api/
    __init__.py
    suites.py                # POST/GET /suites, POST /suites/{id}/prompts
    prompts.py                 # POST /prompts/{prompt_id}/versions
    runs.py                     # POST /runs, GET /runs/{id}, /results, /costs
    ratings.py                    # POST /ratings
    compare.py                      # GET /compare
  schemas/
    __init__.py
    suites.py                    # PromptCreate, SuiteCreate, SuiteResponse
    runs.py                        # RunCreate, RunAccepted, RunStatusResponse, ResultResponse, JudgeEvaluationResponse, CostResponse
    ratings.py                       # RatingCreate, RatingResponse
    compare.py                         # CompareRow, CompareResponse
tests/
  conftest.py                          # MODIFY: add an httpx AsyncClient fixture wired to a test DB
  test_api_suites.py
  test_api_prompts.py
  test_api_runs.py
  test_api_ratings.py
  test_api_compare.py
```

---

### Task 1: Package scaffold — FastAPI app, CORS, session dependency

**Files:**
- Modify: `platform/api/pyproject.toml`
- Create: `platform/api/evalforge/db/session.py`
- Create: `platform/api/evalforge/main.py`
- Create: `platform/api/evalforge/api/__init__.py` (empty)
- Create: `platform/api/evalforge/schemas/__init__.py` (empty)
- Modify: `platform/api/tests/conftest.py`

- [ ] **Step 1: Add FastAPI to dependencies**

In `platform/api/pyproject.toml`, add to the `dependencies` list (after `"typer>=0.12",`):

```toml
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
```

- [ ] **Step 2: Install**

Run: `.venv\Scripts\pip install -e ".[dev]"`

- [ ] **Step 3: Write `evalforge/db/session.py`**

```python
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
from evalforge.db.engine import init_db, make_engine, make_session_factory

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
```

- [ ] **Step 4: Write `evalforge/main.py`**

```python
"""FastAPI app entry point. Run with: uvicorn evalforge.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evalforge.api import compare, prompts, ratings, runs, suites
from evalforge.db.session import _engine
from evalforge.db.engine import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(_engine)
    yield


app = FastAPI(title="EvalForge API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(suites.router, prefix="/api/v1")
app.include_router(prompts.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(ratings.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
```

(The router imports will fail until Tasks 2-6 create those modules — that's
expected; this file is completed in Task 6's final wiring step. For now,
just create the two directories below and confirm the app can eventually
import.)

- [ ] **Step 5: Create the two empty package `__init__.py` files**

`platform/api/evalforge/api/__init__.py` — empty file.
`platform/api/evalforge/schemas/__init__.py` — empty file.

- [ ] **Step 6: Add shared test fixtures for the API client and its session factory**

Read `platform/api/tests/conftest.py` first to see the existing `session`
fixture (it creates its own local `engine`/`factory` and doesn't expose
them). **Important architectural note for Task 4's background-task tests:**
SQLite `:memory:` databases are unique per connection/engine — a background
task that opens a brand-new engine via `make_engine(settings)` would connect
to a completely different, empty database than whatever the test's request
used, even though both say `:memory:`. So the test fixtures need to expose
the *same* session factory to both the HTTP layer (via FastAPI's
`get_session` override) and anything that needs to simulate "the background
task's own session" — Task 4 will monkeypatch a seam function for exactly
this reason.

Add these fixtures alongside the existing `session` fixture (don't remove
it — other tests still use it directly). Restructure so `session` is now
built from a new `session_factory` fixture, keeping `session`'s existing
public behavior unchanged for every test that already depends on it:

```python
import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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
```

`Base` needs to be imported from `evalforge.db.models` if the existing
`session` fixture didn't already import it under that name — check the
existing import at the top of `conftest.py` and reuse it rather than adding
a duplicate import.

Note: this fixture can't be exercised yet since `evalforge.main` imports
routers that don't exist until later tasks — that's fine, later tasks are
what make this importable. Do not attempt to run tests yet in this task.

- [ ] **Step 7: Commit**

```bash
git add platform/api/pyproject.toml platform/api/evalforge/db/session.py platform/api/evalforge/main.py platform/api/evalforge/api/__init__.py platform/api/evalforge/schemas/__init__.py platform/api/tests/conftest.py
git commit -m "chore: scaffold FastAPI app, CORS, and per-request session dependency"
```

---

### Task 2: Suites endpoints

**Files:**
- Create: `platform/api/evalforge/schemas/suites.py`
- Create: `platform/api/evalforge/api/suites.py`
- Test: `platform/api/tests/test_api_suites.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_suites.py
async def test_create_suite_returns_id_and_prompt_count(api_client):
    response = await api_client.post(
        "/api/v1/suites",
        json={
            "name": "demo-qa",
            "description": "a demo suite",
            "prompts": [
                {"input_text": "What is 2+2?", "expected_output": "4"},
                {"input_text": "Capital of France?", "expected_output": "Paris"},
            ],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "demo-qa"
    assert body["description"] == "a demo suite"
    assert body["prompt_count"] == 2
    assert "id" in body


async def test_create_suite_requires_name(api_client):
    response = await api_client.post("/api/v1/suites", json={"prompts": []})
    assert response.status_code == 422


async def test_list_suites_returns_created_suites(api_client):
    await api_client.post("/api/v1/suites", json={"name": "suite-a", "prompts": []})
    await api_client.post("/api/v1/suites", json={"name": "suite-b", "prompts": []})

    response = await api_client.get("/api/v1/suites")
    assert response.status_code == 200
    names = {s["name"] for s in response.json()}
    assert names == {"suite-a", "suite-b"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_api_suites.py -v`
Expected: FAIL — `ModuleNotFoundError: evalforge.main` (routers don't exist
yet, so `main.py`'s imports fail). This is expected per Task 1's note.

- [ ] **Step 3: Write `evalforge/schemas/suites.py`**

```python
from uuid import UUID

from pydantic import BaseModel


class PromptCreate(BaseModel):
    input_text: str
    expected_output: str | None = None


class SuiteCreate(BaseModel):
    name: str
    description: str | None = None
    prompts: list[PromptCreate] = []


class SuiteResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    prompt_count: int
```

- [ ] **Step 4: Write `evalforge/api/suites.py`**

```python
from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import Prompt, PromptVersion, Suite
from evalforge.db.session import get_session
from evalforge.schemas.suites import SuiteCreate, SuiteResponse

router = APIRouter(tags=["suites"])


@router.post("/suites", response_model=SuiteResponse, status_code=status.HTTP_201_CREATED)
async def create_suite(
    body: SuiteCreate, session: AsyncSession = Depends(get_session)
) -> SuiteResponse:
    suite = Suite(name=body.name, description=body.description)
    session.add(suite)
    for p in body.prompts:
        prompt = Prompt(suite=suite)
        session.add(prompt)
        session.add(
            PromptVersion(
                prompt=prompt,
                version_number=1,
                input_text=p.input_text,
                expected_output=p.expected_output,
            )
        )
    await session.flush()
    return SuiteResponse(
        id=suite.id,
        name=suite.name,
        description=suite.description,
        prompt_count=len(body.prompts),
    )


@router.get("/suites", response_model=list[SuiteResponse])
async def list_suites(session: AsyncSession = Depends(get_session)) -> list[SuiteResponse]:
    suites = (await session.execute(select(Suite))).scalars().all()
    result = []
    for s in suites:
        count = (
            await session.execute(
                select(func.count(Prompt.id)).where(Prompt.suite_id == s.id)
            )
        ).scalar_one()
        result.append(
            SuiteResponse(id=s.id, name=s.name, description=s.description, prompt_count=count)
        )
    return result
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_api_suites.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv\Scripts\ruff check evalforge tests
.venv\Scripts\mypy evalforge
git add platform/api/evalforge/schemas/suites.py platform/api/evalforge/api/suites.py platform/api/tests/test_api_suites.py
git commit -m "feat: add POST/GET /suites endpoints"
```

---

### Task 3: Prompt-versioning endpoints (split routes, per design)

**Files:**
- Modify: `platform/api/evalforge/schemas/suites.py`
- Create: `platform/api/evalforge/schemas/prompts.py`
- Modify: `platform/api/evalforge/api/suites.py`
- Create: `platform/api/evalforge/api/prompts.py`
- Test: `platform/api/tests/test_api_prompts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_prompts.py
async def test_create_prompt_under_suite_starts_at_version_1(api_client):
    suite_resp = await api_client.post("/api/v1/suites", json={"name": "s", "prompts": []})
    suite_id = suite_resp.json()["id"]

    response = await api_client.post(
        f"/api/v1/suites/{suite_id}/prompts",
        json={"input_text": "q1", "expected_output": "a1"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["version_number"] == 1
    assert "prompt_id" in body


async def test_create_prompt_under_missing_suite_returns_404(api_client):
    response = await api_client.post(
        "/api/v1/suites/00000000-0000-0000-0000-000000000000/prompts",
        json={"input_text": "q"},
    )
    assert response.status_code == 404


async def test_add_version_increments_version_number(api_client):
    suite_resp = await api_client.post("/api/v1/suites", json={"name": "s", "prompts": []})
    suite_id = suite_resp.json()["id"]
    create_resp = await api_client.post(
        f"/api/v1/suites/{suite_id}/prompts", json={"input_text": "q1"}
    )
    prompt_id = create_resp.json()["prompt_id"]

    response = await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions", json={"input_text": "q1-edited"}
    )
    assert response.status_code == 201
    assert response.json()["version_number"] == 2


async def test_add_version_to_missing_prompt_returns_404(api_client):
    response = await api_client.post(
        "/api/v1/prompts/00000000-0000-0000-0000-000000000000/versions",
        json={"input_text": "q"},
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_api_prompts.py -v`
Expected: FAIL — the `POST /suites/{id}/prompts` route doesn't exist yet
(only bare `/suites` does), and `evalforge.api.prompts` doesn't exist.

- [ ] **Step 3: Write `evalforge/schemas/prompts.py`**

```python
from uuid import UUID

from pydantic import BaseModel


class PromptVersionResponse(BaseModel):
    prompt_id: UUID
    version_number: int
```

- [ ] **Step 4: Add the create-prompt-under-suite route to `evalforge/api/suites.py`**

Add these imports to the top of the existing file (merge with existing
imports, don't duplicate):

```python
from fastapi import HTTPException
from evalforge.schemas.suites import PromptCreate
from evalforge.schemas.prompts import PromptVersionResponse
```

Append this route at the end of `evalforge/api/suites.py`:

```python
@router.post(
    "/suites/{suite_id}/prompts",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_prompt(
    suite_id: str, body: PromptCreate, session: AsyncSession = Depends(get_session)
) -> PromptVersionResponse:
    suite = await session.get(Suite, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id} not found")
    prompt = Prompt(suite=suite)
    session.add(prompt)
    version = PromptVersion(
        prompt=prompt,
        version_number=1,
        input_text=body.input_text,
        expected_output=body.expected_output,
    )
    session.add(version)
    await session.flush()
    return PromptVersionResponse(prompt_id=prompt.id, version_number=1)
```

- [ ] **Step 5: Write `evalforge/api/prompts.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import Prompt, PromptVersion
from evalforge.db.session import get_session
from evalforge.schemas.prompts import PromptVersionResponse
from evalforge.schemas.suites import PromptCreate

router = APIRouter(tags=["prompts"])


@router.post(
    "/prompts/{prompt_id}/versions",
    response_model=PromptVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_prompt_version(
    prompt_id: str, body: PromptCreate, session: AsyncSession = Depends(get_session)
) -> PromptVersionResponse:
    prompt = await session.get(Prompt, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail=f"prompt {prompt_id} not found")
    max_version = (
        await session.execute(
            select(func.max(PromptVersion.version_number)).where(
                PromptVersion.prompt_id == prompt_id
            )
        )
    ).scalar_one()
    next_version = (max_version or 0) + 1
    version = PromptVersion(
        prompt=prompt,
        version_number=next_version,
        input_text=body.input_text,
        expected_output=body.expected_output,
    )
    session.add(version)
    await session.flush()
    return PromptVersionResponse(prompt_id=prompt.id, version_number=next_version)
```

- [ ] **Step 6: Update `evalforge/main.py`'s router imports**

The `from evalforge.api import compare, prompts, ratings, runs, suites` line
already written in Task 1 anticipates this module — no change needed here,
but `compare.py`, `ratings.py`, `runs.py` still don't exist, so the app
still can't fully import. Confirm this is still expected and move on.

- [ ] **Step 7: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_api_prompts.py -v`
Expected: still FAILS at collection (ModuleNotFoundError for runs/ratings/
compare) — this is expected until Tasks 4-6 land. **Do not skip ahead or
treat this as a real failure**; verify by reading the traceback that it's
specifically the missing `runs`/`ratings`/`compare` modules causing it, not
anything in prompts.py/suites.py.

- [ ] **Step 8: Lint and typecheck what exists so far, then commit**

```bash
.venv\Scripts\ruff check evalforge/schemas/prompts.py evalforge/api/prompts.py evalforge/api/suites.py tests/test_api_prompts.py
git add platform/api/evalforge/schemas/prompts.py platform/api/evalforge/api/prompts.py platform/api/evalforge/api/suites.py platform/api/tests/test_api_prompts.py
git commit -m "feat: add prompt-versioning endpoints (create under suite, append version)"
```

---

### Task 4: Runs endpoints — create (with correct background-task session handling), get status

**Files:**
- Create: `platform/api/evalforge/schemas/runs.py`
- Create: `platform/api/evalforge/api/runs.py`
- Test: `platform/api/tests/test_api_runs.py`

- [ ] **Step 1: Write the failing tests**

These tests must never make a real network call — they use a fake provider
registered directly against the module's provider lookup, following the
same pattern `tests/test_runner.py` already established.

```python
# tests/test_api_runs.py
import asyncio

import pytest

from evalforge.db.models import CandidateModel, Prompt, PromptVersion, RunStatus, Suite
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_api_runs.py -v`
Expected: FAIL — `evalforge.api.runs` doesn't exist (and `main.py` still
can't import `ratings`/`compare` either — expected until those tasks land;
if this specific test file's failures are all about `runs` specifically,
that confirms things are on track).

- [ ] **Step 3: Write `evalforge/schemas/runs.py`**

```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    suite_id: UUID
    candidates: list[str]
    judges: list[str] = ["exact_match"]
    concurrency: int = Field(default=3, ge=1, le=20)


class RunAccepted(BaseModel):
    run_id: UUID


class RunStatusResponse(BaseModel):
    id: UUID
    status: str
    completed_steps: int
    total_steps: int
    started_at: datetime | None
    finished_at: datetime | None


class JudgeEvaluationResponse(BaseModel):
    judge_name: str
    score: float
    justification: str | None


class ResultResponse(BaseModel):
    id: UUID
    prompt_version_id: UUID
    candidate_model: str
    status: str
    generated_text: str
    error: str | None
    latency_ms: int
    cost_usd: float
    judge_evaluations: list[JudgeEvaluationResponse]


class CostResponse(BaseModel):
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    by_candidate: dict[str, float]
```

- [ ] **Step 4: Write `evalforge/api/runs.py`**

```python
"""POST /runs must never hand its request-scoped session to the background
task — see the module-level note on `_run_in_background` below and
docs/superpowers/specs/2026-07-03-phase3a-fastapi-backend-design.md section
4 for why: the request's AsyncSession is torn down as part of the request/
response lifecycle, and BackgroundTasks execute after the response is sent.
Sharing it produces a closed-session/detached-instance error the first time
the background task touches the ORM objects.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.config import Settings
from evalforge.db.engine import make_engine, make_session_factory
from evalforge.db.models import CandidateModel, JudgeEvaluation, Result, Run, RunStatus, Suite
from evalforge.db.session import get_session
from evalforge.judges import get_judge
from evalforge.providers import get_provider
from evalforge.runner import RunConfig, execute_run
from evalforge.schemas.runs import (
    CostResponse,
    JudgeEvaluationResponse,
    ResultResponse,
    RunAccepted,
    RunCreate,
    RunStatusResponse,
)

router = APIRouter(tags=["runs"])
logger = logging.getLogger(__name__)


def _make_background_session_factory(settings: Settings):
    """Extracted into its own function (rather than inlined in
    _run_in_background) purely so tests have a seam to monkeypatch: SQLite
    `:memory:` databases are unique per engine, so a real call to
    make_engine(settings) in a test would connect to an empty database the
    test's own session/api_client fixtures never touched. Tests monkeypatch
    this exact function to return the shared in-memory session_factory
    fixture instead — see tests/test_api_runs.py."""
    return make_session_factory(make_engine(settings))


async def _run_in_background(run_id: str, candidate_ids: list[str], judge_names: list[str]) -> None:
    """Opens its OWN session — never reuses the request's. Re-fetches the
    Run and CandidateModel rows by id inside that session."""
    settings = Settings()
    factory = _make_background_session_factory(settings)
    async with factory() as session:
        run = await session.get(Run, run_id)
        candidates = [await session.get(CandidateModel, cid) for cid in candidate_ids]
        providers = {c.provider: get_provider(c.provider, settings) for c in candidates}
        judges = [get_judge(name, settings) for name in judge_names]
        config = RunConfig(providers=providers, judges=judges)
        try:
            await execute_run(session, run, candidates, config)
        except Exception:
            # execute_run() already sets status=FAILED and commits before
            # re-raising (see runner.py). This catch exists purely so the
            # failure is visible in the API server's own logs — Starlette
            # would otherwise swallow an exception escaping a BackgroundTask
            # with only an easy-to-miss ASGI-server-level log line.
            logger.exception("run %s failed", run_id)
    # No explicit engine.dispose() here: _make_background_session_factory's
    # engine isn't exposed to this function (by design — see its docstring,
    # this keeps the test monkeypatch seam trivial, since a mocked version
    # only needs to return a factory, not a factory+engine pair to manage).
    # A short-lived per-run engine is GC'd rather than explicitly disposed;
    # an accepted minor tradeoff, not a resource leak in the connection-pool
    # sense (SQLite/aiosqlite hold no persistent server-side connection slot).


@router.post("/runs", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: RunCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> RunAccepted:
    suite = await session.get(Suite, body.suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {body.suite_id} not found")

    candidates: list[CandidateModel] = []
    for spec in body.candidates:
        provider_name, _, model_name = spec.partition(":")
        if not model_name:
            raise HTTPException(
                status_code=400, detail=f"candidate '{spec}' must be provider:model"
            )
        try:
            get_provider(provider_name, Settings())
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown provider '{provider_name}'"
            ) from exc
        candidates.append(CandidateModel(name=model_name, provider=provider_name))
    session.add_all(candidates)

    for name in body.judges:
        try:
            get_judge(name, Settings())
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"unknown judge '{name}'") from exc

    run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=body.concurrency)
    session.add(run)
    await session.flush()

    candidate_ids = [str(c.id) for c in candidates]
    background_tasks.add_task(_run_in_background, str(run.id), candidate_ids, body.judges)

    return RunAccepted(run_id=run.id)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)) -> RunStatusResponse:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return RunStatusResponse(
        id=run.id,
        status=run.status.value,
        completed_steps=run.completed_steps,
        total_steps=run.total_steps,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.get("/runs/{run_id}/results", response_model=list[ResultResponse])
async def get_run_results(
    run_id: str,
    judge_name: str | None = None,
    status_filter: str | None = None,  # query param name is "status_filter", not "status" —
    # avoids shadowing the `status` module imported from fastapi at the top of this file.
    limit: int = 1000,
    session: AsyncSession = Depends(get_session),
) -> list[ResultResponse]:
    if limit > 5000:
        limit = 5000
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    query = select(Result).where(Result.run_id == run_id)
    if status_filter is not None:
        query = query.where(Result.status == status_filter)
    query = query.limit(limit)
    results = (await session.execute(query)).scalars().all()

    responses = []
    for r in results:
        evals_query = select(JudgeEvaluation).where(JudgeEvaluation.result_id == r.id)
        if judge_name is not None:
            evals_query = evals_query.where(JudgeEvaluation.judge_name == judge_name)
        evals = (await session.execute(evals_query)).scalars().all()
        if judge_name is not None and not evals:
            continue
        candidate = await session.get(CandidateModel, r.candidate_model_id)
        responses.append(
            ResultResponse(
                id=r.id,
                prompt_version_id=r.prompt_version_id,
                candidate_model=f"{candidate.provider}:{candidate.name}",
                status=r.status.value,
                generated_text=r.generated_text,
                error=r.error,
                latency_ms=r.latency_ms,
                cost_usd=r.cost_usd,
                judge_evaluations=[
                    JudgeEvaluationResponse(
                        judge_name=e.judge_name, score=e.score, justification=e.justification
                    )
                    for e in evals
                ],
            )
        )
    return responses


@router.get("/runs/{run_id}/costs", response_model=CostResponse)
async def get_run_costs(run_id: str, session: AsyncSession = Depends(get_session)) -> CostResponse:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    results = (
        await session.execute(select(Result).where(Result.run_id == run_id))
    ).scalars().all()

    by_candidate: dict[str, float] = {}
    total_cost = 0.0
    total_in = 0
    total_out = 0
    for r in results:
        candidate = await session.get(CandidateModel, r.candidate_model_id)
        key = f"{candidate.provider}:{candidate.name}"
        by_candidate[key] = by_candidate.get(key, 0.0) + r.cost_usd
        total_cost += r.cost_usd
        total_in += r.input_tokens
        total_out += r.output_tokens

    return CostResponse(
        total_cost_usd=total_cost,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        by_candidate=by_candidate,
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_api_runs.py -v`
Expected: FAIL still at collection — `main.py` imports `ratings` and
`compare`, which don't exist until Tasks 5-6. To unblock testing THIS
task's routes now, temporarily comment out the `ratings` and `compare`
imports/includes in `evalforge/main.py` (both the import line and the two
`app.include_router(...)` lines), run the tests, confirm 6 PASSED, then
**restore those two commented lines exactly as they were** before moving
on — do not leave them commented out, Task 5/6 depend on them being active
again so their own tests can pass against the real app.

Expected after the temporary restore-and-verify cycle: 6 PASSED.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv\Scripts\ruff check evalforge/schemas/runs.py evalforge/api/runs.py tests/test_api_runs.py
.venv\Scripts\mypy evalforge/schemas/runs.py evalforge/api/runs.py
git add platform/api/evalforge/schemas/runs.py platform/api/evalforge/api/runs.py platform/api/tests/test_api_runs.py
git commit -m "feat: add run creation (background execution, own session) and status/results/costs endpoints"
```

---

### Task 5: Ratings endpoint

**Files:**
- Create: `platform/api/evalforge/schemas/ratings.py`
- Create: `platform/api/evalforge/api/ratings.py`
- Test: `platform/api/tests/test_api_ratings.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_ratings.py
from evalforge.db.models import (
    CandidateModel,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)


async def _make_two_results(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    version = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model_a = CandidateModel(name="model-a", provider="fake")
    model_b = CandidateModel(name="model-b", provider="fake")
    run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    result_a = Result(
        run=run, prompt_version=version, candidate_model=model_a,
        status=ResultStatus.OK, generated_text="answer A", latency_ms=10,
    )
    result_b = Result(
        run=run, prompt_version=version, candidate_model=model_b,
        status=ResultStatus.OK, generated_text="answer B", latency_ms=10,
    )
    session.add_all([suite, prompt, version, model_a, model_b, run, result_a, result_b])
    await session.commit()
    return version, result_a, result_b


async def test_create_rating_records_chosen_result(api_client, session):
    version, result_a, result_b = await _make_two_results(session)
    response = await api_client.post(
        "/api/v1/ratings",
        json={
            "prompt_version_id": str(version.id),
            "result_a_id": str(result_a.id),
            "result_b_id": str(result_b.id),
            "chosen_result_id": str(result_a.id),
        },
    )
    assert response.status_code == 201
    assert "id" in response.json()


async def test_create_rating_allows_skip_with_no_choice(api_client, session):
    version, result_a, result_b = await _make_two_results(session)
    response = await api_client.post(
        "/api/v1/ratings",
        json={
            "prompt_version_id": str(version.id),
            "result_a_id": str(result_a.id),
            "result_b_id": str(result_b.id),
            "skipped": True,
        },
    )
    assert response.status_code == 201
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_api_ratings.py -v`
Expected: FAIL — `evalforge.api.ratings` doesn't exist.

- [ ] **Step 3: Write `evalforge/schemas/ratings.py`**

```python
from uuid import UUID

from pydantic import BaseModel


class RatingCreate(BaseModel):
    prompt_version_id: UUID
    result_a_id: UUID
    result_b_id: UUID
    chosen_result_id: UUID | None = None
    skipped: bool = False
    rater_session: str | None = None


class RatingResponse(BaseModel):
    id: UUID
```

- [ ] **Step 4: Write `evalforge/api/ratings.py`**

```python
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import HumanRating
from evalforge.db.session import get_session
from evalforge.schemas.ratings import RatingCreate, RatingResponse

router = APIRouter(tags=["ratings"])


@router.post("/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
async def create_rating(
    body: RatingCreate, session: AsyncSession = Depends(get_session)
) -> RatingResponse:
    rating = HumanRating(
        prompt_version_id=body.prompt_version_id,
        result_a_id=body.result_a_id,
        result_b_id=body.result_b_id,
        chosen_result_id=body.chosen_result_id,
        skipped=body.skipped,
        rater_session=body.rater_session,
    )
    session.add(rating)
    await session.flush()
    return RatingResponse(id=rating.id)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_api_ratings.py -v`
Expected: FAIL still at collection until Task 6's `compare` module exists
(same temporary-comment-out procedure as Task 4, Step 5, this time only for
the `compare` import/include lines — `ratings` is now real). Verify 2
PASSED, then restore the commented lines exactly as before.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv\Scripts\ruff check evalforge/schemas/ratings.py evalforge/api/ratings.py tests/test_api_ratings.py
.venv\Scripts\mypy evalforge/schemas/ratings.py evalforge/api/ratings.py
git add platform/api/evalforge/schemas/ratings.py platform/api/evalforge/api/ratings.py platform/api/tests/test_api_ratings.py
git commit -m "feat: add POST /ratings endpoint"
```

---

### Task 6: Compare endpoint + final wiring

**Files:**
- Create: `platform/api/evalforge/schemas/compare.py`
- Create: `platform/api/evalforge/api/compare.py`
- Modify: `platform/api/evalforge/main.py` (confirm final state, no commented lines)
- Test: `platform/api/tests/test_api_compare.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_compare.py
from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)


async def test_compare_two_runs_matches_by_prompt_version_and_candidate(api_client, session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    version = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="model-a", provider="fake")

    run_a = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    run_b = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)

    result_a = Result(
        run=run_a, prompt_version=version, candidate_model=model,
        status=ResultStatus.OK, generated_text="old answer", latency_ms=10,
    )
    result_b = Result(
        run=run_b, prompt_version=version, candidate_model=model,
        status=ResultStatus.OK, generated_text="new answer", latency_ms=10,
    )
    eval_a = JudgeEvaluation(result=result_a, judge_name="exact_match", score=1.0)
    eval_b = JudgeEvaluation(result=result_b, judge_name="exact_match", score=0.0)

    session.add_all(
        [suite, prompt, version, model, run_a, run_b, result_a, result_b, eval_a, eval_b]
    )
    await session.commit()

    response = await api_client.get(
        f"/api/v1/compare?run_id={run_a.id}&run_id={run_b.id}"
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["run_a_result"]["generated_text"] == "old answer"
    assert row["run_b_result"]["generated_text"] == "new answer"
    assert row["score_delta"]["exact_match"] == -1.0


async def test_compare_requires_exactly_two_run_ids(api_client):
    response = await api_client.get("/api/v1/compare?run_id=00000000-0000-0000-0000-000000000000")
    assert response.status_code == 422


async def test_compare_with_missing_run_returns_404(api_client, session):
    suite = Suite(name="s")
    run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    session.add_all([suite, run])
    await session.commit()

    response = await api_client.get(
        f"/api/v1/compare?run_id={run.id}&run_id=00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv\Scripts\python -m pytest tests/test_api_compare.py -v`
Expected: FAIL — `evalforge.api.compare` doesn't exist.

- [ ] **Step 3: Write `evalforge/schemas/compare.py`**

```python
from uuid import UUID

from pydantic import BaseModel

from evalforge.schemas.runs import ResultResponse, RunStatusResponse


class CompareRow(BaseModel):
    prompt_version_id: UUID
    candidate_model: str
    run_a_result: ResultResponse | None
    run_b_result: ResultResponse | None
    score_delta: dict[str, float]


class CompareResponse(BaseModel):
    run_a: RunStatusResponse
    run_b: RunStatusResponse
    rows: list[CompareRow]
```

- [ ] **Step 4: Write `evalforge/api/compare.py`**

```python
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import CandidateModel, JudgeEvaluation, Result, Run
from evalforge.db.session import get_session
from evalforge.schemas.compare import CompareResponse, CompareRow
from evalforge.schemas.runs import JudgeEvaluationResponse, ResultResponse, RunStatusResponse

router = APIRouter(tags=["compare"])


def _run_status_response(run: Run) -> RunStatusResponse:
    return RunStatusResponse(
        id=run.id,
        status=run.status.value,
        completed_steps=run.completed_steps,
        total_steps=run.total_steps,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


async def _result_response(session: AsyncSession, result: Result) -> ResultResponse:
    candidate = await session.get(CandidateModel, result.candidate_model_id)
    evals = (
        await session.execute(select(JudgeEvaluation).where(JudgeEvaluation.result_id == result.id))
    ).scalars().all()
    return ResultResponse(
        id=result.id,
        prompt_version_id=result.prompt_version_id,
        candidate_model=f"{candidate.provider}:{candidate.name}",
        status=result.status.value,
        generated_text=result.generated_text,
        error=result.error,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        judge_evaluations=[
            JudgeEvaluationResponse(judge_name=e.judge_name, score=e.score, justification=e.justification)
            for e in evals
        ],
    )


@router.get("/compare", response_model=CompareResponse)
async def compare_runs(
    run_id: list[UUID] = Query(..., min_length=2, max_length=2),
    session: AsyncSession = Depends(get_session),
) -> CompareResponse:
    run_a_id, run_b_id = run_id
    run_a = await session.get(Run, run_a_id)
    run_b = await session.get(Run, run_b_id)
    if run_a is None:
        raise HTTPException(status_code=404, detail=f"run {run_a_id} not found")
    if run_b is None:
        raise HTTPException(status_code=404, detail=f"run {run_b_id} not found")

    results_a = (await session.execute(select(Result).where(Result.run_id == run_a_id))).scalars().all()
    results_b = (await session.execute(select(Result).where(Result.run_id == run_b_id))).scalars().all()

    by_key_a = {(r.prompt_version_id, r.candidate_model_id): r for r in results_a}
    by_key_b = {(r.prompt_version_id, r.candidate_model_id): r for r in results_b}
    all_keys = set(by_key_a) | set(by_key_b)

    rows: list[CompareRow] = []
    for prompt_version_id, candidate_model_id in all_keys:
        result_a = by_key_a.get((prompt_version_id, candidate_model_id))
        result_b = by_key_b.get((prompt_version_id, candidate_model_id))
        candidate = await session.get(CandidateModel, candidate_model_id)

        response_a = await _result_response(session, result_a) if result_a else None
        response_b = await _result_response(session, result_b) if result_b else None

        score_delta: dict[str, float] = {}
        if response_a and response_b:
            scores_a = {e.judge_name: e.score for e in response_a.judge_evaluations}
            scores_b = {e.judge_name: e.score for e in response_b.judge_evaluations}
            for judge_name in set(scores_a) & set(scores_b):
                score_delta[judge_name] = scores_b[judge_name] - scores_a[judge_name]

        rows.append(
            CompareRow(
                prompt_version_id=prompt_version_id,
                candidate_model=f"{candidate.provider}:{candidate.name}",
                run_a_result=response_a,
                run_b_result=response_b,
                score_delta=score_delta,
            )
        )

    return CompareResponse(
        run_a=_run_status_response(run_a), run_b=_run_status_response(run_b), rows=rows
    )
```

- [ ] **Step 5: Confirm `evalforge/main.py` has no commented-out lines**

Read the current `evalforge/main.py`. Confirm both
`from evalforge.api import compare, prompts, ratings, runs, suites` and all
five `app.include_router(...)` calls are active (uncommented). If Task 4 or
5's temporary comments were left in place by mistake, restore them now.

- [ ] **Step 6: Run the FULL test suite**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: all tests pass — this includes every pre-existing Phase 1 test
(runner, providers, judges, CLI, models) plus every new API test from
Tasks 2-6. Count should be 49 (Phase 1 baseline) + new API tests.

- [ ] **Step 7: Lint and typecheck the whole package**

```bash
.venv\Scripts\ruff check evalforge tests
.venv\Scripts\mypy evalforge
```
Both must be clean.

- [ ] **Step 8: Commit**

```bash
git add platform/api/evalforge/schemas/compare.py platform/api/evalforge/api/compare.py platform/api/evalforge/main.py platform/api/tests/test_api_compare.py
git commit -m "feat: add GET /compare endpoint and finalize FastAPI app wiring"
```

---

### Task 7: Real end-to-end smoke test — actually run the server

**Files:** none created; this is a manual verification task.

- [ ] **Step 1: Start the real server**

Run (from `platform/api/`): `.venv\Scripts\uvicorn evalforge.main:app --reload --port 8000`
Expected: server starts, logs `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **Step 2: Check the auto-generated OpenAPI docs**

Open `http://127.0.0.1:8000/docs` in a browser (or `curl
http://127.0.0.1:8000/openapi.json` to confirm it returns valid JSON).
Expected: all 8 endpoints listed (POST/GET /suites, POST /suites/{id}/prompts,
POST /prompts/{id}/versions, POST /runs, GET /runs/{id}, GET
/runs/{id}/results, GET /runs/{id}/costs, POST /ratings, GET /compare).

- [ ] **Step 3: Create a suite and run it against real local Ollama**

This uses a real local model (no cost), proving the background-task session
fix from Task 4 actually works outside of mocked tests.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/suites -H "Content-Type: application/json" -d "{\"name\": \"smoke-test\", \"prompts\": [{\"input_text\": \"What is 2+2? Answer with the number only.\", \"expected_output\": \"4\"}]}"
```
Note the returned `id`, then:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/runs -H "Content-Type: application/json" -d "{\"suite_id\": \"<id-from-above>\", \"candidates\": [\"ollama:llama3.2\"], \"judges\": [\"exact_match\"]}"
```
Note the returned `run_id`, wait a few seconds for the background task to
finish, then:
```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>
```
Expected: `"status": "completed"`, `"completed_steps": 1`,
`"total_steps": 1`. This confirms the background task's own-session
pattern genuinely works end-to-end, not just against mocks.

- [ ] **Step 4: Stop the server**

Ctrl+C in the terminal running uvicorn.

## Definition of done (Phase 3a)

- All tests pass (`pytest tests -q`), ruff clean, mypy clean.
- Real end-to-end smoke test against local Ollama confirms the
  background-task session-handling fix works outside of mocks.
- 9 endpoints live and documented at `/docs`.

## Explicitly deferred (do not build in this plan)

- The Next.js dashboard (Phase 3b — separate spec, separate plan).
- Authentication, rate limiting, full offset/cursor pagination (per the
  design's section 9).
