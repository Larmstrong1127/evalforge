# EvalForge Phase 3a — FastAPI HTTP API Design

Date: 2026-07-03
Status: Approved
Parent spec: `docs/superpowers/specs/2026-07-02-evalforge-design.md` (sections 5, 11)

## 1. Scope

Wraps the existing, already-tested runner/DB/CLI logic (`evalforge.runner`,
`evalforge.db.models`, `evalforge.providers`, `evalforge.judges`) in a real
HTTP API. No new business logic — this phase is entirely about exposing what
already exists over HTTP, plus the request/response schemas and execution
wiring needed to do that correctly. The Next.js dashboard that consumes this
API is an explicitly separate, later spec (Phase 3b) — it does not exist yet
and nothing in this phase depends on it.

## 2. File structure

```
platform/api/evalforge/
  main.py                 # FastAPI app: CORS, router mounting, DI wiring
  db/
    session.py             # NEW: get_session() FastAPI dependency (per-request AsyncSession)
  api/
    __init__.py
    suites.py              # POST /suites, POST /suites/{id}/prompts, GET /suites
    runs.py                 # POST /runs, GET /runs/{id}, GET /runs/{id}/results, GET /runs/{id}/costs
    ratings.py                # POST /ratings
    compare.py                 # GET /compare
  schemas/
    __init__.py
    suites.py                # PromptCreate, SuiteCreate, SuiteResponse
    runs.py                    # RunCreate, RunStatusResponse, ResultResponse, JudgeEvaluationResponse, CostResponse
    ratings.py                   # RatingCreate, RatingResponse
    compare.py                     # CompareResponse, CompareRow
```

`GET /suites` is added beyond the parent spec's list — the dashboard's runs
list and suite picker need a way to enumerate suites, and the CLI already
has this exact capability (`suite list`); exposing it over HTTP is a
one-line addition, not scope creep.

## 3. Database session handling

`db/session.py` provides `get_session() -> AsyncGenerator[AsyncSession, None]`,
a FastAPI dependency yielding one `AsyncSession` per request from the
existing `make_session_factory()`, committing on clean exit and rolling back
on exception — the standard FastAPI+SQLAlchemy pattern. This is new: the CLI
creates one session per command invocation, not per-request; the dependency
just wraps the same `make_session_factory()` in a request-scoped generator.

## 4. Execution model

`POST /runs`:
1. Validates the request body (Pydantic).
2. Looks up the suite; 404 if not found.
3. Parses `candidate` specs (`provider:model`); 400 if malformed or provider
   unknown (same validation the CLI already does, now returning HTTP status
   instead of exit codes).
4. Creates `CandidateModel` rows, a `Run` row (status=QUEUED), commits.
5. Returns `202 Accepted` with `{"run_id": ...}` immediately.
6. Hands `execute_run(session, run, candidates, config)` to FastAPI's
   `BackgroundTasks`, which runs it as an asyncio task after the response is
   sent — no new infrastructure, consistent with ADR-001 (asyncio, not
   Celery). If the server process restarts mid-run, that run is orphaned in
   RUNNING state — an already-accepted v1 limitation (see `runner.py`'s own
   docstring on interrupted runs not being resumed).

## 5. Request/response schemas

All fields map directly onto the existing ORM models (no new business
fields invented for this layer).

**`POST /api/v1/suites`**
```python
class PromptCreate(BaseModel):
    input_text: str
    expected_output: str | None = None

class SuiteCreate(BaseModel):
    name: str
    description: str | None = None
    prompts: list[PromptCreate]

class SuiteResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    prompt_count: int
```

**`POST /api/v1/suites/{id}/prompts`** — body: single `PromptCreate`;
appends a new `PromptVersion` (version_number = max existing + 1) to an
existing `Prompt`, or creates a new `Prompt` if `prompt_id` isn't supplied.
Response: the created version as `{"prompt_id": UUID, "version_number": int}`.

**`GET /api/v1/suites`** — response: `list[SuiteResponse]`.

**`POST /api/v1/runs`**
```python
class RunCreate(BaseModel):
    suite_id: UUID
    candidates: list[str]              # ["ollama:llama3.2", "anthropic:claude-sonnet-5"]
    judges: list[str] = ["exact_match"]
    concurrency: int = Field(default=3, ge=1, le=20)

class RunAccepted(BaseModel):
    run_id: UUID
```

**`GET /api/v1/runs/{id}`**
```python
class RunStatusResponse(BaseModel):
    id: UUID
    status: str                        # RunStatus.value
    completed_steps: int
    total_steps: int
    started_at: datetime | None
    finished_at: datetime | None
```
404 if the run doesn't exist.

**`GET /api/v1/runs/{id}/results`** (optional query params: `judge_name`,
`status` for filtering — direct WHERE clauses, no new logic)
```python
class JudgeEvaluationResponse(BaseModel):
    judge_name: str
    score: float
    justification: str | None

class ResultResponse(BaseModel):
    id: UUID
    prompt_version_id: UUID
    candidate_model: str                # "provider:name"
    status: str
    generated_text: str
    error: str | None
    latency_ms: int
    cost_usd: float
    judge_evaluations: list[JudgeEvaluationResponse]
```
Response: `list[ResultResponse]`.

**`GET /api/v1/runs/{id}/costs`**
```python
class CostResponse(BaseModel):
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    by_candidate: dict[str, float]      # "provider:name" -> cost_usd
```
Computed by aggregating `results` for the run — no new stored data.

**`POST /api/v1/ratings`**
```python
class RatingCreate(BaseModel):
    prompt_version_id: UUID
    result_a_id: UUID
    result_b_id: UUID
    chosen_result_id: UUID | None = None   # None + skipped=True for a skip
    skipped: bool = False
    rater_session: str | None = None

class RatingResponse(BaseModel):
    id: UUID
```
This endpoint only records a vote — pair *selection* (which two results to
show next) is dashboard-side logic against `GET /runs/{id}/results`, not a
backend concern; nothing here presumes how the dashboard picks pairs.

**`GET /api/v1/compare?runs={a},{b}`**
```python
class CompareRow(BaseModel):
    prompt_version_id: UUID
    candidate_model: str
    run_a_result: ResultResponse | None    # None if this run has no result for this (prompt_version, candidate) pair
    run_b_result: ResultResponse | None
    score_delta: dict[str, float]          # judge_name -> (run_b score - run_a score), only where both have that judge

class CompareResponse(BaseModel):
    run_a: RunStatusResponse
    run_b: RunStatusResponse
    rows: list[CompareRow]
```
Rows are built by joining each run's results on
`(prompt_version_id, candidate_model_id)` — per the parent spec's note, this
is a **run-vs-run** diff (did a model/prompt change regress outputs?), not a
prompt-version-vs-prompt-version diff. 404 if either run id doesn't exist.

## 6. CORS

`CORSMiddleware` in `main.py` allowing `http://localhost:3000` and
`http://127.0.0.1:3000` (default Next.js dev ports), all methods, all
headers — configured now per your call, so the not-yet-built dashboard
doesn't hit a CORS wall on day one.

## 7. Error handling

- 404: unknown suite id, run id, or prompt id.
- 400: malformed `candidate` spec (missing `:`), unknown provider/judge name.
- 422: Pydantic validation failure (FastAPI's default, unchanged).
- 500: unexpected errors are not caught specially — they surface as FastAPI's
  default 500 with the exception logged; no new error-handling philosophy
  beyond what `runner.py` already guarantees (a run's *internal* per-item
  failures never crash the run itself; this section is about the HTTP layer
  around it).

## 8. Testing

`httpx.AsyncClient(app=app, base_url="http://test")` against the FastAPI app
in-process — no real server process needed. Test database: same
`sqlite+aiosqlite:///:memory:` pattern already used in `tests/conftest.py`,
wired as a `get_session` override via FastAPI's dependency-override
mechanism (`app.dependency_overrides[get_session] = ...`), not real
environment config. `POST /runs` tests inject fake providers/judges the same
way `test_runner.py` already does, so no test makes a real network call or
starts a real background run against a real provider.

## 9. Explicitly deferred

- The Next.js dashboard (Phase 3b — separate spec, separate implementation
  plan, built only after this API exists and is tested).
- Authentication/authorization (no login system exists anywhere in EvalForge
  yet; this is a local-first tool, not scoped for this phase).
- Rate limiting / pagination on list endpoints (acceptable for a
  local/demo-scale tool; a real production deployment would need this, but
  it's not part of the portfolio scope here).
