# EvalForge Phase 1: Core Engine Implementation Plan


**Goal:** A working eval engine: define a suite of prompts, run it against any of four providers (Claude/OpenAI/Gemini/Ollama) with bounded concurrency and retries, score outputs with pluggable judges, persist everything to a database, and drive it all from a CLI.

**Architecture:** Async-first Python package. Two plugin interfaces (`Provider`, `Judge`) with single-file implementations. SQLAlchemy 2.0 async ORM with immutable prompt versioning; generation (`results`) and judgment (`judge_evaluations`) are separate tables. An asyncio runner with per-provider semaphores and exponential-backoff retries. No web server in this phase — the CLI exercises everything.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.0 (aiosqlite dev / asyncpg prod), Pydantic v2, httpx (all four providers — uniform adapter pattern, no vendor SDKs), pytest + pytest-asyncio + respx, ruff + mypy.

**Conventions for every task:** run commands from `platform/api/`. All code is fully type-annotated. Commit messages are conventional commits with no AI attribution.

---

## File structure (locked in by this plan)

```
platform/api/
  pyproject.toml
  evalforge/
    __init__.py
    config.py            # settings from env (pydantic-settings)
    db/
      __init__.py
      engine.py          # async engine/session factory
      models.py          # all ORM models
    schemas.py           # Pydantic v2 API-boundary schemas
    providers/
      __init__.py        # Provider protocol + registry + errors
      ollama.py
      anthropic.py
      openai.py
      gemini.py
    judges/
      __init__.py        # Judge protocol + registry
      exact_match.py
      llm_judge.py
    pricing.py           # per-model $/token table + cost calc
    runner.py            # async eval runner
    cli.py               # typer CLI: suite create/run/results
  tests/
    conftest.py
    test_models.py
    test_schemas.py
    test_providers.py
    test_judges.py
    test_pricing.py
    test_runner.py
    test_cli.py
  suites/
    demo.json            # seed suite for the CLI smoke test
```

---

### Task 1: Project scaffold

**Files:**
- Create: `platform/api/pyproject.toml`
- Create: `platform/api/evalforge/__init__.py`
- Create: `platform/api/tests/conftest.py`
- Create: `.gitignore` (repo root)

- [ ] **Step 1: Write `.gitignore`** (repo root)

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.mypy_cache/
.ruff_cache/
.pytest_cache/
*.egg-info/
dist/
.env
*.db
node_modules/
.next/
training/data/raw/
training/checkpoints/
training/runs/
```

- [ ] **Step 2: Write `platform/api/pyproject.toml`**

```toml
[project]
name = "evalforge"
version = "0.1.0"
description = "LLM evaluation platform: multi-provider eval runs with pluggable judges"
requires-python = ">=3.11"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "httpx>=0.27",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.5",
    "mypy>=1.10",
]
postgres = ["asyncpg>=0.29"]

[project.scripts]
evalforge = "evalforge.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["evalforge"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 3: Write `platform/api/evalforge/__init__.py`**

```python
"""EvalForge: multi-provider LLM evaluation engine."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `platform/api/tests/conftest.py`**

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from evalforge.db.models import Base


@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()
```

- [ ] **Step 5: Create venv, install, verify tooling runs**

Run (from `platform/api/`):
```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -c "import evalforge; print(evalforge.__version__)"
```
Expected: `0.1.0` (pytest will fail to collect until models exist — that's fine).

- [ ] **Step 6: Commit**

```bash
git add .gitignore platform/api/pyproject.toml platform/api/evalforge/__init__.py platform/api/tests/conftest.py
git commit -m "chore: scaffold evalforge python package with tooling config"
```

---

### Task 2: Database models (immutable prompt versioning)

**Files:**
- Create: `platform/api/evalforge/db/__init__.py` (empty)
- Create: `platform/api/evalforge/db/models.py`
- Create: `platform/api/evalforge/db/engine.py`
- Test: `platform/api/tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import uuid

from sqlalchemy import select

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


async def test_suite_prompt_version_chain(session):
    suite = Suite(name="qa-basics")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="What is 2+2?", expected_output="4")
    session.add_all([suite, prompt, v1])
    await session.commit()

    fetched = (await session.execute(select(PromptVersion))).scalar_one()
    assert fetched.input_text == "What is 2+2?"
    assert fetched.prompt.suite.name == "qa-basics"


async def test_editing_prompt_appends_new_version(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="old")
    v2 = PromptVersion(prompt=prompt, version_number=2, input_text="new")
    session.add_all([suite, prompt, v1, v2])
    await session.commit()

    versions = (await session.execute(select(PromptVersion).order_by(PromptVersion.version_number))).scalars().all()
    assert [v.input_text for v in versions] == ["old", "new"]
    assert v1.input_text == "old"  # old version untouched


async def test_result_and_judge_evaluations_are_separate(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="llama3.2", provider="ollama")
    run = Run(suite=suite, status=RunStatus.RUNNING, concurrency_limit=3)
    result = Result(
        run=run, prompt_version=v1, candidate_model=model,
        status=ResultStatus.OK, generated_text="a", latency_ms=120,
        input_tokens=10, output_tokens=5, cost_usd=0.0,
    )
    j1 = JudgeEvaluation(result=result, judge_name="exact_match", score=1.0)
    j2 = JudgeEvaluation(result=result, judge_name="llm_judge:claude", score=0.9, justification="matches")
    session.add_all([suite, prompt, v1, model, run, result, j1, j2])
    await session.commit()

    evals = (await session.execute(select(JudgeEvaluation))).scalars().all()
    assert len(evals) == 2
    assert {e.result_id for e in evals} == {result.id}


async def test_failed_result_records_error(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="gpt-4o", provider="openai")
    run = Run(suite=suite, status=RunStatus.RUNNING, concurrency_limit=3)
    result = Result(
        run=run, prompt_version=v1, candidate_model=model,
        status=ResultStatus.FAILED, generated_text="", latency_ms=0,
        input_tokens=0, output_tokens=0, cost_usd=0.0, error="429 rate limited",
    )
    session.add_all([suite, prompt, v1, model, run, result])
    await session.commit()
    fetched = (await session.execute(select(Result))).scalar_one()
    assert fetched.status is ResultStatus.FAILED
    assert "429" in fetched.error


async def test_ids_are_uuids(session):
    suite = Suite(name="s")
    session.add(suite)
    await session.commit()
    assert isinstance(suite.id, uuid.UUID)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: evalforge.db.models`

- [ ] **Step 3: Write `evalforge/db/models.py`**

```python
"""ORM models.

Design invariants:
- Prompt text is immutable: edits append a PromptVersion row; results FK the
  exact version, so cross-run diffs stay valid forever.
- Generation (Result) and judgment (JudgeEvaluation) are separate tables: one
  output can be scored by N judges without duplicating rows.
"""
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampedBase(Base):
    __abstract__ = True
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultStatus(enum.Enum):
    OK = "ok"
    FAILED = "failed"


class Suite(TimestampedBase):
    __tablename__ = "suites"
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    prompts: Mapped[list["Prompt"]] = relationship(back_populates="suite")


class Prompt(TimestampedBase):
    __tablename__ = "prompts"
    suite_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("suites.id", ondelete="CASCADE"))
    suite: Mapped[Suite] = relationship(back_populates="prompts")
    versions: Mapped[list["PromptVersion"]] = relationship(back_populates="prompt")


class PromptVersion(TimestampedBase):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_id", "version_number"),)
    prompt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompts.id", ondelete="CASCADE"))
    version_number: Mapped[int] = mapped_column(Integer)
    input_text: Mapped[str] = mapped_column(Text)
    expected_output: Mapped[str | None] = mapped_column(Text, default=None)
    prompt: Mapped[Prompt] = relationship(back_populates="versions")


class CandidateModel(TimestampedBase):
    __tablename__ = "candidate_models"
    name: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))


class Run(TimestampedBase):
    __tablename__ = "runs"
    suite_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("suites.id"))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.QUEUED)
    concurrency_limit: Mapped[int] = mapped_column(Integer, default=3)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    suite: Mapped[Suite] = relationship()
    results: Mapped[list["Result"]] = relationship(back_populates="run")


class Result(TimestampedBase):
    __tablename__ = "results"
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    prompt_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompt_versions.id"))
    candidate_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidate_models.id"))
    status: Mapped[ResultStatus] = mapped_column(Enum(ResultStatus))
    generated_text: Mapped[str] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    latency_ms: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    run: Mapped[Run] = relationship(back_populates="results")
    prompt_version: Mapped[PromptVersion] = relationship()
    candidate_model: Mapped[CandidateModel] = relationship()
    judge_evaluations: Mapped[list["JudgeEvaluation"]] = relationship(back_populates="result")


class JudgeEvaluation(TimestampedBase):
    __tablename__ = "judge_evaluations"
    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    judge_name: Mapped[str] = mapped_column(String(100))
    score: Mapped[float] = mapped_column(Float)
    justification: Mapped[str | None] = mapped_column(Text, default=None)
    result: Mapped[Result] = relationship(back_populates="judge_evaluations")


class HumanRating(TimestampedBase):
    __tablename__ = "human_ratings"
    prompt_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prompt_versions.id", ondelete="CASCADE")
    )
    result_a_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    result_b_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("results.id", ondelete="CASCADE"))
    chosen_result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("results.id", ondelete="CASCADE"), default=None
    )
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    rater_session: Mapped[str | None] = mapped_column(String(100), default=None)
```

- [ ] **Step 4: Write `evalforge/db/engine.py`**

```python
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
    return create_async_engine(settings.database_url)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

- [ ] **Step 5: Write `evalforge/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALFORGE_", env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///evalforge.db"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
```

- [ ] **Step 6: Create `evalforge/db/__init__.py`** (empty file)

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: 5 PASSED

- [ ] **Step 8: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge platform/api/tests/test_models.py
git commit -m "feat: add ORM models with immutable prompt versioning and split judge evaluations"
```

---

### Task 3: Provider interface + Ollama adapter

**Files:**
- Create: `platform/api/evalforge/providers/__init__.py`
- Create: `platform/api/evalforge/providers/ollama.py`
- Test: `platform/api/tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py
import httpx
import pytest
import respx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError, get_provider
from evalforge.providers.ollama import OllamaProvider

SETTINGS = Settings(
    anthropic_api_key="test-key", openai_api_key="test-key", gemini_api_key="test-key"
)


def test_registry_returns_known_providers():
    assert isinstance(get_provider("ollama", SETTINGS), OllamaProvider)


def test_registry_rejects_unknown_provider():
    with pytest.raises(KeyError):
        get_provider("nonexistent", SETTINGS)


@respx.mock
async def test_ollama_generate_returns_completion():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "The answer is 4.",
                "prompt_eval_count": 12,
                "eval_count": 6,
            },
        )
    )
    provider = OllamaProvider(SETTINGS)
    completion = await provider.generate(model="llama3.2", prompt="What is 2+2?")
    assert isinstance(completion, Completion)
    assert completion.text == "The answer is 4."
    assert completion.input_tokens == 12
    assert completion.output_tokens == 6


@respx.mock
async def test_ollama_http_error_raises_provider_error():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(500, text="boom")
    )
    provider = OllamaProvider(SETTINGS)
    with pytest.raises(ProviderError):
        await provider.generate(model="llama3.2", prompt="q")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: evalforge.providers`

- [ ] **Step 3: Write `evalforge/providers/__init__.py`**

```python
"""Provider plugin interface.

Every provider is a single file implementing `Provider.generate()`. The
registry maps a provider name to its class; adding a fifth provider is one
file plus one registry line.
"""
from dataclasses import dataclass
from typing import Protocol

from evalforge.config import Settings


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """A provider call failed after exhausting its own error handling."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str

    async def generate(self, model: str, prompt: str) -> Completion: ...


def get_provider(name: str, settings: Settings) -> Provider:
    from evalforge.providers.anthropic import AnthropicProvider
    from evalforge.providers.gemini import GeminiProvider
    from evalforge.providers.ollama import OllamaProvider
    from evalforge.providers.openai import OpenAIProvider

    registry: dict[str, type] = {
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    return registry[name](settings)  # type: ignore[no-any-return]
```

(Note: `anthropic.py`, `openai.py`, `gemini.py` are written in Task 4; to keep
Task 3 runnable, create those three files as stubs raising
`NotImplementedError` in this step, replaced in Task 4:)

```python
# evalforge/providers/anthropic.py — TEMPORARY stub, replaced in Task 4
from evalforge.config import Settings
from evalforge.providers import Completion


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, model: str, prompt: str) -> Completion:
        raise NotImplementedError
```

(Same shape for `openai.py` with class `OpenAIProvider`, name `"openai"`, and
`gemini.py` with class `GeminiProvider`, name `"gemini"`.)

- [ ] **Step 4: Write `evalforge/providers/ollama.py`**

```python
import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code >= 500 or exc.response.status_code == 429
                raise ProviderError(
                    f"ollama {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=retryable,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"ollama transport error: {exc}") from exc
        data = response.json()
        return Completion(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_providers.py -v`
Expected: 4 PASSED

- [ ] **Step 6: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge/providers platform/api/tests/test_providers.py
git commit -m "feat: add provider plugin interface with ollama adapter"
```

---

### Task 4: Cloud provider adapters (Anthropic, OpenAI, Gemini)

**Files:**
- Modify: `platform/api/evalforge/providers/anthropic.py` (replace stub)
- Modify: `platform/api/evalforge/providers/openai.py` (replace stub)
- Modify: `platform/api/evalforge/providers/gemini.py` (replace stub)
- Test: `platform/api/tests/test_providers.py` (append)

- [ ] **Step 1: Append failing tests to `tests/test_providers.py`**

```python
from evalforge.providers.anthropic import AnthropicProvider
from evalforge.providers.gemini import GeminiProvider
from evalforge.providers.openai import OpenAIProvider


@respx.mock
async def test_anthropic_generate():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "4"}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )
    )
    provider = AnthropicProvider(SETTINGS)
    completion = await provider.generate(model="claude-sonnet-5", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.input_tokens == 12
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "test-key"


@respx.mock
async def test_anthropic_429_is_retryable():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = AnthropicProvider(SETTINGS)
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(model="claude-sonnet-5", prompt="q")
    assert excinfo.value.retryable is True


@respx.mock
async def test_anthropic_401_is_not_retryable():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    provider = AnthropicProvider(SETTINGS)
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(model="claude-sonnet-5", prompt="q")
    assert excinfo.value.retryable is False


@respx.mock
async def test_openai_generate():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "4"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )
    )
    provider = OpenAIProvider(SETTINGS)
    completion = await provider.generate(model="gpt-4o-mini", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.output_tokens == 3


@respx.mock
async def test_gemini_generate():
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "4"}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
            },
        )
    )
    provider = GeminiProvider(SETTINGS)
    completion = await provider.generate(model="gemini-2.0-flash", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.input_tokens == 12
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `.venv/Scripts/python -m pytest tests/test_providers.py -v`
Expected: 4 previous PASS; 6 new FAIL with `NotImplementedError`

- [ ] **Step 3: Write `evalforge/providers/anthropic.py`** (replace stub)

```python
import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 502, 503, 529}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.anthropic_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"anthropic {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"anthropic transport error: {exc}") from exc
        data = response.json()
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        return Completion(
            text=text,
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
```

- [ ] **Step 4: Write `evalforge/providers/openai.py`** (replace stub)

```python
import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 502, 503}


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openai_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"openai {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"openai transport error: {exc}") from exc
        data = response.json()
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
        )
```

- [ ] **Step 5: Write `evalforge/providers/gemini.py`** (replace stub)

```python
import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 503}


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"gemini {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"gemini transport error: {exc}") from exc
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        usage = data.get("usageMetadata", {})
        return Completion(
            text="".join(p.get("text", "") for p in parts),
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )
```

- [ ] **Step 6: Run all provider tests**

Run: `.venv/Scripts/python -m pytest tests/test_providers.py -v`
Expected: 10 PASSED

- [ ] **Step 7: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge/providers platform/api/tests/test_providers.py
git commit -m "feat: add anthropic, openai, and gemini provider adapters"
```

---

### Task 5: Pricing table + cost calculation

**Files:**
- Create: `platform/api/evalforge/pricing.py`
- Test: `platform/api/tests/test_pricing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pricing.py
from evalforge.pricing import cost_usd


def test_known_model_cost():
    # claude-sonnet-5: $3/M input, $15/M output
    assert cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0) == 3.0
    assert cost_usd("claude-sonnet-5", input_tokens=0, output_tokens=1_000_000) == 15.0


def test_unknown_model_costs_zero():
    assert cost_usd("llama3.2", input_tokens=5000, output_tokens=5000) == 0.0


def test_fractional_cost_rounds_to_six_places():
    cost = cost_usd("gpt-4o-mini", input_tokens=1234, output_tokens=567)
    assert cost == round(cost, 6)
    assert cost > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `evalforge/pricing.py`**

```python
"""Per-model pricing in USD per million tokens.

Unknown models (local Ollama models) cost 0. Prices are a point-in-time
snapshot; the benchmark report records the date they were captured.
"""

# (input $/M, output $/M) — snapshot 2026-07
_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gemini-2.0-flash": (0.1, 0.4),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in _PRICES:
        return 0.0
    input_price, output_price = _PRICES[model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 6)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_pricing.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add platform/api/evalforge/pricing.py platform/api/tests/test_pricing.py
git commit -m "feat: add per-model pricing table and cost calculation"
```

---

### Task 6: Judge interface + exact_match + llm_judge

**Files:**
- Create: `platform/api/evalforge/judges/__init__.py`
- Create: `platform/api/evalforge/judges/exact_match.py`
- Create: `platform/api/evalforge/judges/llm_judge.py`
- Test: `platform/api/tests/test_judges.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_judges.py
import httpx
import pytest
import respx

from evalforge.config import Settings
from evalforge.judges import Judgment, get_judge
from evalforge.judges.exact_match import ExactMatchJudge
from evalforge.judges.llm_judge import LlmJudge

SETTINGS = Settings(anthropic_api_key="test-key")


def test_registry():
    assert isinstance(get_judge("exact_match", SETTINGS), ExactMatchJudge)
    assert isinstance(get_judge("llm_judge", SETTINGS), LlmJudge)
    with pytest.raises(KeyError):
        get_judge("nope", SETTINGS)


async def test_exact_match_scores_one_for_match():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="2+2?", expected="4", output="4")
    assert judgment.score == 1.0


async def test_exact_match_is_case_and_whitespace_insensitive():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected="Paris", output="  paris \n")
    assert judgment.score == 1.0


async def test_exact_match_scores_zero_for_mismatch():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected="4", output="5")
    assert judgment.score == 0.0


async def test_exact_match_without_expected_returns_none():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected=None, output="anything")
    assert judgment is None


@respx.mock
async def test_llm_judge_parses_score_and_justification():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": '{"score": 0.8, "justification": "mostly correct"}'}
                ],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )
    )
    judge = LlmJudge(SETTINGS)
    judgment = await judge.score(prompt="2+2?", expected="4", output="four")
    assert judgment is not None
    assert judgment.score == 0.8
    assert judgment.justification == "mostly correct"


@respx.mock
async def test_llm_judge_malformed_json_raises():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "I think it's pretty good!"}],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )
    )
    judge = LlmJudge(SETTINGS)
    with pytest.raises(ValueError):
        await judge.score(prompt="q", expected="4", output="four")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_judges.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `evalforge/judges/__init__.py`**

```python
"""Judge plugin interface.

A judge scores one (prompt, expected, output) triple. Returning None means
"cannot judge this item" (e.g. exact_match with no expected answer) — the
runner records nothing rather than a fake zero.
"""
from dataclasses import dataclass
from typing import Protocol

from evalforge.config import Settings


@dataclass(frozen=True)
class Judgment:
    score: float  # 0.0 - 1.0
    justification: str | None = None


class Judge(Protocol):
    name: str

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None: ...


def get_judge(name: str, settings: Settings) -> Judge:
    from evalforge.judges.exact_match import ExactMatchJudge
    from evalforge.judges.llm_judge import LlmJudge

    registry: dict[str, type] = {
        "exact_match": ExactMatchJudge,
        "llm_judge": LlmJudge,
    }
    return registry[name](settings)  # type: ignore[no-any-return]
```

- [ ] **Step 4: Write `evalforge/judges/exact_match.py`**

```python
from evalforge.config import Settings
from evalforge.judges import Judgment


class ExactMatchJudge:
    name = "exact_match"

    def __init__(self, settings: Settings) -> None:
        pass

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        if expected is None:
            return None
        match = expected.strip().casefold() == output.strip().casefold()
        return Judgment(score=1.0 if match else 0.0)
```

- [ ] **Step 5: Write `evalforge/judges/llm_judge.py`**

```python
"""LLM-as-judge using Claude.

The judge model returns strict JSON {"score": float, "justification": str}.
Malformed judge output raises ValueError — a judge that silently returns a
default score would corrupt the dataset.
"""
import json

from evalforge.config import Settings
from evalforge.judges import Judgment
from evalforge.providers.anthropic import AnthropicProvider

_JUDGE_MODEL = "claude-sonnet-5"

_PROMPT_TEMPLATE = """You are an evaluation judge. Score the RESPONSE to the QUESTION on a scale of 0.0 to 1.0.
{expected_block}
QUESTION:
{prompt}

RESPONSE:
{output}

Reply with strict JSON only, no markdown fences: {{"score": <float 0-1>, "justification": "<one sentence>"}}"""


class LlmJudge:
    name = "llm_judge"

    def __init__(self, settings: Settings) -> None:
        self._provider = AnthropicProvider(settings)

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        expected_block = f"\nREFERENCE ANSWER:\n{expected}\n" if expected else ""
        judge_prompt = _PROMPT_TEMPLATE.format(
            expected_block=expected_block, prompt=prompt, output=output
        )
        completion = await self._provider.generate(model=_JUDGE_MODEL, prompt=judge_prompt)
        try:
            data = json.loads(completion.text)
            score = float(data["score"])
            justification = str(data["justification"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"judge returned malformed output: {completion.text[:200]}") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"judge score out of range: {score}")
        return Judgment(score=score, justification=justification)
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_judges.py -v`
Expected: 7 PASSED

- [ ] **Step 7: Lint, typecheck, commit**

```bash
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge/judges platform/api/tests/test_judges.py
git commit -m "feat: add judge plugin interface with exact_match and llm_judge"
```

---

### Task 7: Async runner (concurrency, retries, persistence)

**Files:**
- Create: `platform/api/evalforge/runner.py`
- Test: `platform/api/tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

The tests use in-memory fake providers/judges — the runner is tested for
orchestration behavior, not HTTP.

```python
# tests/test_runner.py
import asyncio

import pytest
from sqlalchemy import select

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
from evalforge.judges import Judgment
from evalforge.providers import Completion, ProviderError
from evalforge.runner import RunConfig, execute_run


class FakeProvider:
    name = "fake"

    def __init__(self, fail_times: int = 0, track: list | None = None) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.track = track
        self.active = 0
        self.max_active = 0

    async def generate(self, model: str, prompt: str) -> Completion:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.calls <= self.fail_times:
                raise ProviderError("transient", retryable=True)
            return Completion(text=f"answer to {prompt}", input_tokens=10, output_tokens=5)
        finally:
            self.active -= 1


class FakeJudge:
    name = "fake_judge"

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        return Judgment(score=0.5)


async def make_fixture(session, n_prompts: int = 4):
    suite = Suite(name="s")
    model = CandidateModel(name="fake-model", provider="fake")
    versions = []
    for i in range(n_prompts):
        p = Prompt(suite=suite)
        v = PromptVersion(prompt=p, version_number=1, input_text=f"q{i}", expected_output=f"a{i}")
        versions.append(v)
        session.add_all([p, v])
    run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=2)
    session.add_all([suite, model, run])
    await session.commit()
    return run, model, versions


async def test_run_produces_results_and_evaluations(session):
    run, model, versions = await make_fixture(session)
    provider = FakeProvider()
    config = RunConfig(providers={"fake": provider}, judges=[FakeJudge()], max_retries=2)
    await execute_run(session, run, [model], config)

    results = (await session.execute(select(Result))).scalars().all()
    evals = (await session.execute(select(JudgeEvaluation))).scalars().all()
    assert len(results) == 4
    assert all(r.status is ResultStatus.OK for r in results)
    assert len(evals) == 4
    assert run.status is RunStatus.COMPLETED
    assert run.completed_steps == run.total_steps == 4


async def test_concurrency_is_bounded(session):
    run, model, _ = await make_fixture(session, n_prompts=6)
    provider = FakeProvider()
    config = RunConfig(providers={"fake": provider}, judges=[], max_retries=0)
    await execute_run(session, run, [model], config)
    assert provider.max_active <= run.concurrency_limit


async def test_transient_failure_is_retried(session):
    run, model, _ = await make_fixture(session, n_prompts=1)
    provider = FakeProvider(fail_times=2)
    config = RunConfig(
        providers={"fake": provider}, judges=[], max_retries=3, retry_base_delay=0.0
    )
    await execute_run(session, run, [model], config)
    results = (await session.execute(select(Result))).scalars().all()
    assert results[0].status is ResultStatus.OK
    assert provider.calls == 3  # 2 failures + 1 success


async def test_exhausted_retries_mark_result_failed_not_run(session):
    run, model, _ = await make_fixture(session, n_prompts=2)
    provider = FakeProvider(fail_times=100)
    config = RunConfig(
        providers={"fake": provider}, judges=[], max_retries=1, retry_base_delay=0.0
    )
    await execute_run(session, run, [model], config)
    results = (await session.execute(select(Result))).scalars().all()
    assert all(r.status is ResultStatus.FAILED for r in results)
    assert all("transient" in (r.error or "") for r in results)
    assert run.status is RunStatus.COMPLETED  # partial failure ≠ run failure
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: evalforge.runner`

- [ ] **Step 3: Write `evalforge/runner.py`**

```python
"""Async eval runner.

Design (ADR-001): plain asyncio, no Celery/Redis. An eval run is I/O-bound
fan-out over provider APIs; a semaphore per run bounds concurrency, and the
job state lives in the runs table. Per-result failures never fail the whole
run — partial data is still data.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)
from evalforge.judges import Judge
from evalforge.pricing import cost_usd
from evalforge.providers import Provider, ProviderError


@dataclass
class RunConfig:
    providers: dict[str, Provider]
    judges: list[Judge] = field(default_factory=list)
    max_retries: int = 3
    retry_base_delay: float = 1.0


async def _latest_versions(session: AsyncSession, suite_id: object) -> list[PromptVersion]:
    """Latest version of each prompt in the suite."""
    rows = (
        await session.execute(
            select(PromptVersion)
            .join(PromptVersion.prompt)
            .where(PromptVersion.prompt.has(suite_id=suite_id))
            .order_by(PromptVersion.prompt_id, PromptVersion.version_number.desc())
        )
    ).scalars().all()
    latest: dict[object, PromptVersion] = {}
    for v in rows:
        latest.setdefault(v.prompt_id, v)
    return list(latest.values())


async def _generate_with_retries(
    provider: Provider, model: str, prompt: str, config: RunConfig
) -> tuple[str, int, int, int, str | None]:
    """Returns (text, input_tokens, output_tokens, latency_ms, error)."""
    attempt = 0
    while True:
        start = asyncio.get_event_loop().time()
        try:
            completion = await provider.generate(model=model, prompt=prompt)
            latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
            return completion.text, completion.input_tokens, completion.output_tokens, latency_ms, None
        except ProviderError as exc:
            if not exc.retryable or attempt >= config.max_retries:
                return "", 0, 0, 0, str(exc)
            await asyncio.sleep(config.retry_base_delay * (2**attempt))
            attempt += 1


async def _process_one(
    session: AsyncSession,
    run: Run,
    version: PromptVersion,
    candidate: CandidateModel,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
) -> None:
    async with semaphore:
        provider = config.providers[candidate.provider]
        text, tokens_in, tokens_out, latency_ms, error = await _generate_with_retries(
            provider, candidate.name, version.input_text, config
        )

    result = Result(
        run=run,
        prompt_version=version,
        candidate_model=candidate,
        status=ResultStatus.FAILED if error else ResultStatus.OK,
        generated_text=text,
        error=error,
        latency_ms=latency_ms,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost_usd=cost_usd(candidate.name, tokens_in, tokens_out),
    )

    evaluations: list[JudgeEvaluation] = []
    if not error:
        for judge in config.judges:
            judgment = await judge.score(
                prompt=version.input_text,
                expected=version.expected_output,
                output=text,
            )
            if judgment is not None:
                evaluations.append(
                    JudgeEvaluation(
                        result=result,
                        judge_name=judge.name,
                        score=judgment.score,
                        justification=judgment.justification,
                    )
                )

    async with lock:  # AsyncSession is not concurrency-safe
        session.add(result)
        session.add_all(evaluations)
        run.completed_steps += 1
        await session.commit()


async def execute_run(
    session: AsyncSession, run: Run, candidates: list[CandidateModel], config: RunConfig
) -> None:
    versions = await _latest_versions(session, run.suite_id)
    run.total_steps = len(versions) * len(candidates)
    run.status = RunStatus.RUNNING
    run.started_at = datetime.now(timezone.utc)
    await session.commit()

    semaphore = asyncio.Semaphore(run.concurrency_limit)
    lock = asyncio.Lock()
    tasks = [
        _process_one(session, run, version, candidate, config, semaphore, lock)
        for version in versions
        for candidate in candidates
    ]
    try:
        await asyncio.gather(*tasks)
        run.status = RunStatus.COMPLETED
    except Exception:
        run.status = RunStatus.FAILED
        raise
    finally:
        run.finished_at = datetime.now(timezone.utc)
        await session.commit()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_runner.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Run the whole suite, lint, typecheck, commit**

```bash
.venv/Scripts/python -m pytest -v
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge/runner.py platform/api/tests/test_runner.py
git commit -m "feat: add async eval runner with bounded concurrency and retries"
```

---

### Task 8: CLI

**Files:**
- Create: `platform/api/evalforge/cli.py`
- Create: `platform/api/suites/demo.json`
- Test: `platform/api/tests/test_cli.py`

- [ ] **Step 1: Write `suites/demo.json`**

```json
{
  "name": "demo-qa",
  "description": "Tiny factual QA suite for smoke-testing the engine",
  "prompts": [
    {"input_text": "What is the capital of France? Answer with the city name only.", "expected_output": "Paris"},
    {"input_text": "What is 7 * 8? Answer with the number only.", "expected_output": "56"},
    {"input_text": "What year did the Apollo 11 moon landing happen? Answer with the year only.", "expected_output": "1969"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_cli.py
import json

from typer.testing import CliRunner

from evalforge.cli import app

runner = CliRunner()


def test_suite_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "name": "cli-test",
                "prompts": [{"input_text": "q1", "expected_output": "a1"}],
            }
        )
    )
    result = runner.invoke(app, ["suite", "create", str(suite_file)])
    assert result.exit_code == 0
    assert "cli-test" in result.output

    result = runner.invoke(app, ["suite", "list"])
    assert result.exit_code == 0
    assert "cli-test" in result.output


def test_suite_create_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    result = runner.invoke(app, ["suite", "create", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: evalforge.cli`

- [ ] **Step 4: Write `evalforge/cli.py`**

```python
"""CLI entry point: create suites, run evals, inspect results."""
import asyncio
import json
from pathlib import Path

import typer
from sqlalchemy import select

from evalforge.config import Settings
from evalforge.db.engine import init_db, make_engine, make_session_factory
from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    Prompt,
    PromptVersion,
    Result,
    Run,
    RunStatus,
    Suite,
)
from evalforge.judges import get_judge
from evalforge.providers import get_provider
from evalforge.runner import RunConfig, execute_run

app = typer.Typer(help="EvalForge: multi-provider LLM evaluation engine")
suite_app = typer.Typer(help="Manage eval suites")
app.add_typer(suite_app, name="suite")


def _run(coro):  # bridge typer's sync world to asyncio
    return asyncio.run(coro)


@suite_app.command("create")
def suite_create(path: Path) -> None:
    """Create a suite from a JSON file."""
    if not path.exists():
        typer.echo(f"error: {path} not found", err=True)
        raise typer.Exit(code=1)
    spec = json.loads(path.read_text(encoding="utf-8"))

    async def _create() -> None:
        settings = Settings()
        engine = make_engine(settings)
        await init_db(engine)
        factory = make_session_factory(engine)
        async with factory() as session:
            suite = Suite(name=spec["name"], description=spec.get("description"))
            session.add(suite)
            for item in spec["prompts"]:
                prompt = Prompt(suite=suite)
                session.add(prompt)
                session.add(
                    PromptVersion(
                        prompt=prompt,
                        version_number=1,
                        input_text=item["input_text"],
                        expected_output=item.get("expected_output"),
                    )
                )
            await session.commit()
            typer.echo(f"created suite '{suite.name}' ({suite.id}) with {len(spec['prompts'])} prompts")
        await engine.dispose()

    _run(_create())


@suite_app.command("list")
def suite_list() -> None:
    """List all suites."""

    async def _list() -> None:
        settings = Settings()
        engine = make_engine(settings)
        await init_db(engine)
        factory = make_session_factory(engine)
        async with factory() as session:
            suites = (await session.execute(select(Suite))).scalars().all()
            for s in suites:
                typer.echo(f"{s.id}  {s.name}")
        await engine.dispose()

    _run(_list())


@app.command("run")
def run_eval(
    suite_name: str,
    candidate: list[str] = typer.Option(
        ..., "--candidate", "-c", help="provider:model, e.g. ollama:llama3.2"
    ),
    judge: list[str] = typer.Option(
        ["exact_match"], "--judge", "-j", help="judge name, e.g. exact_match, llm_judge"
    ),
    concurrency: int = typer.Option(3, "--concurrency"),
) -> None:
    """Run a suite against candidate models."""

    async def _execute() -> None:
        settings = Settings()
        engine = make_engine(settings)
        await init_db(engine)
        factory = make_session_factory(engine)
        async with factory() as session:
            suite = (
                await session.execute(select(Suite).where(Suite.name == suite_name))
            ).scalar_one_or_none()
            if suite is None:
                typer.echo(f"error: suite '{suite_name}' not found", err=True)
                raise typer.Exit(code=1)

            candidates: list[CandidateModel] = []
            providers = {}
            for spec in candidate:
                provider_name, _, model_name = spec.partition(":")
                if not model_name:
                    typer.echo(f"error: candidate '{spec}' must be provider:model", err=True)
                    raise typer.Exit(code=1)
                providers[provider_name] = get_provider(provider_name, settings)
                candidates.append(CandidateModel(name=model_name, provider=provider_name))
            session.add_all(candidates)

            run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=concurrency)
            session.add(run)
            await session.commit()

            config = RunConfig(
                providers=providers,
                judges=[get_judge(name, settings) for name in judge],
            )
            typer.echo(f"run {run.id} started ({len(candidate)} candidates)")
            await execute_run(session, run, candidates, config)
            typer.echo(
                f"run {run.id} {run.status.value}: {run.completed_steps}/{run.total_steps} steps"
            )
        await engine.dispose()

    _run(_execute())


@app.command("results")
def show_results(run_id: str) -> None:
    """Show scored results for a run."""

    async def _show() -> None:
        settings = Settings()
        engine = make_engine(settings)
        factory = make_session_factory(engine)
        async with factory() as session:
            results = (
                await session.execute(select(Result).where(Result.run_id == run_id))
            ).scalars().all()
            for r in results:
                evals = (
                    await session.execute(
                        select(JudgeEvaluation).where(JudgeEvaluation.result_id == r.id)
                    )
                ).scalars().all()
                scores = ", ".join(f"{e.judge_name}={e.score:.2f}" for e in evals)
                status = r.status.value
                typer.echo(f"[{status}] {r.generated_text[:60]!r}  {scores}  ${r.cost_usd:.4f}")
        await engine.dispose()

    _run(_show())


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Full suite, lint, typecheck, commit**

```bash
.venv/Scripts/python -m pytest -v
.venv/Scripts/ruff check evalforge tests && .venv/Scripts/mypy evalforge
git add platform/api/evalforge/cli.py platform/api/suites/demo.json platform/api/tests/test_cli.py
git commit -m "feat: add CLI for suite management and eval runs"
```

---

### Task 9: End-to-end smoke test against local Ollama

**Files:**
- Create: `docs/adr/ADR-001-asyncio-not-celery.md`
- Create: `docs/adr/ADR-002-httpx-not-vendor-sdks.md`

- [ ] **Step 1: Verify Ollama is up and a model is pulled**

Run: `ollama list`
Expected: `llama3.2` (or similar) present. If not: `ollama pull llama3.2`.

- [ ] **Step 2: Run the demo suite end to end**

Run (from `platform/api/`):
```bash
.venv/Scripts/evalforge suite create suites/demo.json
.venv/Scripts/evalforge run demo-qa --candidate ollama:llama3.2 --judge exact_match
```
Expected: `run <uuid> completed: 3/3 steps`, then `evalforge results <uuid>`
shows three results with `exact_match` scores. (Small local models may miss
one — a sub-1.0 average is fine; failures to *execute* are not.)

- [ ] **Step 3: Write `docs/adr/ADR-001-asyncio-not-celery.md`**

```markdown
# ADR-001: asyncio task execution, not Celery/Redis

## Status
Accepted (2026-07)

## Context
Eval runs are long-running jobs: N prompts x M candidate models, each an HTTP
call to a provider, plus judge scoring. A common reflex is Celery + Redis.

## Decision
Plain asyncio: a semaphore bounds per-run concurrency, run state persists in
the `runs` table, and per-result failures are recorded without failing the run.

## Rationale
The workload is I/O-bound fan-out, not CPU work needing worker processes. A
single API process comfortably drives hundreds of concurrent HTTP calls.
Celery would add a broker, a result backend, worker deployment, and serialization
constraints — none of which buy anything at this scale.

## Revisit when
Multiple API replicas need to share a work queue, or runs must survive process
restarts mid-flight.
```

- [ ] **Step 4: Write `docs/adr/ADR-002-httpx-not-vendor-sdks.md`**

```markdown
# ADR-002: Direct httpx calls, not vendor SDKs

## Status
Accepted (2026-07)

## Context
Each provider (Anthropic, OpenAI, Google, Ollama) ships a Python SDK.

## Decision
All four adapters call the HTTP APIs directly with httpx.

## Rationale
The adapters need exactly one operation (generate a completion) behind a
shared `Provider` protocol. Four SDKs means four dependency trees, four retry
behaviors, and four exception hierarchies to normalize anyway. Direct HTTP
keeps the adapters symmetric (~50 lines each), makes error taxonomy explicit
(retryable vs not, per status code), and tests mock uniformly with respx.

## Revisit when
We need streaming, tool use, or other provider features where SDK ergonomics
start paying for their weight.
```

- [ ] **Step 5: Commit**

```bash
git add docs/adr
git commit -m "docs: add ADR-001 (asyncio over celery) and ADR-002 (httpx over sdks)"
```

---

## Definition of done (Phase 1)

- `pytest` green (~25 tests), `ruff` clean, `mypy --strict` clean.
- `evalforge suite create` + `evalforge run` + `evalforge results` work end to
  end against local Ollama with zero API keys.
- Cloud adapters fully implemented and unit-tested (respx), exercised for real
  in the Phase 4 benchmark.
- Commit history: ~9 focused conventional commits, no AI attribution.

## Explicitly deferred (do not build in Phase 1)

- FastAPI HTTP server (Phase 3 — the runner/API split is already clean)
- DeBERTa judge (Phase 2 — plugs into the `Judge` registry)
- similarity judge (Phase 2, ships with the embedding infra)
- Alembic migrations (first post-release schema change; noted in engine.py)
- SSE, dashboard, Docker, CI workflows, Terraform (Phases 3–4)
