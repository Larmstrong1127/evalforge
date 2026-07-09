# EvalForge Phase 4: Publish & Polish Implementation Plan


**Goal:** Take the finished three-package project from "works on my machine"
to a publishable portfolio repository: README, LICENSE, CI, Docker Compose
demo, two ADRs, hygiene cleanup, then push to GitHub.

**Architecture:** No product code changes except an opt-in demo seed hook.
All work is packaging: Dockerfiles + compose with Postgres (exercising the
asyncpg path for the first time), three GitHub Actions workflows including
an eval-gate that runs the real engine against a small pinned Ollama model,
and documentation.

**Tech Stack:** Docker/Compose, GitHub Actions, Postgres 16, MIT license.

**Conventions:** Conventional commits, no AI attribution. Backend commands
from `platform/api/`, frontend from `platform/dashboard/`.

---

## File structure (locked in by this plan)

```
LICENSE                                   # NEW: MIT
README.md                                 # NEW: root front door
docker-compose.yml                        # NEW
.github/workflows/ci.yml                  # NEW
.github/workflows/train-check.yml         # NEW
.github/workflows/eval-gate.yml           # NEW
.github/scripts/eval_gate.py              # NEW
.github/eval-baseline.json                # NEW
docs/design/                              # RENAMED from docs/design/
docs/adr/ADR-003-commit-before-background-task.md   # NEW
docs/adr/ADR-004-judge-plugin-optional-extras.md    # NEW
docs/images/                              # NEW: README screenshots
platform/api/Dockerfile                   # NEW
platform/api/README.md                    # NEW (short)
platform/api/scripts/seed_demo.py         # NEW
platform/dashboard/Dockerfile             # NEW
platform/dashboard/README.md              # REWRITTEN (boilerplate removed)
platform/dashboard/CLAUDE.md              # DELETED
platform/dashboard/AGENTS.md              # DELETED
```

---

### Task 1: Repo hygiene + LICENSE

- [ ] Delete `platform/dashboard/CLAUDE.md` and `platform/dashboard/AGENTS.md`.
- [ ] Replace `platform/dashboard/README.md` with ~15 lines: what it is
      (Next.js dashboard for EvalForge), dev commands (`npm run dev/test/lint`),
      pointer to root README.
- [ ] Add `platform/api/README.md` (~15 lines): what it is, venv install,
      `pytest`/`ruff`/`mypy`, `uvicorn evalforge.main:app`, pointer to root.
- [ ] `git mv docs/superpowers docs/design`; update the handful of in-repo
      references to `docs/design/...` paths (grep and fix — they appear
      in module docstrings in `runs.py` and in the docs themselves).
      line from each file in `docs/design/plans/` (content otherwise unchanged).
- [ ] Add `LICENSE` — MIT, `Copyright (c) 2026 Landon Armstrong`.
- [ ] Commit: `chore: repo hygiene for publication (LICENSE, sub-READMEs, docs rename)`

### Task 2: ADR-003 and ADR-004

- [ ] `docs/adr/ADR-003-commit-before-background-task.md` — context (FastAPI
      BackgroundTasks run inside the ASGI layer before the get_session
      dependency's post-yield commit), decision (explicit `commit()` in
      `create_run` before scheduling), consequences, and why mocked tests
      missed it (shared session factory). Source material: the comment block
      in `platform/api/evalforge/api/runs.py` and commit `8263562`.
- [ ] `docs/adr/ADR-004-judge-plugin-optional-extras.md` — context, decision
      (Judge protocol, `None` = cannot-judge, `deberta` optional extra with
      lazy import in `get_judge`), consequences. Source: `judges/__init__.py`,
      `deberta_judge.py`, `pyproject.toml`.
- [ ] Match ADR-001/002's exact format (read them first).
- [ ] Commit: `docs: add ADR-003 (commit-before-background-task) and ADR-004 (judge plugin extras)`

### Task 3: Docker Compose demo

- [ ] `platform/api/Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY evalforge ./evalforge
COPY scripts ./scripts
RUN pip install --no-cache-dir ".[postgres]"
EXPOSE 8000
CMD ["uvicorn", "evalforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] `platform/dashboard/Dockerfile` (NEXT_PUBLIC_ vars are inlined at
      build time — must be a build arg, not runtime env):

```dockerfile
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

      Requires `output: "standalone"` added to `next.config.ts`. If the
      standalone output layout differs on the installed Next 16, adapt the
      COPY lines to what `npm run build` actually produces.

- [ ] Root `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: evalforge
      POSTGRES_PASSWORD: evalforge
      POSTGRES_DB: evalforge
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U evalforge"]
      interval: 2s
      timeout: 2s
      retries: 15
  api:
    build: ./platform/api
    environment:
      EVALFORGE_DATABASE_URL: postgresql+asyncpg://evalforge:evalforge@postgres:5432/evalforge
      EVALFORGE_SEED_DEMO: "1"
      EVALFORGE_OLLAMA_BASE_URL: http://host.docker.internal:11434
    ports: ["8000:8000"]
    depends_on:
      postgres:
        condition: service_healthy
  web:
    build:
      context: ./platform/dashboard
      args:
        NEXT_PUBLIC_API_BASE_URL: http://localhost:8000
    ports: ["3000:3000"]
    depends_on: [api]
```

- [ ] `platform/api/scripts/seed_demo.py` — idempotent (skips if a suite
      named `demo-qa` exists): one suite with 3 prompts, two
      CandidateModels (`demo:model-a`, `demo:model-b`), one COMPLETED Run,
      six OK Results with plausible generated_text/latency/costs, and one
      exact_match JudgeEvaluation per result. Pure ORM writes, no providers.
      Wire into `main.py`'s lifespan: after `init_db`, if
      `EVALFORGE_SEED_DEMO=1` (read via `os.environ`, not Settings — it's a
      demo hook, not app config), call the seed function.
- [ ] Verify: `docker compose up --build` from repo root; suites list,
      run detail, rating room, and compare all render seeded data at
      `localhost:3000` with no API keys.
- [ ] Commit: `feat: add Docker Compose demo with seeded Postgres-backed data`

### Task 4: ci.yml + train-check.yml

- [ ] `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push: {branches: [master]}
  pull_request:
jobs:
  api:
    runs-on: ubuntu-latest
    defaults: {run: {working-directory: platform/api}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e ".[dev]"
      - run: ruff check evalforge tests
      - run: mypy evalforge
      - run: pytest tests -q
  training:
    runs-on: ubuntu-latest
    defaults: {run: {working-directory: training}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e ".[dev]"
      - run: ruff check training tests
      - run: mypy training
      - run: pytest tests -q
  dashboard:
    runs-on: ubuntu-latest
    defaults: {run: {working-directory: platform/dashboard}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: 22, cache: npm, cache-dependency-path: platform/dashboard/package-lock.json}
      - run: npm ci
      - run: npm run lint
      - run: npm run build
      - run: npm run test
```

      Adapt package/dir names for the training job to the real
      `training/pyproject.toml` layout (check its package name and test
      paths before writing). CPU-only torch: if `pip install -e ".[dev]"`
      in training pulls the full CUDA torch (~2.5GB), add
      `pip install torch --index-url https://download.pytorch.org/whl/cpu`
      FIRST so the resolver keeps the CPU wheel.
- [ ] `.github/workflows/train-check.yml` — on PR with `paths: [training/**]`,
      same steps as the `training` job above (it exists as a separate named
      check so training changes get an explicit gate even when ci.yml
      filters change later; duplication is acceptable and documented).
- [ ] Verify locally by running each command sequence manually.
- [ ] Commit: `ci: add lint/type/test workflows for api, training, dashboard`

### Task 5: eval-gate.yml (dogfood gate)

- [ ] `.github/scripts/eval_gate.py` — standalone script using the evalforge
      package directly (temp SQLite DB): builds a 5-prompt deterministic
      suite (single-token answers: arithmetic, capitals), runs
      `execute_run` with the real Ollama provider (`llama3.2:1b`) and
      `exact_match` judge, computes mean score, loads
      `.github/eval-baseline.json` (`{"model": "llama3.2:1b", "baseline": <float>, "min_allowed": <float>}`),
      prints a per-prompt table to stdout and appends a summary to
      `$GITHUB_STEP_SUMMARY` if set, exits 1 if mean < min_allowed.
- [ ] `.github/workflows/eval-gate.yml`:

```yaml
name: Eval Gate
on:
  pull_request:
    paths: ["platform/**", ".github/workflows/eval-gate.yml", ".github/scripts/**"]
jobs:
  eval-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: curl -fsSL https://ollama.com/install.sh | sh
      - run: ollama serve & sleep 3 && ollama pull llama3.2:1b
      - run: pip install -e "platform/api[dev]"
      - run: python .github/scripts/eval_gate.py
```

- [ ] Baseline: run the script locally against `llama3.2:1b` (pull it first)
      to measure the real score; set `baseline` to that measurement and
      `min_allowed` generously below it (e.g. baseline 0.8 → min 0.4) —
      the gate catches collapses, not jitter.
- [ ] Verify: local run of the script passes and exits 0; force a fake
      regression (temporarily set min_allowed above baseline) and confirm
      exit 1, then restore.
- [ ] Commit: `ci: add eval-gate workflow — the platform gates its own CI with an eval run`

### Task 6: Root README

- [ ] Write `README.md` per the design doc's section 2 (pitch, badges,
      architecture ASCII, screenshots from `docs/images/`, results tables
      copied faithfully from `training/README.md`, compose quickstart,
      real-provider setup, layout, quality/testing section, roadmap).
      Numbers must match `training/README.md` exactly — copy, don't recall.
- [ ] Commit: `docs: add root README`

### Task 7: Compose E2E + screenshots

- [ ] Fresh `docker compose up --build`; walk all five views in a browser;
      capture 3 screenshots (run results, rating room blind pair, compare
      table) to `docs/images/`; confirm README image links resolve.
- [ ] Commit: `docs: add dashboard screenshots to README`

### Task 8: Publish (user-gated)

- [ ] Secrets sweep: `git log -p | grep -cE "sk-ant-|sk-proj-|AQ\."` must be 0;
      `git ls-files | grep .env` must be empty.
- [ ] **Ask the user for explicit confirmation**, then `gh repo create
      evalforge --public --source . --push` (or push to a repo they create).
- [ ] Confirm CI runs green on GitHub; fix-forward any runner-only failures.
- [ ] Remind user: rotate the three provider API keys in `platform/api/.env`.

## Definition of done

- `docker compose up` from a clean checkout shows seeded data in all views.
- All three workflows green on GitHub (eval-gate may need one fix-forward).
- Root README renders with real numbers, screenshots, license, badges.
- Repo public under the user's GitHub account; no secrets in history.
