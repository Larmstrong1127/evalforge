# EvalForge Phase 4 — Publish & Polish Design

Date: 2026-07-08
Status: Approved
Parent spec: `docs/superpowers/specs/2026-07-02-evalforge-design.md` (sections 7-9)
Depends on: Phases 1, 2, 3a, 3b-1, 3b-2 (all merged to master)

## 1. Scope

Everything the parent spec's Phase 4 promised except Terraform (deferred by
explicit decision — the Compose demo plus README screenshots deliver most of
the reviewer-facing signal at none of the AWS cost, and the MedInsight
project already covers the Terraform/ECS story on the resume):

1. **Root README** — the repo's front door: pitch, architecture, real
   benchmark results, quickstart, honest limitations.
2. **LICENSE** — MIT. The resume calls the project open-source; that claim
   needs the file.
3. **CI (GitHub Actions)** — `ci.yml` (lint/type/test both packages),
   `train-check.yml` (training package test suite on CPU), and the flagship
   `eval-gate.yml` (the platform gating its own CI with an eval run —
   dogfooding).
4. **Docker Compose demo** — `docker compose up` gives a seeded, browsable
   demo with zero API keys.
5. **Two new ADRs** — the background-task commit-ordering decision and the
   judge plugin interface (both non-obvious calls that earned documentation).
6. **Repo hygiene for publication** — remove tool-generated scaffold stubs,
   replace boilerplate sub-READMEs, rename `docs/superpowers/` →
   `docs/design/` and strip tool-specific header boilerplate from the plans
   (the substantive design/plan content stays — it documents genuine process).
7. **Publish to GitHub** — create the public repo and push. Gated on an
   explicit user confirmation immediately before pushing (publishing is
   outward-facing), and preceded by a secrets sweep. The API keys currently
   in the gitignored `platform/api/.env` must be rotated by the user before
   or shortly after publication since they have appeared in local tooling
   output during development.

## 2. Root README

Single `README.md` at repo root containing, in order:

- Name + one-sentence pitch: open-source LLM evaluation platform — run
  prompt suites against multiple providers, score with pluggable judges
  (including a fine-tuned local model), collect human preference votes,
  diff runs.
- Badges: CI status (after first push), license.
- Architecture diagram (ASCII, adapted from the parent spec — kept as text,
  not an image, so it renders everywhere and diffs cleanly).
- Screenshots (2-3: run results view, rating room, compare view) captured
  from the running Compose demo, stored in `docs/images/`.
- **Results section** — the differentiator. Real numbers from
  `training/README.md`: the fine-tuned DeBERTa judge's in-distribution vs
  out-of-distribution F1 (0.9937 vs 0.5067), and the benchmark table
  (local judge $0 / 27ms / 47.5% agreement vs Claude / GPT-4o / Gemini).
  Framed honestly: the local judge is a cheap first-pass filter, not a paid
  judge replacement — with a link to the full training write-up including
  the "what didn't work" section.
- Quickstart: `docker compose up` → URL → what you can click. Separate
  short section for running against real providers (env keys) or host
  Ollama.
- Project layout: `platform/api`, `platform/dashboard`, `training`,
  `docs/adr`, `docs/design`.
- Testing/quality: test counts, lint/type tooling, the eval-gate CI concept.
- Deferred/roadmap: SSE, reward-model training on collected preference
  pairs, Terraform deploy.

## 3. CI workflows

- **`ci.yml`** — on PR and push to master. Two jobs:
  - `backend`: Python 3.12, `pip install -e ".[dev]"` in `platform/api`,
    then ruff, mypy, pytest. A second matrix entry (or step) does the same
    for `training/` (its tests are mocked and CPU-fast).
  - `frontend`: Node 20, `npm ci` in `platform/dashboard`, then lint,
    build (covers tsc), vitest.
- **`train-check.yml`** — on PR touching `training/**`: runs the training
  package's pytest suite on CPU (29 mocked tests, fast). The parent spec's
  "10-step training smoke" is satisfied by the existing loop tests; a real
  10-step run would need dataset downloads and is not CI-appropriate.
- **`eval-gate.yml`** — the flagship, on PR touching `platform/**`:
  installs Ollama on the runner, pulls a small pinned model
  (`llama3.2:1b`), starts the API, creates a fixed 5-prompt suite via the
  CLI/API, runs it with the `exact_match` judge, compares the aggregate
  score against a committed baseline (`.github/eval-baseline.json`), and
  fails if the score drops more than the threshold. Score summary is
  written to the GitHub Actions step summary. This is genuinely the product
  gating its own CI. Accepted tradeoffs: ~5-8 min runtime; small-model
  scores are noisy, so the threshold is generous (fail only on collapse,
  not jitter) and the suite uses deterministic single-token-answer prompts.

## 4. Docker Compose

- `platform/api/Dockerfile` — python slim, install with `postgres` extra,
  uvicorn entrypoint.
- `platform/dashboard/Dockerfile` — multi-stage node build, `next start`.
- `docker-compose.yml` at repo root: `postgres` (16-alpine, healthcheck),
  `api` (waits on postgres healthy, `EVALFORGE_DATABASE_URL` pointed at
  postgres via asyncpg), `web` (`NEXT_PUBLIC_API_BASE_URL` →
  `http://localhost:8000`).
- **Seeding**: a small idempotent script (`platform/api/scripts/seed_demo.py`)
  run by the api container on startup when `EVALFORGE_SEED_DEMO=1` (compose
  sets it): creates a demo suite with a handful of prompts, one completed
  run with two fake-candidate results and exact_match evaluations — enough
  that the suites list, run detail, rating room, and compare views all have
  real data to show with zero API keys. Seeded rows are created directly via
  the ORM (not by calling providers).
- Ollama is NOT in compose (multi-GB images); the README documents pointing
  `EVALFORGE_OLLAMA_BASE_URL` at `host.docker.internal:11434` to use a host
  Ollama for live runs.

## 5. New ADRs

- **ADR-003 — commit-before-BackgroundTask**: records the real production
  bug (BackgroundTasks execute before the request-scoped session's
  post-yield commit; a flush()-only row is invisible to the task's own
  session), why the fix is an explicit commit in `create_run`, and why the
  mocked test suite could not have caught it (shared session factory).
- **ADR-004 — judge plugin interface with optional heavy extras**: records
  the `Judge` protocol, `None`-means-cannot-judge semantics, and the
  decision to ship the DeBERTa judge behind an optional `deberta` extra
  with lazy registry import so the base platform never pays the
  torch/transformers cost.

## 6. Repo hygiene

- Delete `platform/dashboard/CLAUDE.md` and `platform/dashboard/AGENTS.md`
  (tool-generated scaffold stubs).
- Replace `platform/dashboard/README.md` (create-next-app boilerplate) with
  a short real one; add a short `platform/api/README.md` and keep
  `training/README.md` as-is (it is already the strongest doc in the repo).
- `git mv docs/superpowers docs/design`; strip the "For agentic workers /
  REQUIRED SUB-SKILL" header block from the plan files (content otherwise
  unchanged — the specs and plans document real engineering process and
  stay public).
- Add `docs/images/` for README screenshots.

## 7. Publish

- Secrets sweep (`git log -p` grep for key patterns; confirm `.env` never
  tracked) before anything leaves the machine.
- `gh repo create` public under the user's account + push — **only after
  an explicit user go-ahead at that step**. CI badges added to the README
  once the first workflow run exists (or added optimistically with the
  final repo path).
- User action item (called out, not automatable): rotate the Anthropic/
  OpenAI/Gemini keys in `platform/api/.env`.

## 8. Testing this phase

- Compose demo verified end-to-end locally: fresh `docker compose up
  --build`, browse all five views against seeded data, capture the README
  screenshots during this same session.
- Workflows validated by (a) local equivalent commands, and (b) actually
  running on GitHub after the push — the eval-gate in particular can only
  be fully proven on a real runner; a follow-up fix commit after first push
  is an accepted, normal outcome.

## 9. Explicitly deferred

- Terraform/AWS deploy (user decision this phase).
- SSE, reward-model training, pagination/auth — unchanged from prior specs.
