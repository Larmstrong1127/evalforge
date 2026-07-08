# EvalForge — Design Document

Date: 2026-07-02
Status: Approved pending final review

## 1. Purpose

EvalForge is an open-source LLM evaluation platform: submit a suite of prompts,
run them against any set of candidate models, score the outputs with any set of
judges (LLM-as-judge or a locally fine-tuned classifier), and track results,
regressions, and costs over time in a dashboard.

It is the successor to AgentForge (side-by-side multi-LLM comparison). AgentForge
proved that manual comparison of model outputs does not scale; EvalForge automates
the judgment.

Secondary purpose: portfolio project targeting specific Apple Seattle roles.
Each layer maps to a job family:

| Layer | Role targeted |
|---|---|
| PyTorch fine-tuned judge + benchmark | AIML - SWE, AI Evaluation; SWE in NLP & ML; ML Software Engineer |
| Next.js dashboard + A/B rating room | Full Stack SWE, Productivity Apps; Full-Stack Engineer, App Store |
| Async job runner + FastAPI | Server-Side SWE, ASE Enterprise |
| Terraform + cost/efficiency dashboard | Cloud Infrastructure and AI Efficiency Engineer |
| Eval regression gate in CI | Software Tools and Automation Engineer |

## 2. Architecture

```
Dashboard (Next.js 15 + TypeScript)
  runs list · results explorer · diff view · A/B rating · cost view
        | REST (OpenAPI)
API (FastAPI + Pydantic)
  suites · runs · results · ratings · costs · compare
Eval Runner (asyncio job queue)
  bounded concurrency · per-provider rate limits · retries · progress
Providers (plugin interface)     Judges (plugin interface)
  claude / openai / gemini /       llm_judge / deberta (ours) /
  ollama                           similarity / exact_match
        |
PostgreSQL (SQLite fallback for zero-setup demo)

training/   — separate PyTorch package; produces the DeBERTa judge artifact
```

### Key decisions (ADRs)

1. **`platform/` and `training/` are separate packages.** Training produces a
   model artifact; the platform consumes it as just another judge. Mirrors the
   industry separation of training and serving.
2. **Two plugin interfaces.** `Provider.generate()` and `Judge.score()`. All
   providers and judges are single-file implementations of these contracts.
   Adding a fifth of either is one file.
3. **asyncio, not Celery/Redis.** Eval runs are I/O-bound API fan-out with
   bounded concurrency; a persistent job table plus asyncio covers it. The
   README documents why ("why not Celery?").
4. **Local inference never blocks the event loop.** The DeBERTa judge wraps its
   forward pass in `asyncio.to_thread()`. A process pool is unnecessary because
   PyTorch releases the GIL during tensor ops; this reasoning is documented in
   code.
5. **Prompts are immutable and versioned.** Editing a prompt creates a new
   version row; results FK to the exact `prompt_version_id`. Cross-run diffs
   and regression tracking stay valid forever.
6. **Polling first, SSE second.** V1 dashboard polls run status via TanStack
   Query. V1.1 replaces polling with Server-Sent Events (FastAPI
   `StreamingResponse`) as a deliberate, visible scaling commit — the author
   has prior SSE production experience in AgentForge.
7. **Hybrid provider economics.** Demo mode runs 100% free and local (Ollama
   candidates + DeBERTa judge, no API keys). A one-time ~$10-15 cloud spend
   generates the flagship benchmark against Claude/GPT/Gemini.

## 3. Data model (PostgreSQL)

Generation and judgment are separate tables: one result (a model's output) can
be scored by N judges without duplicating output rows, and judge-disagreement
queries stay trivial.

- `suites` — name, description
- `prompts` — suite FK, immutable; `prompt_versions` — text, expected answer,
  version_number, created_at (append-only; UNIQUE(prompt_id, version_number))
- `candidate_models` — name, provider (normalized; results FK here)
- `runs` — suite FK, status (queued/running/done/failed), concurrency limit,
  progress counters, started/finished timestamps
- `results` — run FK, prompt_version FK, candidate_model FK, output text,
  status (ok/failed) + error text, latency_ms, tokens_in/out, cost_usd
- `judge_evaluations` — result FK, judge name, score, optional justification
  (LLM judges return reasoning; the DeBERTa judge returns probability)
- `ratings` — prompt_version FK, result_a FK, result_b FK, chosen FK, skipped
  flag, rater session, created_at (the phase-2 preference dataset)
- `cost_records` — per-run per-provider aggregates

## 4. Training package (PyTorch)

**Task:** binary hallucination detection — (question, context, answer) →
faithful | hallucinated.

**Model:** `microsoft/deberta-v3-base` (~184M params). Trains on a consumer GPU
in under an hour per run.

**Data:** HaluEval (~35K labeled examples) for train/val; RAGTruth held out
entirely as a cross-dataset generalization test.

**Approach:** HuggingFace `transformers` for model/tokenizer; hand-written
PyTorch training loop (not `Trainer`, not `accelerate`): explicit
zero_grad/backward/step, LR warmup schedule, gradient clipping, mixed precision
via the modern `torch.amp.autocast("cuda")` / `torch.amp.GradScaler("cuda")`
APIs (the `torch.cuda.amp.*` variants are deprecated in PyTorch 2.x and must
not appear in the codebase), early stopping on validation F1.

**Evaluation metrics:** precision/recall/F1, confusion matrix, and Expected
Calibration Error (ECE) — a judge whose 0.9 probability actually means ~90%
correct is usable for thresholding; an overconfident one is not.

**Artifact distribution:** the fine-tuned model is published to HuggingFace Hub
(`save_pretrained` format + model card), never distributed as a pickled
`torch.load` blob. The platform and CI pull it from the Hub at a pinned
revision. The public model card is itself a portfolio artifact.

**Structure:**

```
training/
  data/         download + preprocess (raw data gitignored)
  train.py      the loop
  evaluate.py   precision/recall/F1, confusion matrix, calibration
  export.py     weights + tokenizer + model card
  configs/      yaml per experiment
  README.md     results, loss curves (TensorBoard), "what didn't work" log
```

**Flagship deliverable:** benchmark report answering "is a fine-tuned 184M
classifier a viable replacement for LLM-as-judge?" — DeBERTa vs Claude vs GPT
vs Gemini as judges on the held-out set: agreement with ground truth, cost per
1K evaluations, p50/p95 latency. This table is the README centerpiece.

## 5. API surface

All routes under `/api/v1/`. `POST /runs` returns `202 Accepted` immediately
with the run id (async job semantics).

```
POST   /api/v1/suites                  create suite
POST   /api/v1/suites/{id}/prompts     add prompt (edit = new version)
POST   /api/v1/runs                    submit run → 202 + run_id
GET    /api/v1/runs/{id}               status + progress
GET    /api/v1/runs/{id}/results       scored results, filterable
GET    /api/v1/runs/{id}/costs         tokens + $ per provider/judge
POST   /api/v1/ratings                 human A/B preference vote
GET    /api/v1/compare?runs=a,b        cross-run regression diff
```

Pydantic v2 schemas at every boundary. OpenAPI docs served at `/docs`.

Note on the diff axis: comparisons are run-vs-run over identical prompt
versions (did a model/prompt change regress outputs?). Prompt-version-vs-
prompt-version diffing is not a primary view.

## 6. Dashboard (Next.js 15, App Router, TypeScript)

Stack: Tailwind + shadcn/ui, TanStack Query (server state; no global state
manager — documented YAGNI), Recharts.

Five views:

1. **Runs list** — status, progress, cost so far.
2. **Results explorer** — prompt × candidate × judge table; filter by score
   range, judge disagreement, provider; sortable.
3. **Diff view** — two runs side-by-side per prompt: regressions, improvements,
   judge score deltas. Depends on immutable prompt versions.
4. **A/B rating room** — blind pairwise voting; provider identity hidden until
   after the vote (LMSYS Chatbot Arena methodology, documented); keyboard
   shortcuts (arrows to vote, S to skip). Votes persist as preference pairs —
   the phase-2 reward-model training set.
5. **Cost & efficiency** — cost per 1K evals per judge, latency percentiles,
   token burn; headline chart is fine-tuned judge vs cloud judges.

## 7. Infra & CI

- **Docker Compose:** `api` + `web` + `postgres`; `docker compose up` gives a
  seeded, zero-API-key demo.
- **GitHub Actions:**
  - `ci.yml` — ruff/eslint, mypy/tsc, pytest/vitest on every PR.
  - `eval-gate.yml` — runs a fixed suite against a pinned Ollama model on PR;
    fails the build on score regression beyond threshold; posts score-diff
    comment. Dogfoods the product. The DeBERTa judge is pulled from HuggingFace
    Hub at a pinned revision and cached via actions/cache.
  - `train-check.yml` — 10-step CPU smoke test of the training loop.
- **Terraform (one environment):** ECS Fargate (api, web), RDS Postgres
  (smallest tier), ECR, ALB, Secrets Manager. Deploy, screenshot, destroy.
  Optional free-tier live demo: Vercel + Neon Postgres, read-only seeded data.

## 8. Phased build

| Phase | Contents | Duration |
|---|---|---|
| 1. Core engine | schemas, 4 provider adapters, judge interface, async runner, CLI | week 1 |
| 2. Training | fine-tune, evaluate, export; judge plugged into platform | week 1-2 (overlaps) |
| 3. Dashboard | 5 views, A/B room, polling | week 2-3 |
| 4. Infra & polish | Docker, 3 CI workflows, Terraform, benchmark, README | week 3 |
| 5. v1.1+ | SSE progress, reward model on collected preference pairs (DPO write-up) | post-ship |

Each phase ends in a merged PR with a substantive description.

## 9. Quality bar ("not vibe-coded")

- Conventional commits; no monolithic initial dump; history tells the story.
- `docs/adr/` with numbered records for every non-obvious call: ADR-001
  asyncio-not-Celery, ADR-002 to_thread-not-process-pool, ADR-003
  polling-then-SSE, ADR-004 no-global-state-manager, ADR-005
  HF-Hub-artifact-distribution.
- No deprecated APIs (e.g. `torch.cuda.amp.*`, compose `version:` key) — stale
  idioms are a generated-code tell.
- Tests written alongside code, not retrofitted.
- Training README includes a failure log (learning rates that diverged,
  overfitting observations) with TensorBoard evidence.
- No AI attribution in commit history.
- README written last, from real results, with real benchmark numbers.

## 10. Testing strategy

- **Platform:** pytest — unit tests for adapters (mocked HTTP), judge scoring,
  runner concurrency/retry logic; API contract tests via FastAPI TestClient.
- **Dashboard:** vitest + Testing Library for components; one Playwright smoke
  path (create suite → run → see results) if time allows.
- **Training:** deterministic-seed smoke test (10 steps, loss decreases);
  evaluate.py tested against hand-computed fixtures.
- **CI as test:** the eval-gate workflow is itself an integration test of the
  whole engine.

## 11. Error handling

- Provider failures: retry with exponential backoff (bounded); a failed
  completion marks the result failed, never the whole run.
- Judge failures: recorded per-result; run completes with partial scores
  flagged in the UI.
- Run cancellation: cooperative — runner checks a cancel flag between tasks.
- All costs recorded even for failed results (tokens were still spent).
