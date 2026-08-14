# Portfolio Audit — Strict Hiring-Manager Pass

**Date:** 2026-08-11 (audit executed 2026-08-13)
**Persona:** senior hiring manager / skeptic in the debrief
**Candidate:** MS CS May 2024, zero professional engineering years
**Targets:** Netflix INK SWE 3 (GenAI animation studio platform); junior AI/ML broadly
**Scope:** everything except the flagship pair (`asset-provenance`, `comfy-workflow-pack`), which were reviewed and upgraded immediately prior. Nothing glaring was found in either during incidental checks; their headline numbers were independently re-verified (134 tests, 118 tests, 1,791 assets / 934 derivations) and hold.
**Constraint honored:** no project code was modified. This document is the only file written.

---

## Verdict table

| Project | Verdict | One-line reason |
|---|---|---|
| **evalforge** | **SOLID** | Best artifact in the portfolio. 166 tests green, `mypy --strict`, ADRs, honest README. Loses points for a known-broken judge encoding left in the serving path and stale self-reported counts. |
| **dentavision-v3** | **NEEDS WORK** | Real tests (20/20 pass, `supertest` + in-memory Mongo), honest HIPAA scoping — undercut by an async-middleware crash bug and `err.message` leaking to clients. |
| **DocuChat** | **NEEDS WORK** (LIABILITY as presented) | Competent RAG core; the service wrapper is undeployable, and CI is rigged so a green badge sits over zero tests. |
| **agentforge** | **LIABILITY** | `npm run build` fails. CI edited to swallow it. Repo description advertises Gemini support and memory management that do not exist. |
| **WAVets2Tech** | **LIABILITY** | The "led a team of 4 through full SDLC" claim is actively contradicted by the artifact: 9 solo zip-dump commits, no branches, no PRs. The advertised React SPA is `<h1>THIS IS TEST CODE</h1>`. |
| **Echoed-Nights-Video-Game** (public) | **NEEDS WORK** | The reframe fixed the old overclaim and introduced new ones, including a fabricated citation to a PDF sitting in the same repo. |
| **shorts-factory** (showcase) | **NEEDS WORK** | README is unusually honest and its counts verify; `docs/architecture.html` and `config.schema.json` are stale and contradict it. |
| **A-Star-Algorithm** | **SOLID** (cosmetic nits) | Builds, 10/10 tests pass, no secrets. One README claim exceeds the code. |
| **TechCon-Convention-Site** | **SOLID** (cosmetic nits) | No secrets, README matches code. EOL framework, unrenamed coursework naming. |
| **Profile README + larmstrong1127.github.io** | **NEEDS WORK** | Every hard number verified true. Exposure is in the unmaintained edges: a dead demo still labeled "live", a stale resume, and a competing public portfolio. |
| **portfolio-website** (public, unhosted) | **LIABILITY** | Still carries the retracted Echoed Nights FSM claim and links a now-private repo. |

---

## The cross-cutting pattern

Read across all eleven surfaces, one failure mode dominates, and it is not sloppiness — it is something subtler and more damaging:

> **The candidate documents defects beautifully and then does not fix them.**

- `evalforge` diagnosed that its DeBERTa judge truncates the answer off 51% of RAGTruth examples, measured that a corrected encoding lifts ROC-AUC 0.444 → 0.603, published the whole analysis — and left the broken encoding in the serving path.
- The Echoed Nights README describes a flee behaviour that exists only inside a `/* */` block.
- The shorts-factory showcase claims FLUX fp8 when nf4 is what runs, and "fully autonomous" when 4 of 21 accounts are on auto-schedule.
- `evalforge`'s own README flags that `docker compose up` — the very first command in the Quickstart — has never been executed.

The honesty is genuinely rare and genuinely valuable; several of these disclosures are the strongest signal in the entire application. But a hiring manager draws a line between *"finds real problems"* and *"closes real problems"*, and right now the evidence lands almost entirely on the first side. Every one of the fixes above is bounded — hours, not days.

A second, narrower pattern: **two public repos have CI configured to hide failure** (`agentforge`, `DocuChat`). That is the one thing in this audit I would read as a character signal rather than an experience gap, and it is why both are ranked above much larger technical problems in the fix list.

---

## 1. evalforge — SOLID

Independently confirmed by running everything: **80 backend pytest + 76 training pytest + 10 dashboard vitest = 166, all green.** CI runs `ruff` + `mypy --strict` + `pytest` + `tsc --noEmit` + `next build` on every PR, plus a self-hosted eval gate. Four ADRs. 119 of 126 commits conventional. Git history clean of secrets (`git log -p --all -S "sk-ant-api"` and `-S "AIzaSy"` both empty). `.gitignore` is correct and nothing junk is tracked (452 KB pack). The README's PostgreSQL claim is real (`asyncpg` extra + `docker-compose.yml:3`).

The code quality is well above junior median. `platform/api/evalforge/runner.py:150-200` carries a genuinely sophisticated comment explaining why re-acquiring the lock in the exception handler cannot deadlock; `judges/reward_judge.py:1-48` is a model of how to document a score's statistical semantics. This is the artifact to lead with.

### Top issues

**E1 — The known-broken judge encoding is still in the serving path. (worst)**
`platform/api/evalforge/judges/deberta_judge.py:88-89`:
```python
text = f"Q: {question} C: {context} A: {answer}"
inputs = self._tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
```
`README.md:111-127` documents that this exact encoding cuts the ANSWER — the only thing being classified — off entirely on 102/200 RAGTruth examples, and that an answer-preserving encoding raises ROC-AUC from 0.444 to 0.603. The diagnosis shipped; the ~10-line fix did not. Meanwhile `reward_judge.py` derives its budget from the checkpoint config precisely so this class of drift *cannot* recur. Same author, same repo, same defect class, fixed in one place and left in the other. A reviewer who reads the README and then opens the file will ask about this, and there is no good answer.

**E2 — Stale numbers in the section titled "Quality".**
`README.md:152-154` claims "76 backend, 50 training, 10 frontend". Measured: **80 backend, 76 training**, 10 frontend. Training is understated by 26. Understating is far less damaging than overstating, but a skeptic checks the Quality section first precisely because it invites checking.

**E3 — The architecture diagram omits the flagship.**
`README.md:21` lists judges as `exact_match · llm_judge · deberta-hallucination (ours)`. The `reward` judge is registered (`judges/__init__.py:56-60`) and is the headline artifact at `README.md:13`. `README.md:146` likewise describes `training/` as being for "the DeBERTa judge" only.

**E4 — Zero authentication on a platform that spends money.**
No auth, no rate limiting anywhere in `platform/api/evalforge/` (grep for `HTTPBearer`/`Depends(auth`/`rate_limit` returns nothing). `POST /api/v1/runs` is an unauthenticated endpoint that fans out to billed provider APIs with a caller-supplied concurrency limit. For a *local dev tool* this is defensible; the README presents EvalForge as a platform and never states the boundary. One sentence fixes the framing; ~3h fixes the substance.

**E5 — Asymmetric generation budgets silently bias every eval.**
`providers/anthropic.py:26` hardcodes `"max_tokens": 1024`. `providers/openai.py`, `gemini.py`, and `ollama.py` set no cap at all. So a head-to-head run truncates Claude at 1024 tokens and lets GPT-4o and Gemini run free, then judges all of them on the result. In a tool whose entire purpose is fair model comparison, this is a correctness bug, not a config nit — and it is thematically the *same* truncation-bias family as E1 and as the disclosed HF seq-len defect. Three strikes in one repo.

**E6 — The human-preference table accepts referentially invalid rows.**
`api/ratings.py:11-24` inserts a `HumanRating` with no validation that `result_a_id` / `result_b_id` / `chosen_result_id` exist, and no check that `chosen_result_id ∈ {a, b}`. `schemas/ratings.py:10` is a bare `UUID | None`. `db/engine.py` never issues `PRAGMA foreign_keys=ON`, so SQLite enforces nothing. This table feeds `export_rating_pairs.py`, i.e. it is an ML training-data source with no integrity constraint.

**E7 — N+1 queries on both read paths.**
`api/runs.py:195-205` issues one `SELECT` per result inside the response loop (`limit` clamps at 5000 → up to 5000 queries). `api/compare.py:29-33` has the identical pattern. One `IN` query plus a group-by fixes both.

**E8 — Smaller things a reviewer would circle.**
- `judges/llm_judge.py:24` — `_JUDGE_MODEL = "claude-sonnet-5"` hardcoded, not configurable. The judge model is the single most important knob in an LLM-as-judge system.
- `api/runs.py:60,65` — `assert run is not None` for control flow; `python -O` strips these.
- `Settings()` is constructed per-request (`api/runs.py:88`) and per-background-task (`:52`), re-parsing `.env` each time. `lru_cache` dependency is the idiom.
- `main.py:35` — CORS origins hardcoded to `localhost:3000`; not configurable for any real deployment.
- `api/runs.py:180` — `status_filter` is an unvalidated raw string compared against an Enum column; garbage yields an empty list rather than a 422.
- Reproducibility gap: `.gitignore` excludes `training/runs/`, but `docs/hf-model-audit-2026-07-26.md:92` cites `training/runs/reward-lr2e5-512-stdout.log` as the evidence for the 512 re-measurement. A reviewer cannot see the evidence for the correction.
- Open from the prior audit: **B2** (no `metrics:`/`model-index:` block, no `library_name`), **B4** (identity gap — HF handle `DantheMan124` appears nowhere else on the application; no legal name in either model card), **C2** (length-bias probe), **C3–C7**. B4 is ~10 minutes and has outsized effect: a reviewer cross-referencing three identities has a genuine "is this his model?" moment.
- Docker: `README.md:60-66` honestly discloses that `docker compose up --build` has never been run. I inspected both Dockerfiles, the compose file, and the dashboard `.dockerignore` and found **no defect** — Next 16 standalone defaults `HOSTNAME` to `0.0.0.0`, the build-arg wiring is correct, and the API image copies only what it needs. The path looks sound; it is simply unverified.

### Fix order (evalforge)

| # | Fix | Effort |
|---|---|---|
| 1 | Fix `deberta_judge.py` encoding (truncate context, keep Q and A whole), re-run the 200-example benchmark, update the README table | 2–3 h |
| 2 | Correct `README.md:152-154` counts; add `reward` to the diagram at `:21` and to `:146` | 20 min |
| 3 | Set an explicit, equal `max_tokens` across all four providers; make it a run parameter | 1.5 h |
| 4 | Validate ratings (`chosen ∈ {a,b}`, FKs exist) + enable `PRAGMA foreign_keys=ON` | 1.5 h |
| 5 | Close B4 (identity) and B2 (`model-index`) on both HF cards | 45 min |
| 6 | Either add an API key + rate limit, or state the local-tool boundary in the README | 3 h / 10 min |
| 7 | Fix both N+1 loops | 1.5 h |
| 8 | Install Docker, actually run `docker compose up --build`, replace the caveat with a verified claim | 1 h |
| 9 | Make the judge model configurable; replace `assert`s; cache `Settings` | 1.5 h |
| 10 | C2 length-bias probe (~20 lines, high interview value) | 1.5 h |

---

## 2. dentavision-v3 (public: DentaVision) — NEEDS WORK

Better than most junior portfolio code: `helmet`, per-route rate limiting, `express-mongo-sanitize`, bcrypt via model hooks, password stripped from responses. **Suite verified: 20/20 pass in 32s** via real `supertest` + `mongodb-memory-server` — not mocked-trivial, and the README's test claim is accurate. The HIPAA disclaimer at `README.md:37-46` is honest and shows judgment. Model IDs are current. **No secrets in git history**; the live keys in `server/.env` are correctly gitignored (rotate anyway).

### Top issues

- **D1 — Unhandled async rejections in auth middleware.** `server/middleware/auth.js:20-40`: `requireClinic` and `requirePatient` are `async` with no `try/catch`, and Express 4 does not catch async rejections. A malformed ObjectId in a JWT payload produces a `CastError` that never reaches the error handler — the request hangs, and on Node ≥15 the process terminates. This is on the hot path for every authenticated route.
- **D2 — Internal error text leaked to clients.** `server/app.js:85-91` gates only `err.stack` behind `NODE_ENV`; `err.message` always ships. Worse, every route bypasses that handler with its own `catch (err) { res.status(500).json({ error: err.message }) }` (`routes/auth.js:39,60,92,124`; `routes/scan.js:93,122,145,167,194`). Confirmed live: the deployed demo returns `{"error":"Operation \`clinics.findOne()\` buffering timed out after 10000ms"}` to the browser.
- **D3 — Claude service is fragile.** `server/services/aiService.js:167` indexes `response.content[0].text` blind (a refusal → `TypeError`); `:168` strips fences with a regex that any preamble defeats; no client timeout override, no cost ceiling despite a 10 MB image upload path (`routes/scan.js:13`). `output_config.format` with a JSON schema would delete the fragile block outright.
- **D4 — Access control expressed by accident.** `server/routes/scan.js:153-159` fetches an *arbitrary* patient of the clinic and then searches that one patient's subdocuments for the plan. It fails closed (wrong patient → 404) so it is not exploitable, but the intent is not encoded and the endpoint cannot work reliably.
- **D5 — Repo hygiene.** A directory literally named `{client` sits at the repo root — a brace-expansion `mkdir` that ran on a shell without brace support. `client/build/` and duplicate `screenshots/` + `docs/screenshots/` are committed. Visible in the first ten seconds.
- Minor: `@anthropic-ai/sdk: ^0.24.0` will never pick up fixes (0.x caret pins to 0.24.x). No startup assertion for `JWT_SECRET` / `MONGODB_URI`. JWT in `localStorage` (defensible — no `innerHTML` sinks anywhere in `client/src`, so the XSS surface is genuinely closed).

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | Rotate the Anthropic key and Atlas password | 30 min |
| 2 | `asyncHandler` wrapper across middleware and routes | 1 h |
| 3 | Generic 500s + server-side logging; delete per-route catches | 2 h |
| 4 | `aiService`: `stop_reason` check, structured outputs, explicit timeout | 2–3 h |
| 5 | Scope the plan lookup by plan ID **and** clinic | 30 min |
| 6 | Startup env assertions | 30 min |
| 7 | Delete `{client`, `client/build/`, duplicate screenshots; bump SDK | 1 h |

---

## 3. DocuChat — NEEDS WORK (LIABILITY as currently presented)

The RAG core is competent: cosine collection with `hnsw:space`, correct `relevance_score = 1 - distance`, a properly grounded prompt (`rag.py:211-219`), and citation numbering that genuinely lines up. Deps are pinned (except one). No secrets in history.

### Top issues

- **DC1 — CI is rigged and the badge is a claim.** `.github/workflows/ci.yml:32` runs `python -m pytest -v --tb=short || true`. There are **zero test files**. `README.md:3` displays a green CI badge. A green badge over no tests is worse than no badge, because it asserts something. **This is the single highest hiring-impact / lowest-effort fix in the entire portfolio.**
- **DC2 — The delete bug is still present and silently lies.** `backend/rag.py:255-273` keys deletion on `source: filename` while `ingest` mints a fresh `doc_id` per upload (`:129`) — upload the same file twice and delete nukes both. It also returns `{"deleted": True}` unconditionally, so `main.py:167-171` reports success for files that were never indexed.
- **DC3 — README model drift confirmed, and it names a retired model.** `README.md:68,83-84` claim `claude-3-5-haiku`; `rag.py:227` calls `claude-haiku-4-5`. The docstring at `rag.py:182` independently claims `claude-3-5-haiku-20241022`, **retired 2026-02-19**. The running code is right; the documentation describes a dead API call.
- **DC4 — Undeployable service wrapper.** No auth on any route; a single global Chroma collection named `"documents"` (`rag.py:45-48`) means every user shares one index and can delete each other's documents. `main.py:29-35` sets `allow_origins=["*"]` with `allow_credentials=True` — a combination browsers reject outright, so it is simultaneously maximally permissive and broken. `main.py:96` reads the entire upload into memory with no size cap. `main.py:159` uses `{filename:path}`, permitting traversal that is harmless only because the value never touches the filesystem.
- **DC5 — Every `async def` route blocks the event loop.** `rag.py:132,190` call sentence-transformers synchronously; `rag.py:226` uses the blocking `anthropic.Anthropic()` client. One upload freezes every concurrent request. Also `main.py:142` calls `rag.list_documents()` on every chat request, and `rag.py:160` does an unlimited `collection.get(include=["metadatas"])` — O(total chunks) per question, to answer a boolean.
- Minor: character-window chunking with no sentence/page awareness (`rag.py:54-97`); `RAGPipeline()` constructed at import time (`main.py:45`); `anthropic>=0.40.0` floats across majors.

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | **Remove `\|\| true` from `ci.yml`; write real tests** (chunk/ingest/delete/upload-validation, mock Anthropic) | 4–6 h |
| 2 | Fix delete (key on `doc_id`, real count); drop `:path` | 1.5 h |
| 3 | Correct the model name in `README.md:68,83-84` and `rag.py:182` | 15 min |
| 4 | **Either** add auth + per-user namespacing **or** state plainly in the README that this is a single-user local demo | 4 h / 15 min |
| 5 | Lock CORS; enforce a max upload size | 1.5 h |
| 6 | `AsyncAnthropic` + `run_in_threadpool`; `collection.count()` guard | 1.75 h |
| 7 | Lifespan handler; pin `anthropic`; rotate the key | 1 h |

---

## 4. agentforge — LIABILITY

**The repo does not build, and CI was edited to hide it.**

```
$ npm run build   → exit 1
Turbopack build failed: the chunking context does not support external modules
  (request: node:fs/promises)
  ./node_modules/@anthropic-ai/sdk/... → ./lib/providers/anthropic.ts → ./app/compare/page.tsx [Client Component Browser]
```

Root cause: `components/AgentForm.tsx:5-6` and `app/compare/page.tsx:4-5` are `"use client"` files importing `@/lib/providers/{openai,anthropic}`, and `lib/providers/anthropic.ts:3` / `openai.ts:3` instantiate the Node SDK clients at module scope — dragging the SDK and an `apiKey` read into the browser bundle. `npx tsc --noEmit` passes, so this is purely a bundling failure.

`.github/workflows/ci.yml:34` runs `npm run build || echo "Build skipped (requires live DB)"` and `:33` runs `npm run lint || true`. `README.md:3` shows the resulting green badge.

### Claims vs code

- **GitHub repo description advertises Gemini support — there is no Gemini code anywhere.** `lib/providers/` contains only `openai.ts` and `anthropic.ts`. This is the most damaging single line on the profile: it is a flatly false capability claim on a public repo.
- **"Memory management"** is `prisma.message.findMany` scoped to one run (`app/api/chat/route.ts:21-24`). Not a memory subsystem.
- Tool calling and agent loops **are** real and correctly implemented (`app/api/chat/route.ts:44-87` OpenAI, `:101-155` Anthropic, both handling multi-turn tool results). Credit where due — this is the salvageable part.
- `README.md:29` says "Next.js 14"; `package.json:16` pins `16.2.6`. `README.md:14` still calls web search "simulated"; commit `1dc77fd` made it a real DuckDuckGo call.

### Other issues

Zero auth on every route — `app/api/agents/[id]/route.ts:36-39` allows unauthenticated `DELETE` of any agent; `app/api/compare/route.ts:9` passes an arbitrary caller-supplied `model` string to billed API keys with no allowlist or rate limit (a billing incident if deployed). No zod on `/api/compare` (the CRUD routes have it). `lib/tools/index.ts:66` evaluates model output via `Function(...)` — the regex on `:65` makes it non-exploitable as written, but it is exactly what a reviewer greps for. `app/api/chat/route.ts:41` never resets `fullContent`, so persisted assistant messages concatenate pre-tool text with the final answer. `next.config.ts:2` imports `dotenv`, which is not a declared dependency. Two of three Anthropic model IDs (`lib/providers/anthropic.ts:6-7`) are non-API-valid and will 404. No tests. `npm audit --omit=dev`: 10 vulns, 6 high. **Secrets clean** — `.env`/`.env.local` gitignored, absent from history.

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | Move model-ID arrays to `lib/models.ts` (plain data, no SDK import) — **fixes the build** | 1 h |
| 2 | Rewrite the repo description (drop Gemini + memory); README Next 14→16; un-"simulate" web search | 30 min |
| 3 | Un-rig CI: remove `\|\| echo` and `\|\| true` | 15 min |
| 4 | Model allowlist + zod + per-IP rate limit on `/api/compare` | 2 h |
| 5 | Replace `Function()` calculator; reset `fullContent`; declare `dotenv` | 1.5 h |
| 6 | Any tests at all | 3 h |
| 7 | `npm audit fix` / bump Next | 30 min |

**If items 1–3 (~2 h) are not done this week, archive the repo.** A public repo that fails to build, shows a green badge, and advertises a provider it does not support is worse than no repo.

---

## 5. WAVets2Tech — LIABILITY

**The artifact contradicts the résumé line it exists to support.**

- 9 commits total, **all by the candidate** under three identities. Zero other authors, zero branches beyond `master`/`main`, **zero merges, zero PRs**.
- The commits are zip dumps: `9576cac "Add project files."`, `c1a63dd "Add files via upload"`, `ab46727 "Delete WA VETS2TECH (1).zip"`.

A hiring manager clicking **Insights → Contributors** sees a solo repo with no version-control discipline. "Led team of 4 through full SDLC" is not merely unsupported here — the artifact is affirmative evidence *against* the "full SDLC" half of the claim. This is the item most likely to be raised in a debrief as a credibility question rather than a skills question.

**The advertised React SPA does not exist.** `WAVets2Tech API/ClientApp/src/App.js:22-24` is three `<h1> THIS IS TEST CODE </h1>` elements and one `fetch("api/company")`. No router, no components directory. The README claims Job Listings, Veteran Profiles, an Employer Directory, and *"Architected and built the React single-page application."* There is **no `JobController.cs`** despite `Job.cs` and `BookmarkedJob.cs` models existing — the flagship feature has neither API nor UI. README "How to Run" says `cd client`; the directory is `WAVets2Tech API/ClientApp`.

**Security.** No authentication or authorization anywhere — zero `[Authorize]` attributes, no auth middleware in `Program.cs`. Anonymous `DELETE /api/student?id=1&id=2` bulk-deletes veteran records (`Controllers/StudentController.cs:83-100`); anonymous `POST /api/admin` creates an admin (`AdminController.cs:52-59`). EF entities are returned directly with no DTOs — `StudentController.cs:30` serializes `Student` including `PasswordHash` (`Models/Student.cs:36`), address, phone, and a profile-picture blob to any anonymous caller. `PostStudent(Student)` / `PutStudent(int, Student)` bind entities straight from the request body (mass assignment). `Program.cs:29-35` is `AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()`.

**On the connection string — calibrating this correctly:** `Models/Wavets2TechContext.cs:47` and `appsettings.json:11` hardcode `Server=.;Database=WAVets2Tech; Integrated Security = True; TrustServerCertificate=True;`. This is **not a credential leak** — it is Windows integrated auth against localhost, so there is no password to steal. What makes it a finding is that Microsoft's scaffolded `#warning` telling you not to do this **is still in the file**. A reviewer reads that as "was told, ignored."

Also: `Program.cs:11,15-18` registers the DbContext twice; `WeatherForecastController.cs` is untouched scaffolding; a dead `IConfiguration _configuration` field is copy-pasted into all six controllers; no tests; client deps are 2021-era (`react-scripts ^4.0.3`, `--openssl-legacy-provider`) and will not install cleanly on modern Node. No secrets leaked.

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | **Decide:** make private and drop from the profile, **or** invest ~10 h below. Do not leave it as-is. | 0 h |
| 2 | Rewrite the README "My Role" section to describe the coordination work without implying this git history evidences it | 30 min |
| 3 | If keeping: build one real screen so the SPA claim is true; fix `cd client` | 4 h |
| 4 | Remove the hardcoded connection string + the `#warning`; delete the duplicate `AddDbContext`; scope CORS | 1 h |
| 5 | Read/write DTOs so `PasswordHash` stops leaving the server | 3 h |
| 6 | `[Authorize]` + JWT — **or** a one-line README "no auth; coursework prototype" disclaimer | 4 h / 10 min |
| 7 | Delete `WeatherForecast*` and the dead `IConfiguration` fields | 20 min |

---

## 6. Coursework tier — SOLID (cosmetic only)

**A-Star-Algorithm.** `dotnet test` → **10/10 pass**. No secrets, no `bin/`/`obj/` tracked, no PII. Two nits: `README.md:14` claims *"Configurable heuristic function (Manhattan, Euclidean)"* but only Euclidean exists, hardcoded at `CSC595 FinalCode/Program.cs:206-209` — nothing is configurable (**15 min**: delete the parenthetical or add the option). `CSC595 FinalCode.csproj:5` targets `net6.0` (out of support). The directory name `CSC595 FinalCode` reads as an unedited homework drop.

**TechCon-Convention-Site.** No secrets (`appsettings.json` has no connection string; the SQLite path is inline at `LandonWebsite06/Startup.cs:20`), no tracked binaries, README matches the code. Nits: targets `netcoreapp3.1`, **EOL December 2022** — the README honestly says 3.1 so there is no contradiction, but it invites the question. The repo is named `TechCon-Convention-Site` while the solution/namespace is `LandonWebsite06`. `wwwroot/Images/` commits NVIDIA/Intel/Nintendo/Blizzard logos as seed data — low risk, but it is third-party trademark imagery in a public repo. Nothing embarrassing; no instructor or classmate PII.

---

## 7. Echoed Nights (public docs repo vs private Unity source) — NEEDS WORK

The reframe fixed the old overclaim and introduced two new ones. Ground truth from `D:\ClaudeProjects\echoed-nights\Assets\Scripts\Shadowlop_Scripts\Monster.cs` (the only enemy AI script, attached in `Night1/Night2/GameStart-Night1.unity`): **live code is lines 1–104 only.** Lines 117–193, 199–269, and 276–356 are inside `/* */`. The shipped AI is a **NavMesh waypoint patrol plus a distance-triggered, 3-second-dwell chase.** That is all of it.

| # | Public claim | Reality |
|---|---|---|
| 1 | `README.md:38,48,66-68` + the **GitHub repo description**: "coroutine-driven wander/chase/flee" | **False.** No coroutine in live `Monster.cs`. The wander/chase/flee coroutines are `:276-356` — commented out *and* outside the class body, i.e. not even compilable. **There is no flee behaviour in the shipped enemy.** `README.md:71-74` calls the flee branch "the part worth defending" — that paragraph defends dead comment text. |
| 2 | `README.md:66`: wander picks "random reachable NavMesh points" | **False.** Live wander is a deterministic waypoint ring (`:86`). `RandomNavMeshLocation()` exists only in the comment blocks. |
| 3 | `README.md:60-61`: "the waypoint-patrol FSM exists but is commented out" | **Inverted.** Patrol is the part that *is* live (`:82-92`). |
| 4 | `README.md:70`: "no memory — once chasing it never returns to wandering" | **False.** `:73-78` re-enters patrol every `FixedUpdate` the player is outside `attackRadius`. |
| 5 | `README.md:55-58`: "The capstone report specifies a four-state FSM: Patrol, Alert, Chase, Search" | **Not in the report.** All 18 pages of `MSCS-ProjectReport-Landon.pdf` contain zero occurrences of "four", "Alert", "Search", "waypoint", or "NavMesh". The only FSM mention is a related-work note about F.E.A.R. The report's actual AI section (§3.9) specifies reinforcement learning. **This is a fabricated citation to a PDF sitting in the same repo** — the highest-credibility-cost item in the entire audit. |

Undisclosed wart: `Monster.cs:68` calls `Vector3.MoveTowards(transform.position, target.position, 100f)` — a 100-unit step just returns the player's position, so it is a no-op dressing on `navmesh.destination`, and `transform.LookAt` fights the NavMeshAgent's own rotation.

**What is defensible and currently *under*-sold:** the reinforcement-learning claim (`README.md:21`) is the best-evidenced claim in the repo. The private project contains ML-Agents + Barracuda, `config/MoveToGoal.yaml`, a `venv/`, and `results/Test1…Test18` training runs, and the report's QA section documents "would progress, then regress." That is a real, honestly-abandoned RL effort. Steam: `steam_appid.txt` = `4340810`, matching the store URL; Steamworks.NET vendored and `SteamManager` referenced in three scenes. No liability found.

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | Delete or correct `README.md:55-58` — the fabricated report citation | 15 min |
| 2 | Rewrite `README.md:36-78` **and the GitHub repo description** to "NavMesh waypoint patrol with a distance-triggered, timer-gated chase". Remove every mention of coroutines, wander, and flee | 30 min |
| 3 | Fix `:60-61` (patrol is live) and `:70` (chase does return to patrol) | 15 min |
| 4 | **Promote the RL story to the headline** — "prototyped enemy AI with ML-Agents/PyTorch, 18 runs in `results/`, abandoned after reward regression; shipped a deterministic patrol/chase instead". A stronger junior story than the current one, and every word is backed by artifacts | 30 min |
| 5 | Own the `MoveTowards(…, 100f)` / `LookAt` redundancy as a known wart, or say nothing about chase internals | 15 min |

---

## 8. shorts-factory showcase vs private reality — NEEDS WORK

**The README is the best-behaved document in the portfolio** and should be the template for the others. Verified against the private repo: "119 modules / ~57k lines" → actual 120 / 57,469; "224 test files" → 226; "~21 configured accounts" → exactly 21. `README.md:21` explicitly *refuses* to quote a videos-per-month figure, and `:83-100` states plainly that this is one operator, one machine, pre-monetization, no views or revenue. **Secrets clean** — 2 commits, 3 tracked files, no keys, no internal paths, no persona names.

The damage is in the two documents beside it.

| # | Location | Claim | Reality |
|---|---|---|---|
| 1 | `docs/architecture.html:748` (footer) and `:193` (badge) | "26 core modules · 40+ API endpoints · 6 YouTube · 4 TikTok · **Runs fully autonomous Sun–Thu**" | 120 modules, 75 Flask routes, 21 accounts. Only **4 of 21** have `auto_schedule: true`, and `core/autoschedule.py:4` records that TikTok requires per-post human consent. **Two public docs in the same repo state different numbers** — that is what a reviewer cites first. |
| 2 | `architecture.html:355` | module `core/wan_video.py`, `--wan` flag | No such file. Wan2.1 lives in `core/aivideo.py`, selected by config. |
| 3 | `config.schema.json:24-31` | `autopilot.daily_quota`, `autopilot.window` | Neither key exists. The window is **hard-coded** at `core/schedule_window.py:12-13`, which makes `README.md:97` ("the schedule is config, and it changes") false as written. |
| 4 | `config.schema.json:6` | `gemini.min_score: 75` | Real key is `gemini.hook_min_score`, live value **80**. `:34-51` also documents `dubbing`/`thumbnail_ab`/`competitor`/`playlists` blocks — none of those keys exist. |
| 5 | `README.md:55` | provider_score weights "cost 10%, latency 10%" | `core/provider_score.py:35-41` says latency **5%**. The published weights sum to 100%; the real table sums to **0.95**. A clean-looking number that the code does not produce. |
| 6 | `README.md:38` | "FLUX.1-dev **fp8**" | `core/aiimage.py:68` defaults to **nf4** and config sets no override. fp8 exists as a code path but is not what runs. Classic written-not-operated. |
| 7 | `README.md:64` | "Top 3 videos **auto-dubbed** … published to dedicated language channels" | `core/dubbing.py:246 dub_top_videos` has **no caller in `run_autopilot.py`** — only the dashboard and a CLI. It lands dubs in the library, not on a channel. |
| 8 | `README.md:3,19,42` | "running unattended" | Never states TikTok is **drafts-only**. `core/publishers/tiktok.py:9,15,272,390` + `config.json tiktok.mode = "draft"`: unaudited apps hit the inbox/draft endpoint and the owner finishes the post manually. The diagram admits this (`architecture.html:660,728`); the README, which is what gets read, does not. |
| 9 | `architecture.html:698` | `"exclude_themes": ["zodiac_ai"]` | Live value is seven themes. Autopilot covers much less than implied. |
| 10 | `architecture.html:366,698,704` | — | Exposes internal theme id `zodiac_ai` and a stale/fabricated account id `facts-yt`. No credential leakage. |
| 11 | working tree | — | `shorts-factory-showcase/audit video/` holds three screen recordings including `tiktok-audit-demo.mp4`. Untracked (`?? "audit video/"`) — but the repo has **no `.gitignore` at all**, so one `git add .` publishes TikTok account capture to a public repo. |

### Fix order (~5 h)

1. **0.5 h** — Move `audit video/` out of the repo directory, or add a `.gitignore`. Only item with real downside. Do it first.
2. **1 h** — Rewrite `architecture.html:193,748`: drop "fully autonomous", correct to 21 accounts / 120 modules / 75 endpoints.
3. **1 h** — Regenerate `config.schema.json` from the real key set; note that the window is hard-coded.
4. **0.5 h** — Delete/rename the `wan_video.py` card (`architecture.html:344-360`).
5. **0.5 h** — `README.md:64` → "operator-triggered dubbing (CLI + dashboard)".
6. **0.5 h** — Add to `README.md:19/42`: "TikTok is sandbox drafts-only pending app audit; YouTube uploads go up `private` with a `publishAt`." **This strengthens the candidate** — it reads as ToS discipline.
7. **0.25 h each** — `README.md:55` latency → 5%; `README.md:38` → nf4 (and add *why* — fp8 + offload thrashed at 24 GB is a genuinely good engineering anecdote); add "4 of 21 accounts are on auto-schedule"; scrub `zodiac_ai`/`facts-yt`.

---

## 9. Profile README + larmstrong1127.github.io — NEEDS WORK

**Every hard number survived audit.** All 17 outbound links return 200. The live site is byte-identical to local HEAD. asset-provenance (134 tests, 1,791 assets / 934 derivations, 1,943 scanned, 110.0 s, 50/1,791 C2PA), comfy-workflow-pack (118 tests, ComfyUI 0.32.0), medinsight-api (40 tests + the honest "never run against live AWS" caveat), DentaVision (20 Jest), and EvalForge (0.7026 / 0.6009 / N=1,987 / 184M / 435M) all verify against the repos on disk and against the corrected HF card. Neither surface claims star counts — correctly, since the max across all repos is 1.

The exposure is entirely in unmaintained edges.

- **P1 — The DentaVision "live demo" is functionally dead but returns 200.** The Vercel frontend loads, so a link-checker sees green; `POST /api/auth/clinic/login` returns **500 with a raw Mongoose buffering-timeout string**. A reviewer clicking "Live demo" gets a login form that 500s with database internals. Worse than no demo. Occurrences: `Larmstrong1127/README.md:45` (heading `*(deployed — try the live demo)*`), `:47`, `:51` (badge); `larmstrong1127.github.io/index.html:496,507,511`. Also `dentavision-v3/client/.env.example:1` points at `dentavision-api.onrender.com`, a **deleted** service.
- **P2 — `portfolio-website` is a public, stale, competing portfolio.** Unhosted (`has_pages: false`) but fully browsable from the profile, and it contradicts the corrected narrative: `index.html:888` describes Echoed Nights as *"enemy finite state machine architecture"* — **the exact overclaim already retracted, still public**. `:908` links `Game-Library-Website`, which is **private** → 404 for every visitor. `:728,881` re-advertise the dead demo. `README.md:8` gives the dead `@stmartin.edu` address. Its project list has no EvalForge, no asset-provenance, no MedInsight — it presents a .NET MVC coursework candidate, the opposite of the current positioning.
- **P3 — The resume PDF on the site is ~2 months stale and omits both lead projects.** `index.html:279,313` link `assets/Landon_Armstrong_Resume.pdf`, last committed **2026-06-10**; the source at `D:\ClaudeProjects\resume\` was rebuilt **2026-08-01** with a different md5. Worse, `build_resume.py` (the base variant, which is what ships) contains **no EvalForge, no asset-provenance, no comfy-workflow-pack**. The page leads with a reward model and a provenance registry, and the resume it hands you mentions neither.
- **P4 — The site repo README describes a website that no longer exists.** `larmstrong1127.github.io/README.md:27-33` advertises a typewriter hero, a filterable project grid, scroll-triggered animations, and hamburger navigation. The current `index.html` has **one** `<script>` (`:670`) that sets the copyright year. It is verbatim-identical leftover copy from `portfolio-website/README.md`.
- Minor: dead `.featured-media` CSS (`index.html:121-125`); the two surfaces present different project sets (profile lists DocuChat/TechCon/A\*, site lists Shorts Factory).

### Fix order

| # | Fix | Effort |
|---|---|---|
| 1 | Relabel DentaVision on both surfaces — drop "live demo"/"deployed", keep screenshots + source (4 edits) | 10 min |
| 2 | **Archive or privatize `portfolio-website`** — unhosted, so nothing breaks; removes the retracted FSM claim and the 404 in one action | 5 min |
| 3 | Rebuild the base resume with EvalForge + asset-provenance; copy the fresh PDF into `assets/` | 45–60 min |
| 4 | Rewrite `larmstrong1127.github.io/README.md` to describe the actual page | 15 min |
| 5 | *(Alternative to #1, higher value)* Restore a free-tier Atlas cluster + seeded demo login, then restore the "live demo" label honestly | 1–2 h |
| 6 | Friendly 500 handler in `dentavision-v3/server/app.js` so outages never leak Mongoose internals | 20 min |
| 7 | Update `client/.env.example:1`; delete dead CSS; reconcile the two project lists | 15 min |

---

## Top 5 fixes across the portfolio, ranked by hiring-impact ÷ effort

**1. Remove `|| true` from DocuChat's CI, and `|| echo` / `|| true` from AgentForge's. — 30 minutes, both repos.**
Two public repos display green CI badges over, respectively, zero tests and a build that fails. This is the only finding in the audit that a reviewer could read as a character signal rather than an experience gap, and it is the cheapest thing on this list. A badge is a claim; right now both claims are false. If the honest state is "no tests yet", delete the badge — that costs nothing and reads as integrity.

**2. Fix AgentForge's build and rewrite its repo description — or archive the repo. — 2 hours, or 5 minutes.**
The description advertises Gemini support that does not exist in any file. The build fails. Both are visible from the profile without cloning. Item 1 of the AgentForge fix list (move model-ID arrays into a plain `lib/models.ts`) is a one-hour change that turns "does not build" into "builds." If that hour is not available this week, archiving is strictly better than the status quo.

**3. Correct the Echoed Nights README — starting with the fabricated report citation. — 1 hour.**
`README.md:55-58` attributes a four-state FSM design to a capstone report that contains none of those words, and the PDF is in the same repo. Everything else in that README overstates a dead comment block as shipped behaviour. Then spend 30 of those minutes *promoting* the ML-Agents work, which is real, evidenced by 18 training runs on disk, and a materially better story than the one currently told.

**4. Relabel the dead DentaVision demo and archive `portfolio-website`. — 15 minutes total.**
The "live demo" link hands a reviewer a 500 with a Mongoose stack-trace string. The stale portfolio still carries the Echoed Nights FSM claim that was already retracted elsewhere. Both are trivially reversible, and both are exactly the kind of thing that gets noticed in the first five minutes of a profile skim rather than the fortieth.

**5. Fix the DeBERTa judge encoding in evalforge, re-run the benchmark, and correct the stale test counts. — 3 hours.**
This is the highest-*ceiling* item on the list. The candidate already did the hard part: diagnosed the defect, quantified the fix (ROC-AUC 0.444 → 0.603), and published it. Landing the ten-line change converts the portfolio's central story from *"I found a subtle measurement bug"* into *"I found it, fixed it, and re-measured"* — which is the difference between a good disclosure and a demonstrated engineering loop. Bundle the 20-minute README count correction and the B4 identity fix into the same session.

---

## Three things I would NOT bother fixing

**1. Framework version bumps on the coursework repos (`netcoreapp3.1` → net8, `net6.0` → net8, renaming `LandonWebsite06`). ~2 hours, near-zero return.**
Nobody is hiring or rejecting on the target framework of a clearly-labeled 2021 class project. The READMEs state the versions honestly, so there is no claim mismatch — the only defect is age, and age is not a defect in coursework. These hours belong on AgentForge. The one exception worth 15 minutes is the A-Star README's "(Manhattan, Euclidean)" claim, which *is* a claim mismatch and should be deleted.

**2. Adding authentication to DocuChat, and building out the dashboard/component test coverage in evalforge. ~8 hours combined.**
For DocuChat, one honest README sentence — "single-user local demo; no auth, no tenancy" — closes the gap for 15 minutes of work and is *more* impressive than a bolted-on auth layer, because scoping judgment is the scarcer skill. For evalforge, 10 dashboard tests against 14 components is thin, but the backend is at 156 tests with `mypy --strict`, and no reviewer forms an opinion on a portfolio project's frontend coverage ratio. Both are real gaps; neither is a *hiring* gap.

**3. Rehabilitating WAVets2Tech into a repo that supports the "led a team of 4" claim. ~15 hours.**
This is the one I feel strongly about. The problem is not fixable by writing code, because the deficiency is in the *history* — 9 solo zip-dump commits with no branches or PRs — and that history cannot be honestly reconstructed. Building a real job-listings screen in 2026 does not make the 2022 git log show four contributors. Make it private, drop it from the profile, and keep the team-lead experience on the résumé as a described responsibility rather than a linked artifact. Thirty minutes of rewording beats fifteen hours of code that still would not prove the claim.

---

## The single thing I would raise in the debrief

Not any individual bug. This:

**Across eleven surfaces, the candidate's documentation is more advanced than the candidate's code — and in several places the documentation is describing code that does not run.**

The disclosures here are genuinely exceptional for a junior. "51% of examples had the answer truncated off before the model saw it, and fixing the encoding still does not beat the majority-class baseline" is a sentence most *senior* engineers would not volunteer. The instinct is right, the analysis is right, and it is the strongest evidence in the whole application that this person has real engineering judgment despite zero professional years.

But every one of those disclosures currently terminates in prose. The broken encoding is still in the serving path. The flee coroutine is still commented out under a README that defends it. The fp8 config is still documented while nf4 is what runs. The Quickstart still opens with a Docker command that has never been executed. The CI badges are green because failure was routed around rather than fixed.

The question that produces in a debrief is: *does this person ship, or does this person write?* Right now the artifacts answer "writes" more clearly than they answer "ships" — and for a candidate whose entire case rests on self-directed work substituting for professional experience, that is the wrong answer to be strongest on. The remedy is small and mostly measured in hours, not weeks: close the loop on three or four of the defects that are already diagnosed, and let the write-up end with "fixed, re-measured, here is the new number." That single change in shape does more for this portfolio than any new project would.
