# Portfolio Honesty Re-Audit — 2026-08-16

**Scope:** every public-facing surface — the personal site (live at
`larmstrong1127.github.io`), the GitHub profile README, all public repo
descriptions and READMEs, the published reward-model card, and the resume PDF
the site links from its own navigation.

**Doctrine applied:** every public claim must be classifiable as
*verified-operated*, *written-not-operated*, *stale*, or *unverifiable*.
Overclaiming one thing retroactively kills the credible claims — and this
portfolio's credible claims are unusually strong, which is exactly what raises
the cost of the weak ones.

**Method:** claims were checked against code, committed artifacts, live HTTP
responses, and GitHub's run/PR history. Test suites were executed from each
repo's own venv. CPU only; no GPU was used.

**Predecessor:** `docs/portfolio-hm-audit-2026-08-11.md` (5 days prior). Items
that audit raised are marked **[still open since 08-11]** or **[fixed since
08-11]** where relevant.

**Nothing was fixed as part of this audit.** This document is the only file
written.

---

## Headline

Three findings would cost real credibility if a reviewer found them first, and
all three share one shape: **a correction was made on one surface and not swept
to the others, or a capability was written and then described in the operating
present tense.**

1. **Echoed Nights describes its shipped enemy AI backwards** — on three
   surfaces at once. The behaviour singled out as "the part I'd defend hardest"
   is inside a `/* */` block.
2. **EvalForge's eval gate has never executed** — zero runs, in a repo with zero
   pull requests, while three surfaces describe it in the present tense.
3. **The resume PDF the site publishes is the May 30 build** and still carries
   all four of the known-bad overclaim patterns that were corrected everywhere
   else, including in three newer resume variants sitting in the same folder.

The pattern from the 08-11 audit — *documents defects beautifully, then does not
close them* — has a sibling worth naming: **fixes propagate along the path the
author was looking at, not along the path the reader travels.**

---

## Findings, ranked by damage-if-caught

| # | Surface | Exact claim (quoted) | Verdict | Evidence | Suggested replacement wording |
|---|---|---|---|---|---|
| 1 | Echoed Nights — profile README, repo description, repo README, site | "the monster wanders to random reachable points, chases when you cross a threshold, and — the part I'd defend hardest — **backs away when you get too close**" | **OVERCLAIM (inverted)** | `Assets/Scripts/Shadowlop_Scripts/Monster.cs`: live code is lines 1–116 — `attackRadius = 20f`, `FixedUpdate`, `BackToPath()` cycling a hand-placed `destinations[]` waypoint array. `grep -n "fleeDistance\|chaseDistance\|StartCoroutine\|/\*\|\*/" Monster.cs` → `chaseDistance` (282), `fleeDistance` (283), `StartCoroutine(Wander())` (292), `StartCoroutine(Chase())` (308) all fall between `/*` at line 276 and `*/` at line 356. Zero coroutines run. **No live code path ever moves the agent away from the player.** | "Enemy behaviour runs on Unity NavMesh: the monster patrols a fixed waypoint loop and commits to a chase only after you stay inside a 20-unit radius for three seconds — a deliberate delay, so the moment it notices you is legible. A coroutine rewrite with random-point wander and a flee-when-crowded branch is in the file but commented out; it did not make the release build." |
| 2 | EvalForge — profile README:20; site `index.html`:411; repo `README.md`:206-209 | profile: "**CI gates every merge on an eval-regression check.**" · site: "CI now runs a fixed suite against a pinned local model on **every pull request that touches the API**, and fails the build if scores collapse: the platform gates itself with itself." · repo README: "a fixed prompt suite **runs** against a pinned local model on every platform PR and fails the build on score collapse. The platform **gates its own CI with itself.**" | **OVERCLAIM (written, never operated)** | `.github/workflows/eval-gate.yml` triggers on `pull_request` only. `gh pr list --repo Larmstrong1127/evalforge --state all --limit 100 --jq 'length'` → **0**. `gh run list --limit 100 --jq '[.[].name]\|group_by(.)'` → `CI: 13, Train package check: 8, dependency graph: 1` — **no "Eval gate" runs**. `gh run list --workflow=eval-gate.yml` → `HTTP 404: workflow not found on the default branch` (the signature of never-executed). The file *is* pushed (`gh api .../contents/.github/workflows/eval-gate.yml` returns a sha) — it is written and correct, and has never fired, because merges do not go through PRs. | "An eval-regression gate is committed as a CI workflow: a fixed suite against a pinned local model, failing the build on score collapse. It is wired to `pull_request` and has not yet run — this repo's history is direct-to-master." *(Or make it true: add `push:` to the trigger, or open one PR.)* **Do not over-correct the neighbouring claim:** "lint/type/test on every PR" is fine — `ci.yml` also triggers on `push: [master]` and has **13 successful runs**. Only the eval gate has never fired. |
| 3 | Resume PDF — `assets/Landon_Armstrong_Resume.pdf`, linked from site nav and served live | "**Provisioned** containerized FastAPI web services on AWS ECS Fargate with infrastructure-as-code automation using Terraform, AWS ECR deployment setups, S3 asset buckets, and DynamoDB data states." | **OVERCLAIM (the known-bad AWS pattern, still live)** | Live file is byte-identical to the May 30 local build: `curl -sI https://larmstrong1127.github.io/assets/Landon_Armstrong_Resume.pdf` → `Content-Length: 104691`; local `assets/Landon_Armstrong_Resume.pdf` = 104,691 bytes, dated May 30. Every other surface states the opposite — site: "I have never run them against a live AWS account. The CD workflow has no successful run." Confirmed: `gh run list --repo Larmstrong1127/medinsight-api` → every CD run `failure` at 7–13s, zero successes. | "Defined AWS ECS Fargate infrastructure in four Terraform modules with an OIDC-authenticated GitHub Actions deploy pipeline; exercised the S3, DynamoDB and KMS paths against LocalStack rather than a live account." *(This exact wording already exists in `build_resume_aiml.py:179` — it simply was never swept into the default build or republished.)* |
| 4 | Resume PDF (same file) | "Shipped a live dental SaaS (Vercel + Render) with dual-role JWT auth, clinic dashboards, NPI validation, **Stripe billing**, and a GitHub Actions CI/CD pipeline" | **OVERCLAIM (the known-bad Stripe pattern, still live)** | Stripe is implemented (`server/routes/billing.js` — checkout sessions, billing portal, `webhooks.constructEvent`) but cannot charge: `server/render.yaml` declares **no `STRIPE_SECRET_KEY`**, so the deployed backend cannot reach Stripe. Every other surface hedges correctly; this one does not. | "…dual-role JWT auth, clinic dashboards, NPI validation, and a Stripe subscription integration with webhook-driven status sync (wired end to end, never processed a live payment); 20 Jest tests gate every deploy through GitHub Actions CI/CD." *(Already written verbatim at `build_resume_aiml.py:163`.)* |
| 5 | Resume PDF (same file) | "Built a LangGraph **multi-agent** pipeline with RAG retrieval, **AES-256 encryption**, and **HIPAA-compliant** audit logging" | **OVERCLAIM ×3** | (a) Single ReAct agent, not multi-agent: `app/agents/clinical_agent.py:97` `create_react_agent(...)`, exactly four `@tool` functions. (b) Encryption is **Fernet** (`app/core/security.py:5,28,34`) = AES-**128**-CBC + HMAC-SHA256, not AES-256. (c) "HIPAA-compliant" is a legal/certification claim no portfolio project can assert; nothing in the repo substantiates it, and the repo's own README scopes it honestly. | "Built a LangGraph ReAct agent with four tools over an encrypted document store — Fernet encryption at rest and an append-only audit trail; 40 pytest tests cover auth, encryption, and the audit trail end to end." |
| 6 | EvalForge — site, profile README:31, repo README:13 & :232, `MODEL_CARD_preference_reward.md`:69 | "`OpenAssistant/reward-model-deberta-v3-large-v2` \| 435M \| **0.6009**" — the baseline the whole "beats a public model 2.4× its size" headline rests on | **UNVERIFIABLE (no artifact on disk)** | My model's **0.7026** re-ran bit-for-bit on CPU (`eval_reward.py --checkpoint checkpoints/reward-lr2e5`, offline → `pairwise accuracy = 0.7026`, N=1987), corroborated by `training/runs/reward-lr2e5-512-stdout.log` (`eval_pairwise_acc=0.7026`, `eval dropped 1/1988`). **0.6009 appears only in prose** — README, model card, docs. `eval_reward_baseline.py` exists and is testable, but its output was never saved anywhere. Re-running needs a 1.7 GB download + ~1h CPU, so it was not re-derived here. Every other headline number has a committed artifact; this one does not — and it is the number that makes the comparison mean anything. | Don't change the wording — **produce the artifact.** Re-run `eval_reward_baseline.py` and commit a small `training/reward_results.json` capturing 0.7026 / 0.6009 / 0.5098 / T / N. Until then the honest hedge is: "0.6009 for the public baseline, measured under the same harness on the same split; the baseline run's output is not committed." |
| 7 | EvalForge — repo README:119-121 vs README:143-144 vs `MODEL_CARD_hallucination_judge.md`:84 | "**102 of the 200 examples (51%)** the ANSWER … is cut off entirely … a further **30 (15%)** partially cut" | **DRIFTED (self-contradiction, 20 lines apart)** | The README's own table at `:143-144` says `101/200 (50.5%)` and `31/200 (15.5%)`, and the model card at `:84` also says 50.5%. Neither pair is backed: `ragtruth_diagnostic.json`'s top-level (legacy) block has **no `answer_truncation` key at all** — only the `rerun_2026_08_13_fixed_encoding` block has one, and it correctly reads `n_answer_fully_dropped: 0` / `n_answer_partially_dropped: 0`, which does back the "0" column. | Change `:120` to "**101 of the 200 examples (50.5%)** … a further **31 (15.5%)**" to match the table and the card, and add the legacy `answer_truncation` counts to the JSON's top-level block so the number has an artifact. |
| 8 | EvalForge — repo README:143-150, `MODEL_CARD_hallucination_judge.md`:103-108 | legacy column: "F1 at the best-accuracy operating point \| **0.092**" and "best achievable accuracy \| **0.605**" | **UNVERIFIABLE (committed JSON disagrees)** | `ragtruth_diagnostic.json`'s legacy `best_accuracy` block reads `accuracy: 0.495, f1: 0.5388` — not 0.605 / 0.092. There is no top-level `best_accuracy_score_grid`; the fixed-encoding block has one, and it *does* back its 0.615 / 0.627 / 0.605 cells. So the legacy column's two score-grid cells were computed on a grid never written back. The rest of the legacy column verifies exactly (ROC-AUC 0.444, best-F1 0.566, acc@0.5 0.475, majority 0.610). | Re-emit the legacy block with its `best_accuracy_score_grid`, or footnote: "the legacy column's grid figures were recomputed at re-run time and are not in the committed top-level JSON." |
| 9 | EvalForge — repo README:197-201 | "**Tests:** 197 total — 95 backend …, **92 training** …, 10 frontend (vitest)" | **DRIFTED (understated by 14)** | Measured twice, independently: `platform/api` → **95 collected, 95 passed**; `training/.venv` → **106 collected**, 105 passed + 1 skipped; frontend **10**. Total is **211, not 197**. Cause: the 197/92 text landed in `1b761ca` and `test_run_rewardbench2.py` arrived after it in `4842430`. | "**Tests:** 211 total — 95 backend (pytest, …), 106 training (…), 10 frontend (vitest)." |
| 10 | EvalForge — repo README:88-90, `MODEL_CARD_hallucination_judge.md`:55 | OOD "F1 **0.5067**" and ECE "**0.4010** / 0.6163 / 0.5766" | **UNVERIFIABLE** | `grep -rn "0.5067\|0.4010\|0.4735\|0.4814" --include=*.json --include=*.csv --include=*.log .` → zero hits outside prose. The RAGTruth OOD evaluation output was never persisted. The three *in-distribution* F1s in the same table (0.9937 / 0.9942 / 0.9937) **do** verify, from TensorBoard `metrics/val_f1` scalars. This is the numbers-load-bearing claim behind the README's entire generalization-gap narrative. | Re-run the OOD eval and commit its output, or footnote the row: "OOD figures from the 2026-08-09 run; the evaluation output was not persisted." |
| 11 | WAVets2Tech — site (project card), repo description, repo README | site: "with a **React client** hosted inside the same project, covering job listings, veteran profiles, and an employer directory"; description: "**React SPA** + ASP.NET Core Web API" | **OVERCLAIM** | The entire React layer is three files (`ClientApp/src/{App.js,index.js,setupProxy.js}`). `App.js` renders literally `<h1> THIS IS TEST CODE </h1>` three times over a raw `fetch("api/company")` dump. There is no job listing, veteran profile, or employer directory UI. **[still open since 08-11]** | "An ASP.NET Core Web API over SQL Server with Entity Framework Core — six controllers and seventeen EF Core models covering job listings, veteran profiles, and an employer directory. The work is entirely the C# side; the bundled React client never got past a scaffold and is not part of the delivered value." |
| 12 | WAVets2Tech — site, repo description, repo README | "**Led a team of four** through the full SDLC — requirements through client delivery." | **UNVERIFIABLE (self-asserted only)** | Public repo has exactly **one** commit: `af7d13e Add professional README`. No co-contributors, no branches, no PRs, no sprint artifacts. The repo does contain a client DB diagram PDF, a pitch deck, and a promo flyer, which support *client delivery* but say nothing about team size or role. The claim is restated in three places and substantiated in none. **[still open since 08-11]** | Keep the substance, drop the process framing that invites an artifact request: "Team lead and front-end developer on a four-person client delivery for the Saint Martin's WAVets2Tech program." |
| 13 | DentaVision — profile README, site, resume | profile: "*(deployed — demo database being restored)*" + a **Live Demo** badge; site: "a **deployed** dental SaaS"; repo README publishes demo credentials | **DRIFTED (hedge accurate, presentation is not)** | Frontend and API are up — `curl -sI https://denta-vision.vercel.app` → 200; `/api/health` → `{"status":"ok"}`. But the published demo credentials fail: `POST /api/auth/clinic/login` → HTTP 500 `{"error":"Operation \`clinics.findOne()\` buffering timed out after 10000ms"}`. A reviewer who clicks the badge and pastes the README credentials sees a raw Mongoose stack string — that reads as a broken app, not a paused demo. **[still open since 08-11]** | profile: "*(frontend + API live; demo database offline, so login is currently unavailable)*", and put the same one-line notice directly above the credentials in the repo README (or remove them until the DB is back). |
| 14 | dividend-desk — repo README:27 | "Fix commit: **`5b5a063`**" | **UNVERIFIABLE (broken reference)** | `git log --all --oneline` in the repo → exactly one commit, `1423b3d`. `gh api repos/Larmstrong1127/dividend-desk/commits/5b5a063` → `422 No commit found for SHA: 5b5a063`. History was squashed at publish, so the anchor of the repo's entire before/after narrative points at a commit no reader can open. | "The fix is squashed into the initial public commit `1423b3d`; the pre-fix state is not in this repo's history — the before/after numbers come from `docs/committee_scorecard.md`, which is committed." |
| 15 | EvalForge — site index.html (judge table) | "Hallucination judge vs. cloud judges — 200 RAGTruth examples … local DeBERTa (mine) **47.5%**" with the note "Generated from the benchmark's own results file, not typed by hand" | **STALE (correct number, retired protocol, missing caveat)** | The number itself verifies exactly against `training/benchmark_results.json` (`agreement: 0.475`, `p50 26.7ms`, `cost 0.0`) — as do all three cloud rows (0.85/0.825/0.785, $4.06/$2.03/$0.08, 1466/1311/1182 ms). **But** the repo README:184-190 now states the 47.5% "**is left as-is, and is the legacy-encoding measurement**", after the 2026-08-13 fix (`69604c3 fix(judges): stop truncating the answer out of the hallucination judge input`). The site carries the number without the caveat and never mentions the encoding fix at all, so the site is one full generation behind the repo's own disclosure. | Add one line under the table: "These rows were measured under the judge's original encoding, which truncated the answer out of long examples. That bug is fixed in the repo; the row is left as measured because the paid cloud rows were not re-run. The comparison that matters after the fix is in the repo README." |
| 16 | Uber-Pickup-Data-Analysis — repo README | "Borough-level demand comparison", "**Geographic heatmaps** of pickup density", "Peak hour identification **by neighborhood**", libraries incl. "**ggmap**" | **OVERCLAIM** | `2014UberPickups.Rmd` never loads `ggmap`; grep for `borough|neighborhood` → zero matches. All three heatmaps are temporal (hour×day, month×day, weekday×month). The only spatial output is one lat/lon scatter. | "Hourly, daily and day-of-week pickup volume trends; month-over-month growth across Apr–Sep 2014; heatmaps of trip volume by hour × day and month × day; a point map of every pickup's lat/lon across the NYC bounding box." Libraries: `ggplot2, ggthemes, dplyr, lubridate, tidyr, scales, DT`. |
| 17 | Thurston-County — repo description + README | description: "Geospatial and **statistical** analysis"; README: "**Spatial correlation** between parks and wetland zones", "Park coverage **by area**", libraries incl. "**ggmap**" | **OVERCLAIM** | `FinalProject.Rmd`: grep for `cor(\|correl\|summary(\|lm(\|st_intersect\|area\|classif` → **zero matches**. The file is three `st_read()` calls plus `tm_shape() + tm_polygons()` overlays. No statistics of any kind are computed; `ggmap` is not loaded. (Also hardcodes an absolute OneDrive path, so it will not knit for a cloner.) | description: "Geospatial mapping of Thurston County, WA wetlands and parks in R — reads county GIS layers with sf and renders them as tmap overlays on OpenStreetMap basemaps." README: list the three maps, then "This is a mapping exercise, not a statistical one — no areas, correlations, or summary statistics are computed." |
| 18 | A-Star — repo description + profile README table | "performance **benchmarks across grid sizes**" | **OVERCLAIM** (the repo's own README is already honest and contradicts them) | No benchmark artifact, results file, or committed output exists in the repo. The only timing test is `FindPath_PerformanceBenchmark_CompletesInTime` — one 50×50 all-open grid asserting completion under 1000 ms. One size, a pass/fail bound, not a benchmark sweep. | description: "A* pathfinding in C# with configurable Manhattan/Euclidean heuristics and grid obstacle support, plus a 10-case xUnit suite including a 50×50 timing bound." profile row: "**A\* Pathfinding** — heuristic search with a 10-case xUnit suite". |
| 19 | Doubly-Linked-List — repo README | "insertion at head, tail, **and arbitrary positions**", "deletion by value **and position**", "**Search and retrieval** operations", "reversal **and length calculation**" | **OVERCLAIM** | `CSC515/DoublyLinkedList.cs` public surface is exactly `AddFirst, AddLast, DeleteFirst, DeleteLast, DeleteValue, DeleteNode(Node<T>), Reverse, IsEmpty, Clear, Print`. Grep for `public .*(Insert\|Find\|Search\|Count\|Length\|Get\|Contains\|At)` → no matches. No insert-at-position, no delete-by-index, no search, no length. | "Insertion at head and tail; deletion at head, tail, by value, and by node reference; in-place reversal; empty check, clear, and forward traversal printing; generic over `T`." |
| 20 | docuchat — repo README:68, :84 | "│ Claude (**claude-3-5-haiku**) │" and "Anthropic Claude (**Haiku 3.5**)" | **DRIFTED** | `backend/rag.py:271` → `model="claude-haiku-4-5"`. The code calls Haiku **4.5**. The stale string is also in the code's own docstring at `backend/rag.py:230`. | README:68 → `│ Claude (claude-haiku-4-5) │`; README:84 → "Anthropic Claude (Haiku 4.5)"; fix the `rag.py:230` docstring too. |
| 21 | asset-provenance — profile README + site | profile: "**134 tests**"; site: "**134 + 118** tests between the two" | **DRIFTED (understated)** | `.venv/Scripts/python.exe -m pytest -q` → **144 passed** in 44.40s (collect-only agrees: 144). The repo's own README already says 144 at lines 59 and 602 — only the two outward-facing surfaces are stale. comfy-workflow-pack's 118 is exact (118 collected, 118 passed). | profile: "144 tests"; site: "144 + 118 tests between the two". |
| 22 | dividend-desk — repo README:8 | "`tools/doctor.py` **asserts** the flag as a health check" | **OVERCLAIM (minor)** | `tools/doctor.py:94` emits `OK if DRY_RUN else WARN` — a **warning**, not a failure; `doctor.py:141-148` exits non-zero only on `FAIL` counts. Doctor would not fail if `DRY_RUN` were flipped. (The core safety claim is sound: `engine.py:33` `DRY_RUN = True`, and the Robinhood execution path raises `NotImplementedError` at `engine.py:292`.) | "`tools/doctor.py` reports the flag as a health check, warning if it is off" — or raise it to `FAIL` in code and keep the stronger wording. |
| 23 | dividend-desk — repo README:106, :116, :131, :65 | source line citations "line 39", "line 185" (×2), "line 272" | **DRIFTED (off by one)** | Actual: `src/analysis/committee.py:38` `ESCALATION_MIN_CONFIDENCE = 75`; `:184` `def _prior_verdicts_block`; `:271` `def _sanitize_headline`. | 39 → 38; 185 → 184 (both occurrences); 272 → 271. |
| 24 | Echoed Nights — profile README | "The **19-page** capstone report is in the repo." | **DRIFTED** | The repo PDF `MSCS-ProjectReport-Landon.pdf` has `/Count 18` and 18 `/Type /Page` objects. | "The 18-page capstone report is in the repo" — or drop the number, which buys nothing and is a free unforced error. |
| 25 | shorts-factory — public showcase README | four features in the operating present tense, e.g. "registers variant B at +48h, **measures CTR delta, auto-swaps winner**"; "**pulls** audience watch-ratio curves"; "Top 3 videos **auto-dubbed** … **published** to dedicated language channels"; "**weekly** YouTube Data API scan" | **UNVERIFIABLE from the public surface** | The source is private by design, so a reader cannot check any of these. The README is otherwise the best-hedged surface in the portfolio — it declines to quote a videos-per-month figure "because there is no committed artifact behind one", stamps its counts as a dated snapshot, and states plainly that the channels are pre-monetization with nothing to report. These four claims are the ones that would land hardest if the honest answer is "built and wired, but that loop's cadence depends on API auth state." | One sentence in "Scale — stated honestly" immunizes all four: "The closed-loop features below are implemented and wired into the pipeline; this README makes no claim about how recently any of them last ran." |
| 26 | agentforge — repo README:3 | CI badge (no accompanying test claim) | **MISLEADING BY OMISSION (minor)** | `.github/workflows/ci.yml` has no test step (checkout → prisma generate → tsc → lint → build), and the repo has no test files at all. The README claims no test count, so nothing is false — but a green CI badge reads as "tests pass" to most viewers. **[fixed since 08-11: CI no longer swallows lint/build failures — `feb1e2d`, `4618c39`; latest run is a genuine success]** | Add one line under the badge: "CI type-checks, lints and builds; there is no automated test suite." |
| 27 | EvalForge — repo README:174 | "Free and **43x** faster" | **DRIFTED (understated — no derivation found)** | No arithmetic over `benchmark_results.json` yields 43. Local p50 is 26.71 ms; the ratios are gemini **44.2×**, gpt-4o **49.1×**, claude **54.9×**, mean-of-cloud **49.4×**. The claim is conservative against every candidate, so it costs nothing but credibility-by-precision — a reviewer who recomputes gets a different number than the one printed. | "Free and **44x** faster than the quickest cloud judge in the table (and ~55x vs the slowest)" — or state the basis: "43x" → "44x faster than `gemini-2.5-flash-lite`, the fastest paid judge measured". |
| 28 | EvalForge — repo README:21-22 | architecture diagram: judges "`exact_match · llm_judge · deberta-hallucination (ours)`"; providers "`claude · openai · gemini · ollama`"; `training/` box "fine-tuned the DeBERTa judge" | **DRIFTED (cosmetic, but it omits the flagship)** | The **`reward` judge is missing from the diagram** although it is registered (`judges/__init__.py:56-61`), reachable, and the subject of the README's own headline at `:13` and a full section at `:218+`. The provider key is **`anthropic`**, not `claude` — README:58 itself uses `anthropic:claude-sonnet-5`. **[still open since 08-11 — this was item E3 in that audit]** | judges → `exact_match · llm_judge · deberta-hallucination (ours) · reward (ours)`; providers → `anthropic · openai · gemini · ollama`; training box → "fine-tuned the DeBERTa judge and the preference reward model". |
| 29 | medinsight-api — repo README:12-15 | ASCII architecture diagram drawing "GitHub Actions CI/CD → ECS deploy" | **MISLEADING BY LAYOUT (minor)** | Not false, and `README.md:131-136` carries an explicit "written, never run against a live account… the CD workflow has no successful run" block — but that block sits ~120 lines below the diagram, and a skimmer reads the diagram as operated. | Annotate the diagram's top box: `GitHub Actions CI/CD → ECS deploy (never applied — see "AWS deployment" below)`. |

---

## Verified — claims that survived the audit

These are load-bearing and now independently re-confirmed. They are the reason
the items above matter.

| Surface | Claim | Evidence |
|---|---|---|
| Site + profile + model card | Reward model **0.7026** vs baseline **0.6009**, both on the same **N=1,987** UltraFeedback `test_prefs` split, 184M vs 435M | `training/MODEL_CARD_preference_reward.md:61,69,71`; collapsed-run row 0.5098 present as disclosed |
| Site + profile | RewardBench 2: mine **25.3** vs baseline **32.0**, 1,865 prompts, best-of-4, 25% floor, Math **47.1 vs 50.3** | `training/rewardbench2_results.json` → `average_unweighted_6_domain: 0.2533`, `Math: 0.4713`, OA `0.32`, `random_baseline: 0.25`. Run on **CPU** (`torch 2.13.0+cpu`), 1h18m |
| Site (judge table) | 47.5 / 85.0 / 82.5 / 78.5 %; $0.00 / $4.06 / $2.03 / $0.08; 27 / 1466 / 1311 / 1182 ms | Exact match to `training/benchmark_results.json`. The "generated, not typed by hand" note is true. *(See finding #15 for the missing protocol caveat.)* |
| Site | Hallucination judge "F1 0.99 in-distribution and 0.51 on RAGTruth" | `MODEL_CARD_hallucination_judge.md:54` F1 0.9937, ECE 0.0044; `:62,:74` F1 ≈ 0.51, ECE ≈ 0.40 |
| Both HF models "published" | `deberta-preference-reward`, `deberta-hallucination-judge` | Both HTTP 200 and public via the HF API (45 and 5 downloads) |
| asset-provenance | 1,943 files → **1,791 assets**, **934 derivations**, **152** duplicate bytes; **0** recoverable models, 60 seeds, 48 prompts, 61 durations; **50 of 1,791** carry C2PA, 0 parse errors, signature validation recorded as not-attempted | All confirmed by direct query against `provenance.db`; per-edge-type counts match the README exactly. C2PA reader is a real 452-line stdlib implementation (JPEG APP11 / PNG caBX / BMFF uuid), not a stub |
| asset-provenance | SHA-256 identity, DAG lineage, OpenUSD export with provenance in `customData` and lineage as USD relationships | `usd_export.py:124,145` using the real `pxr` API; confirmed in the exported stage (1,791 `def Scope` prims) |
| asset-provenance | `apr browse` and `apr demo`, "install to lineage view in under two minutes" | Both subcommands wired (`cli.py:448,454`); demo built 20 assets/18 derivations in **2.1s**; browse served HTTP 200 + JSON API |
| comfy-workflow-pack | "Executed end-to-end against **ComfyUI 0.32.0**" | Install at `D:\Tools\ComfyUI` is `0.32.0` commit `27bca65`; `docs/live-run.md` cites the same commit; `docs/verify_live_run.py:99,118,120-121` genuinely registers and re-opens the USD stage. 118 tests collected, 118 passed |
| medinsight-api | LangGraph ReAct agent with **four** tools; Fernet at rest; append-only audit; **four** Terraform modules; OIDC pipeline never run live; **40** pytest tests | `clinical_agent.py:4,97` + exactly 4 `@tool` fns; `modules/` = `ecs networking security storage`; every CD run `failure` at 7–13s, zero successes; `pytest --collect-only -q` → 40 |
| DentaVision | **20** Jest tests gating deploys; Stripe wired but not charging | 19 + 1 = 20 test cases; CI gates the deploy job and has 4 genuine historical failures, i.e. the gate has actually bitten; `render.yaml` declares no `STRIPE_SECRET_KEY` |
| docuchat | **54** tests; FastAPI + ChromaDB + Sentence Transformers + Claude all genuinely used; chunking 600/100; cosine space | `pytest --collect-only -q` → 54; `rag.py:14-18,29,47,57-58`; CI runs green **[fixed since 08-11: CI now fails on test failure — `ee4322f`, and a real suite exists — `ac6f116`]** |
| agentforge | Next.js **16.2.6**, Prisma **7.8.0**, real tool-calling loop, real SSE | `package.json`; `app/api/chat/route.ts:29,170` SSE; `:44,:101` per-provider `while (continueLoop)` loops; README already de-scopes memory honestly at `:20` |
| dividend-desk | `DRY_RUN = True` hard lock; Robinhood path raises; **206** tests; every before/after number | `engine.py:33,292-294`; all figures match `docs/committee_scorecard.md` exactly; 206 collected |
| Echoed Nights | Steam page live; Unity 2022.3 | `curl -sI` → 200 on the real app page; `ProjectVersion.txt` → `2022.3.18f1` |
| TechCon | ASP.NET Core 3.1 / EF Core 3.1 / SQLite | `.csproj` → `netcoreapp3.1`, `EntityFrameworkCore.Sqlite 3.1.0` — exact |
| WAVets2Tech (site) | "**six controllers and seventeen EF Core models**"; "The weight of the code is the C# side" | Exact. 7 `*Controller.cs` minus the scaffolded `WeatherForecastController` = **6**; 18 files under `Models/` minus `Wavets2TechContext.cs` = **17** entities. Stack is .NET 6 + EF Core 7.0.4 + SQL Server (no version is claimed on any surface, so no drift). The "weight is the C# side" hedge is correct — see finding #11 for the React half |
| Site + profile, generally | The honest-caveat framing throughout | The site's LocalStack disclosure, the "never processed a live payment" hedge, the human-OOD 0.4000 row, the RewardBench floor, and the C2PA write-up's "I did not validate a single one" are all accurate and all verified. This is the portfolio's strongest asset. |

**[fixed since 08-11]** also: `portfolio-website`, flagged as a LIABILITY for
carrying the retracted Echoed Nights FSM claim, is now **private**. The
evalforge README's stale Quality counts (E2) were corrected, and the truncating
judge encoding (E1) was actually fixed and re-measured — the single biggest item
from that audit is closed.

---

## Two structural gaps worth knowing (not overclaims)

1. **The artifacts that prove the flagship numbers are gitignored.**
   `asset-provenance` excludes `*.db` and `*.usda`; `comfy-workflow-pack`
   excludes `live_run.{db,usda,jsonl}` and those files no longer exist on disk.
   Every headline number verifies against local state, but **a third party who
   clones either repo cannot reproduce any of them.** The `110.0s` wall clock
   and `13.9 GB` figures have no backing artifact of any kind — they exist only
   as a pasted transcript. Consider committing a small redacted export, or say
   plainly "timing from the run transcript; the registry DB is not committed."

2. **EvalForge has the same problem, and it is the one repo where it matters
   most.** `.gitignore:15-16` excludes `training/checkpoints/` and
   `training/runs/`, and `training/data/` is untracked. The artifacts used to
   verify the temperature, the hyperparameters, the collapsed-run number, the F1
   sweep and the 15-vote probe **exist only on this machine.** A reader who
   clones the public repo can verify exactly three files —
   `benchmark_results.json`, `ragtruth_diagnostic.json`, and
   `rewardbench2_results.json` — and nothing else. Cheapest fix with no repo
   bloat: commit the two 2-line training stdout logs, the 15-line
   `data/rating_pairs.jsonl`, and a small `training/reward_results.json`
   capturing 0.7026 / 0.6009 / 0.5098 / T / N. That single file would convert
   findings #6 and #10 from UNVERIFIABLE to backed.

3. **Two "historical" claims are uncheckable by construction.** dividend-desk's
   "140 tests passed the entire time" describes a pre-squash state absent from
   the repo, and the same squash is what breaks the `5b5a063` reference in
   finding #14.

---

## Recommended fix order

Ranked by damage-if-caught per unit of effort. All are wording changes except
where noted.

| # | Action | Effort |
|---|---|---|
| 1 | Rewrite the Echoed Nights AI description on all three surfaces to match `Monster.cs` (finding #1) | 30 min |
| 2 | Rebuild and republish the resume PDF from the corrected variant, then diff every resume variant against each other (findings #3, #4, #5) | 30 min |
| 3 | Either add `push:` to the eval-gate trigger and let it run once, or restate the claim in the past/conditional on all three surfaces (finding #2) | 20 min / 1 h |
| 4 | Fix the dividend-desk phantom commit reference (finding #14) | 5 min |
| 5 | Re-scope the WAVets2Tech React and team claims (findings #11, #12) | 15 min |
| 6 | Add the demo-database notice above the DentaVision credentials and soften the badge (finding #13) | 10 min |
| 7 | Add the legacy-encoding caveat under the site's judge table (finding #15) | 10 min |
| 8 | Sweep the small numeric drifts: 134→144, 19→18 pages, Haiku 3.5→4.5, four line numbers, "asserts"→"warns", 43x→44x (findings #20–#24, #27) | 25 min |
| 9 | Re-scope the three R/data-structures coursework READMEs (findings #16, #17, #19) and the A* benchmark claim (#18) | 30 min |
| 10 | Add the one-line hedges to shorts-factory, agentforge, and the medinsight diagram (findings #25, #26, #28) | 15 min |

---

## The generalizable lesson

The 08-11 audit named the failure mode as *documents defects but does not close
them*. This audit found a second, more dangerous one, and it explains findings
#1, #3, #4, #5 and #21 at once:

> **A correction applied to one surface is unapplied everywhere it was not
> explicitly swept — and the surface the author edits is rarely the surface the
> reader reaches first.**

The Echoed Nights correction fixed the wrong half and inverted the claim. The
AWS and Stripe corrections were written perfectly — into three resume variants
that are not the one the website serves. The test count was updated in the
repo's own README and not on either outward-facing surface. In every case the
author knew the true fact and had already written it down somewhere.

The operational fix is a claims inventory keyed on *(project, metric)* rather
than on *(file)*: any number or capability asserted in more than one place gets
compared across all of its places before publication. A claim that contradicts
another of the author's own claims is already falsified — no code required — and
that check is both the cheapest available and invisible to per-surface review.

The second operational fix, from finding #2: **existence, correctness, and
execution are three separate facts.** A workflow that is committed, correct, and
pushed can still never have run. Verify automation claims against the run
history and against whether the triggering event has ever occurred at all.
