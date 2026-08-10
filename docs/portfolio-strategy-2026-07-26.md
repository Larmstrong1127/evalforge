# Portfolio Strategy — Dividend Desk & EvalForge

**Prepared:** 2026-08-09 (filed under the 2026-07-26 audit series)
**Persona:** senior ML hiring manager / portfolio strategist
**Target role:** Airbnb — Data Scientist, Algorithms (Community Support). Flagship project: *automate the LLM evaluation process.*
**Candidate:** MS CS May 2024, no professional engineering experience, full-time non-engineering day job.

This is an analysis document. Nothing was implemented, pushed, or published in producing it.

---

# QUESTION 1 — Dividend Desk

Repo: `D:\ClaudeProjects\financial-agent` (folder name legacy; project is "Dividend Desk").

## Verdict: **(b) — worth 1–2 focused days, and the first day is almost entirely publishing and reframing, not coding.**

Not (a), because **it is not on GitHub at all** — `git remote -v` returns empty, and `gh repo list Larmstrong1127` shows no corresponding repo. 62 commits and ~9,900 lines of Python exist only on one Windows machine. Its current hiring impact is exactly zero. You cannot "surface" what isn't published.

Not (c), because the code is genuinely above the portfolio-project median and contains one artifact that is directly, almost uncannily on-target for the Airbnb role.

## What I found reading the code

**The committee is a real architecture, not three prompts in a trench coat.** I expected the latter. Specifics that changed my mind, all in `src/analysis/committee.py`:

- **Deterministic arbitration over the LLM.** `ESCALATION_MIN_CONFIDENCE = 75` (line 39) implements a non-LLM override: when Bull is POSITIVE, Bear is NEGATIVE, and the chair approved below 75 confidence, the approval is downgraded to REJECT in Python. That is a real control layer sitting above the model, not a prompt asking nicely.
- **Persistent, outcome-aware memory.** `_prior_verdicts_block()` (line 185) queries SQLite, joining `committee_decisions` against `decision_outcomes` to inject prior verdicts *along with the realized 30-day return that followed*. Memory that carries ground truth, not just chat history.
- **Fail-closed parsing.** JSON-schema-constrained outputs with regex fallback, last-match-wins so instruction echo can't trigger a false approval, and every ambiguity resolving to REJECT/NEUTRAL. The comment block at lines 77–87 explains why, correctly.
- **Prompt-injection defense.** `_sanitize_headline()` strips `VERDICT`/`DECISION` tokens out of third-party news text before it enters a prompt that gates trades. Very few portfolio projects think about this.

**The tests are real, and there are more than reported.** Actual count is **206**, not 149 (`pytest --collect-only`). I sampled `test_committee_parsing.py` and `test_outcome_tracker.py`. These are regression tests written from real bugs, not padding — e.g. `test_do_not_approve_is_reject` carries the comment *"The old parser matched APPROVE as a substring and approved this,"* and `test_instruction_echo_last_match_wins` encodes a genuine LLM failure mode. `test_paper_performance_vs_benchmark` asserts against a benchmark computation, not a getter. No trivial-assert padding in what I sampled.

**The safety posture is real, not aspirational.** `DRY_RUN = True` is a module constant in `engine.py:33`. The Robinhood path doesn't merely skip — it **raises `NotImplementedError`** (engine.py:292). The Schwab path threads `dry_run: bool = True` as a default-safe parameter. `tools/doctor.py` asserts the flag as a health check. Only `BUY` orders are ever constructed. `.gitignore` correctly excludes `.env`, `decisions.db`, `.schwab_tokens.json`, and no secrets are tracked. This would survive a security-minded reviewer.

**Does it have a RESULT? Partially — and this is the honest weak spot.**

Querying `decisions.db` directly:

| Period | Debates | Approvals | Rate |
|---|---|---|---|
| Pre-fix (through ~2026-07-26) | 195 | 2 | **1.0%** |
| Post-fix | 214 | 37 | **17.3%** |
| Total | 409 | 39 | 9.5% |

Also measured: **zero parse failures across all 409 debates** (no `confidence = 0` rows), i.e. 100% format compliance from the JSON-schema approach. And post-fix, rejections carry *higher* mean confidence (83.4) than approvals (78.4) — an interesting, defensible calibration signal.

But: `decision_outcomes` has 257 seeded rows and **0 matured return windows** in any of 30/90/365 days. `paper_orders` has **1 row**. Decisions only start 2026-06-29, so nothing has aged 30 days yet. The outcome loop and the `reflection.py` self-review module (timing / selection / confidence-calibration analysis — genuinely sophisticated) are **built and empty**.

So the honest statement is: **the LLM-system behaviour is measured; the investment performance is not.** Do not claim the latter. The former is enough.

## The actual asset

It is not the stock picking. Finance bots are the single most saturated portfolio genre, and a stock-picking bot applying for an ML role reads as a hobby.

The asset is commit `5b5a063`. Its message diagnoses a degenerate LLM committee — 286 debates → 2 approvals — and root-causes it to three distinct mechanisms: an asymmetric chair prompt stacking two thumbs on one side against a Bear that goes NEGATIVE 69% of the time by design; a self-reinforcing rejection ratchet where memory framed as "weigh consistency with past reasoning" made stocks cite their own rejection history as evidence; and sector-blind payout judgment rejecting REITs for being structurally REIT-shaped. Plus one line I would quote in a debrief: *"A committee that always says no is a constant function — informationally identical to having no committee."*

That is a measured LLM evaluation failure, root-caused, fixed, and verified with a before/after number (1.0% → 17.3%) and a discriminating control panel. **That is precisely the Airbnb job.** It is worth more than the entire dividend domain wrapped around it.

It also pairs with a memory the candidate already holds: a dormant feature has no failure signature, and green tests never catch it. Here, 140 tests passed while the system produced nothing. That's a mature, interview-ready story.

## Where it ranks

1. **EvalForge** — trained models, published HF artifacts, multi-provider platform. Uncontested #1.
2. **Dividend Desk (once public)** — real LLM system engineering, defensive parsing, measured failure analysis, 206 meaningful tests.
3. **MedInsight / DocuChat / AgentForge / DentaVision** — competent, but structurally "called an LLM API behind a REST endpoint." Extremely common.

A related portfolio finding, offered but **not acted on** per the constraints: `larmstrong1127.github.io` mentions Echoed Nights 5 times and DentaVision 4, but EvalForge only once. For an ML role that allocation is inverted. Separately, the public GitHub carries `Doubly-Linked-List`, `A-Star-Algorithm`, and `Cryptography-Implementations` — coursework-tier repos that dilute the signal of the good ones. Archiving is a 10-minute, non-trivial-impact action.

## Punch list (prioritized, with effort)

| # | Action | Effort | Why |
|---|---|---|---|
| **1** | **Publish the repo.** Scrub first: real holdings (SCHB, VGT, RKLB, PLTR), net-worth figures in `dashboard.html` / `portfolio.html` / `digests/`, and account identifiers. `.gitignore` already covers the DB and secrets; verify `git log -p` history is clean of anything that predates it. | 2–3 h | Everything else is worth zero until this is done. |
| **2** | **Rewrite the README to lead with the failure analysis.** Open with the 286→2 finding, the three root causes, the 1.0%→17.3% before/after, and the 100%-schema-compliance number. Demote the dividend pipeline to "the substrate." Title it as an LLM decision system, not an investing engine. | 2–3 h | This converts a saturated finance-bot into an LLM-evaluation case study. Highest-leverage single edit. |
| **3** | **Export a committed scorecard artifact.** `decisions.db` is gitignored (correctly). Add a script emitting an anonymized `docs/committee_scorecard.md` — approval rate by period, confidence distribution by verdict, parse-failure count. | 1–2 h | Right now the best evidence lives in a file no reviewer will ever see. |
| **4** | **State the unmeasured part explicitly.** One README section: "the outcome loop is built and instrumented; no window has matured; here is what will be reported and when." | 20 min | Pre-empts the obvious challenge and demonstrates the same honesty that makes the EvalForge card strong. |
| **5** | Fix the reported test count everywhere to 206. | 5 min | Undercounting your own work is a free correction. |

**Explicitly do NOT** flip `DRY_RUN`, chase the 149→206 test count upward, add broker adapters, or build a nicer dashboard. None move a hiring decision.

---

# QUESTION 2 — EvalForge on HuggingFace

Research findings below come from fetching actual comparable model cards and current HF documentation. Items already closed (A1–A3) and open items already enumerated in `docs/hf-model-audit-2026-07-26.md` (B1–B5, C1–C7) are **not** duplicated here; this covers the strategic layer above them.

## Two findings that overturn the premise

**Finding 1 — the free Gradio Space is largely gone.** Per [HF Spaces overview](https://huggingface.co/docs/hub/spaces-overview): static Spaces are free, but **Gradio and Docker Spaces now require PRO for personal accounts**, with the exception that free accounts in good standing may host **up to 2 Gradio Spaces on ZeroGPU**. So the assumed "free CPU tier layup" isn't one. Viable paths: ZeroGPU (free, 2 slots — overkill for a 184M model but it's the free lane) or PRO at ~$9/mo. CPU Basic itself is 2 vCPU / 16 GB, far more than the ~740 MB fp32 checkpoint needs. Free hardware sleeps on inactivity, so a recruiter clicking a cold link waits through a rebuild.

**Finding 2 — the bigger lever is RewardBench 2, and it is cheap.** RewardBench v2 ([arXiv 2506.01937](https://arxiv.org/pdf/2506.01937)) was **published at ICLR 2026** — live and canonical right now. Dataset [`allenai/reward-bench-2`](https://huggingface.co/datasets/allenai/reward-bench-2) is **1,865 rows / 6.98 MB**. The harness explicitly supports `AutoModelForSequenceClassification` — which is exactly what a DeBERTa Bradley-Terry model is. No custom code, no PR:

```
pip install rewardbench
python scripts/run_v2.py --model=DantheMan124/deberta-preference-reward
python scripts/run_v2.py --model=OpenAssistant/reward-model-deberta-v3-large-v2
```

Minutes on a 3090. That converts "0.7026 on my own split" — which a stranger cannot check, and which the model card itself concedes is in-distribution for him and OOD for the baseline — into a **third-party number produced by someone else's code**. It retires the single biggest remaining credibility gap (eval rigor graded D+).

**Be blunt about the expected outcome: the number will probably be bad.** RB2 is deliberately hard (best-of-4 scoring against three rejected responses, unseen human data) and a 184M encoder will look weak in absolute terms. That is fine — arguably better. The framing is the efficiency frontier: *"184M params scores X on RewardBench 2 vs. the 435M public baseline's Y, at Z× lower inference cost, both run by me on identical third-party code."* Publishing an unflattering-but-verifiable number, having sought it out voluntarily, is stronger evidence of eval judgment than any flattering self-reported one. That judgment *is* the job description.

## What credible reward-model cards do that his doesn't

From fetching [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) (his direct baseline — card is actually *thin*, no `model-index`, beatable on quality), [internlm2-1_8b-reward](https://huggingface.co/internlm/internlm2-1_8b-reward) (closest analogue: a small model done right), [RLHFlow/ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1), and [Skywork-Reward-V2](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B):

- **Third-party named benchmark**, not only a self-held-out split. Universal among the credible cards.
- **Per-category breakdown that shows the weak number.** ArmoRM publishes Chat 96.9 / Chat Hard 76.8 side by side. Showing your worst slice reads as confidence.
- **Helper APIs that do a job** — internlm ships `get_score()` / `compare()` / `rank()` plus a **Best-of-N sampling** section. A reward model that demonstrably *reranks* is more legible than one that emits a scalar.
- **Size/score tradeoff made legible** by putting siblings or baselines in the same table.
- **`library_name: transformers` must now be set explicitly** — auto-detection from `config.json` was removed for repos created after Aug 2024. Worth checking; silent omission.
- `base_model:` places the model in `microsoft/deberta-v3-base`'s fine-tune tree, a real inbound-traffic path.

Note on the new HF [eval-results system](https://huggingface.co/docs/hub/eval-results) (`.eval_results/` YAML keyed to registered Benchmark datasets, with `verified`/`community` badges): it's marked work-in-progress and benchmarks are allow-listed. **UltraFeedback is not registered**, so this route is not currently usable for his headline metric. The legacy `model-index:` block (already scoped as B2) remains the right call.

## Ranked levers — impact ÷ effort

| Rank | Lever | Effort | Impact | Verdict |
|---|---|---|---|---|
| **1** | **RewardBench 2 run of both his model and the OA baseline**, published on the card with the honest framing | 3–4 h | Very high | Retires the D+ eval-rigor grade. Verifiable by a stranger. Do this first. |
| **2** | **Benchmark 2–4 more public reward models on his own split** | **1–2 h** | High | `training/eval_reward_baseline.py` already takes `--model` and reuses the shared harness — this is nearly free. Turns a 2-row comparison into a real leaderboard and demonstrates harness reuse. |
| **3** | **Publish the held-out eval split as an HF dataset**, wired via `datasets:` | 2 h | High | Makes 0.7026-vs-0.6009 reproducible by anyone. Credibility, not vanity. |
| **4** | **Lead with cost/latency, not accuracy** | 1 h | High | `training/benchmark_results.json` already has it: local DeBERTa at **27 ms / $0.00** vs Claude Sonnet at 1,466 ms / $4.06 per 1k. Screening priority reportedly runs production metrics > model metrics. The "184M beats 435M" claim is *inherently* a cost story — tell it that way. |
| **5** | **Blog-style write-up of the Bradley-Terry work** | 4–6 h | Med-High | The hand-written dual-forward loop (no TRL) is the difference between "ran a script" and "implemented the objective." Also the natural home for the training collapse, the OOM, and the chance-level OOD probe. |
| **6** | **A Space — but as a batch eval harness, not a toy** | 4–6 h + PRO/ZeroGPU gate | Medium | See below. |
| **7** | **Hallucination judge's generalization story as a case study** | 2–3 h | Medium | The ID→OOD F1 drop plus ECE is a genuine finding; presently buried in a README section. |
| **8** | **Collection grouping the two models + Space** | 15 min | Low-but-cheap | Pure packaging; drives no downloads. Do it because it costs nothing. |

### On the Space specifically

The premise in the brief was that this is the biggest lever. **I disagree, and would demote it to ~6th.** Reasons: the free tier assumption doesn't hold (PRO or ZeroGPU required); free Spaces sleep, so a cold recruiter link is a slow or broken link; and unpinned `sdk_version` is the standard way a portfolio demo dies six months into a job hunt — pin it.

It is still worth building *after* items 1–4, and the brief's instinct on **pairwise** is exactly right: demo the comparison, never a single absolute score. That is the model's only validated capability, and the A1 audit finding was precisely that the card's original snippet demonstrated an uncalibrated absolute score. A demo that repeats that error on a live page would be worse than no demo.

But the version that actually lands for *this* role is not a two-textbox comparator. It's a **miniature eval harness**: paste or load N labeled pairs → get pairwise accuracy, per-slice breakdown by response-length delta (which doubles as the C2 length-bias probe), p50/p95 latency, cost-per-1k at $0.00, and a disagreement table against the OA baseline. That is "automate the LLM evaluation process" rendered as something a hiring manager can operate in fifteen seconds. Add `models:` to the Space README YAML so it back-links into the model page.

## What does NOT matter — skip these

- **Download counts, likes, leaderboard rank as such.** Lagging vanity metrics. A hiring manager reads the card; nobody is impressed by 40 downloads, and nobody is deterred by them either.
- **Getting listed on the RewardBench leaderboard.** The GitHub issue is a ~10-minute ask with uncertain acceptance. *The score is his to publish regardless.* Run it, publish it, don't wait on Ai2.
- **A long `tags:` list, emoji-heavy formatting, a Collection with nothing substantive in it.**
- **`bfloat16` weights to halve the download (C7)** and **`special_tokens_map.json` (C1)** — real but invisible to a hiring decision.
- **Growing the human OOD probe past N=15 (C4)** — genuinely interesting, but it is slow manual labeling for a caveat that already reads as honest. Low ROI against the same hours spent on RewardBench.
- **A prettier Gradio demo built *instead of* the RewardBench number.** This is the trap. The demo is garnish; the third-party reproducible comparison is the meal.
- **Chasing a better 0.7026.** More training runs won't help. The score is fine for base-size/1-epoch; what's missing is context around it, not magnitude.

---

# Single highest-ROI recommendation overall

**Run RewardBench 2 on both his model and the OpenAssistant baseline, publish both numbers honestly including an unflattering one, and lead the card with the cost/latency frontier rather than accuracy.** Three to four hours. It converts the one remaining D+ on the audit into the strongest evidence in the entire application, and it is a direct, demonstrable rehearsal of the Airbnb role's flagship project.

**Close second, and it should happen the same week:** publish Dividend Desk with a README that leads with the 286-debates-to-2-approvals failure analysis. He currently has a genuine LLM-evaluation war story with before/after numbers sitting on a hard drive where no one can read it. That is the cheapest large win available in the whole portfolio.

Sequenced: publish Dividend Desk (day 1) → RewardBench 2 + extra baselines + cost-first card rewrite (day 2) → eval-split dataset + Collection → Space last, if at all.
