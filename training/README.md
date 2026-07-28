# Training a local hallucination-detection judge

Fine-tunes `microsoft/deberta-v3-base` on HaluEval to classify (question,
context, answer) triples as **faithful** or **hallucinated**, then measures
how well that classifier generalizes to real-world hallucinations (RAGTruth)
and how it stacks up against LLM-as-judge (Claude, GPT-4o, Gemini) on
accuracy, cost, and latency.

## TL;DR

- In-distribution (HaluEval val), the fine-tuned judge is excellent:
  **F1 ≈ 0.994**, extremely well-calibrated (ECE ≈ 0.005).
- Out-of-distribution (RAGTruth — real RAG system hallucinations, never seen
  during training), the same model collapses to **F1 ≈ 0.47–0.51** with poor
  calibration (ECE ≈ 0.40–0.62). This gap is the headline finding, not a
  footnote — see [Why the model doesn't generalize](#why-the-model-doesnt-generalize-and-why-that-mattered-for-model-selection).
- On a 200-example real-world benchmark, the local judge is **free and
  43x+ faster** than any cloud judge, but meaningfully less accurate
  (47.5% agreement vs. 78.5–85% for Claude/GPT-4o/Gemini). It is not a
  drop-in replacement for a paid judge on out-of-distribution data — see
  [Benchmark](#benchmark-local-judge-vs-llm-as-judge).

## Setup

```bash
cd training
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade
#      (plain `pip install torch` pulls the CPU-only wheel — see "What didn't work" below)
```

## Experiments

Three-run learning-rate sweep, all other hyperparameters held fixed
(batch_size=32, max_epochs=6, early_stopping_patience=2, AMP + TF32 enabled):

```bash
python -m training.train --config configs/lr-2e5.yaml
python -m training.train --config configs/lr-5e5.yaml
python -m training.train --config configs/lr-1e4.yaml
```

| Config | Best epoch | Loss | F1 | Precision | Recall | Wall time |
|---|---|---|---|---|---|---|
| lr-2e5 | 4/6 | 0.017 | 0.9937 | 0.999 | 0.989 | 13.6 min |
| lr-5e5 | 6/6 | 0.003 | **0.9942** | 0.999 | 0.990 | 13.5 min |
| lr-1e4 | 6/6 | 0.016 | 0.9937 | 0.999 | 0.989 | 14.3 min |

All three converge to essentially the same in-distribution performance.
`lr-1e4` (the highest learning rate) showed a mild, real dip in validation
F1 at epoch 4 (0.991 vs. epoch 3's 0.992) before recovering — the other two
configs improved monotonically every epoch. That's the kind of small
instability you'd expect from a higher learning rate, and it's the only
qualitative difference between the three runs on the training curve alone.

TensorBoard logs are in `runs/<experiment-name>/`; each config gets its own
subdirectory so curves overlay cleanly.

## Evaluation: in-distribution vs. out-of-distribution

```bash
python -m training.evaluate --checkpoint checkpoints/lr-2e5
```

Every checkpoint is evaluated two ways: against the held-out 10% HaluEval
validation split (in-distribution) and against the **entire RAGTruth test
split (2,700 examples)**, which was never touched during training,
validation, or hyperparameter selection.

| Config | ID F1 | ID ECE | **OOD F1** | **OOD ECE** |
|---|---|---|---|---|
| **lr-2e5** | 0.9937 | 0.0044 | **0.5067** | **0.4010** |
| lr-5e5 | 0.9942 | 0.0056 | 0.4735 | 0.6163 |
| lr-1e4 | 0.9937 | 0.0046 | 0.4814 | 0.5766 |

### Why the model doesn't generalize (and why that mattered for model selection)

`lr-5e5` has the best in-distribution F1 of the three — the number you'd see
first if you only checked validation metrics. It is also the **worst**
generalizer and the worst-calibrated model on real data (OOD ECE 0.62 vs.
`lr-2e5`'s 0.40). Picking the checkpoint with the highest validation score
would have silently picked the model least trustworthy on the data it will
actually see in production. `lr-2e5` — tied for the *lowest* in-distribution
F1 — is the one that should actually ship.

This is the entire reason the design held RAGTruth out as a hard,
untouched evaluation set instead of just reporting HaluEval numbers: HaluEval
hallucinations are synthetically generated (an LLM deliberately prompted to
produce a plausible-but-wrong answer), which gives them a detectable
stylistic signature. A classifier trained only on that signature learns to
detect *that specific kind of wrongness*, not hallucination in general —
and RAGTruth's real-world RAG failures don't share it. The confusion matrix
makes this concrete for `lr-5e5`: `[[221 TN, 1536 FP], [174 FN, 769 TP]]` —
the model massively over-predicts "hallucinated" on real data, guessing
based on surface patterns that don't transfer.

**`lr-2e5` is the checkpoint used for the benchmark below and would be the
one wired into the platform as a live judge.**

## Benchmark: local judge vs. LLM-as-judge

```bash
python run_benchmark.py       # local judge + Claude + GPT-4o + Gemini, 200 RAGTruth examples
```

200 randomly sampled RAGTruth examples (seed 42), scored by the `lr-2e5`
checkpoint locally and by three cloud LLM judges via `evalforge.providers`
(the same provider adapters from the platform's Phase 1 eval engine — this
is the first real integration point between `training/` and `platform/`).

| Judge | Agreement w/ ground truth | Cost / 1K evals | p50 latency | p95 latency |
|---|---|---|---|---|
| **local-deberta (lr-2e5)** | 47.5% | **$0.00** | **27 ms** | 33 ms |
| claude-sonnet-5 | **85.0%** | $4.06 | 1466 ms | 2921 ms |
| gpt-4o | 82.5% | $2.03 | 1311 ms | 1766 ms |
| gemini-2.5-flash-lite | 78.5% | $0.08 | 1182 ms | 2558 ms |

This table is generated from
[`benchmark_results.json`](benchmark_results.json) by
[`scripts/gen_results_table.py`](scripts/gen_results_table.py)
(`python scripts/gen_results_table.py`), so the numbers stay in sync with the
raw results rather than being hand-copied.

Real API spend for this benchmark: **~$1.23 total** across all three cloud
providers.

Takeaways:
- The local judge is essentially free and 43x+ faster than any cloud judge,
  but its accuracy gap on real-world data (this table's 47.5% roughly
  matches the fuller 2,700-example OOD evaluation's 50.7% — the ~3-point
  difference is sampling variance from the smaller 200-example draw, not a
  discrepancy) means it is **not currently a safe drop-in replacement** for
  a paid judge on out-of-distribution content.
- Gemini Flash-Lite gets within 4–6.5 points of GPT-4o's and Claude's
  agreement at roughly **1/25th and 1/49th their cost respectively** — the
  strongest accuracy-per-dollar point on this table.
- The realistic use case for the local judge today is a **cheap first-pass
  filter** (catch the confidently-faithful and confidently-hallucinated
  cases for free, route only the uncertain middle to a paid judge) rather
  than a full replacement — a natural v1.1 direction.

## What didn't work (and what it taught us)

Real bugs and dead ends hit while actually running this pipeline, in the
order they surfaced. Nothing here was hypothetical — every one of these cost
real GPU time, wall-clock time, or (in one case) real money before being
fixed.

**1. `pip install torch` silently installs the CPU-only build.** The first
training attempt ran on CPU without any error, then crashed with
`RuntimeError: mixed dtype (CPU): expect parameter to have scalar type of
Float` — a real dtype bug in this torch version's CPU autocast path under
DeBERTa's LayerNorm. The actual root cause was one level up:
`torch.cuda.is_available()` was `False` because the installed wheel had no
CUDA support at all. Fix: reinstall from PyTorch's CUDA index
(`pip install torch --index-url https://download.pytorch.org/whl/cu126
--upgrade`), now documented directly in `pyproject.toml`.

**2. `AutoModelForSequenceClassification.from_pretrained` silently loaded
fp16 weights.** With CUDA fixed, the next crash was `ValueError: Attempting
to unscale FP16 gradients.` — some recent `transformers` versions load
weights in the checkpoint's stored dtype by default rather than fp32, which
breaks `torch.amp.GradScaler`'s assumption that master weights are fp32.
Fixed by passing `torch_dtype=torch.float32` explicitly in
`build_model()`.

**3. Fixed `max_length=512` padding caused a real CUDA OOM.** `batch_size=64`
with every example padded to a flat 512 tokens measured at 23.9–24.0 GB on
a 24 GB card — essentially zero headroom. DeBERTa's disentangled attention
(content-to-position *and* position-to-content matrices, on top of standard
attention) is substantially more memory-hungry than plain BERT at a given
sequence length, and most HaluEval examples are far shorter than 512
tokens, so static padding wasted enormous activation memory on every batch.
Training ran two full epochs, then OOM'd on a batch that happened to draw
several longer-than-average examples. Fixed with dynamic per-batch padding
(`DataCollatorWithPadding`, pad to the batch's own longest example) plus
reducing `batch_size` to 32 for real margin against length variance.
Side effect: also roughly halved measured epoch time (55+ min/epoch under
memory pressure at batch=64 down to ~2.7 min/epoch at batch=32 + TF32 —
running right at the VRAM ceiling was silently costing far more than the
OOM risk alone suggested, likely allocator/fragmentation overhead).

**4. The RAGTruth loader's label logic had a real, silent bug before it ever
ran once.** The dataset's `hallucination_labels` field is a **JSON-encoded
string** (e.g. `"[]"`), not a real list. `bool("[]")` is `True` in Python —
so checking truthiness on the raw field would have labeled *every single
row* as hallucinated. Caught by loading real rows and inspecting the actual
schema before the first training run, not after. Fixed by `json.loads()`-ing
the field before checking it, with a regression test pinning the exact
failure mode.

**5. `gemini-2.0-flash` was deprecated between when the pricing table was
written and when the benchmark actually ran.** Every call failed with
`404: This model ... is no longer available`, which also explained an
earlier red herring — Google's rate-limit dashboard showed a hard `0/0/0`
quota for that exact model, which looked like a free-tier restriction but
was actually the model's retirement. Fixed by switching to
`gemini-2.5-flash-lite`, its direct successor at an identical price point.
**Lesson: pin external model/dataset identifiers, but don't assume they stay
correct forever — a quick liveness check before a real run is cheap
insurance.**

**6. The benchmark script only persisted results after *all* judges
finished — and that lost a real, already-paid-for result.** GPT-4o
completed a full, successful 200/200 scoring pass; while Gemini was stuck
retrying against an exhausted quota in the same run, the process was killed
(rightly — the retries were futile) and the unsaved GPT-4o result was lost
along with it, along with the real money already spent generating it.
Fixed by rewriting the rerun script to score and persist one judge at a
time, saving immediately on success. **Lesson: any script spending real
money or GPU time on a sequence of independent units of work should
checkpoint each unit's result the moment it lands, not batch-and-write at
the end.**

**7. `score_with_llm_judge` had no retry logic at all**, despite the
platform's own async runner (`platform/api/evalforge/runner.py`) already
having a well-tested exponential-backoff pattern for exactly this failure
class. Firing 200 sequential requests against OpenAI with zero pacing
triggered HTTP 429s on 61 of them; Gemini (before the quota/deprecation
issue was found) hit 429s on effectively all of them. Fixed by adding
retry-with-backoff plus a fixed inter-request delay, mirroring the runner's
already-proven approach rather than rediscovering the same problem in a
second codebase. **Lesson: when a new component makes the same kind of
external call as an already-hardened one elsewhere in the same repo,
default to reusing its resilience pattern.**

**8. `pydantic-settings`' `.env` file path is resolved relative to the
current working directory, not the module that defines the setting.**
Running the benchmark script from `training/` (rather than `platform/api/`,
where the `.env` file actually lives) silently loaded all three API keys as
empty strings — no error, just an immediate 401/invalid-header failure on
first use. Fixed by passing the `.env` path explicitly
(`Settings(_env_file="../platform/api/.env")`) rather than relying on the
default relative path.

None of these were caught by the test suite, and that's by design — every
one of them required the real model, the real GPU, the real dataset schema,
or a real network call to surface. The test suite's job (50 tests, all
passing throughout every one of the above) was to guarantee the *logic*
was correct once the real-world surprises were fixed; it was never going to
catch the surprises themselves. The fast, cheap fix in hindsight for most of
these: a single real training step (or a handful of real API calls) as a
dry run before committing to a full multi-epoch run or a full paid
benchmark — three of the eight issues above would likely have surfaced in
under a minute of real execution instead of across several full-scale
attempts.

## What's next

**Shipped since this study's first write-up:**

- The `lr-2e5` checkpoint is
  [published to Hugging Face Hub](https://huggingface.co/DantheMan124/deberta-hallucination-judge)
  (`publish_to_hub.py`; model card:
  [`MODEL_CARD_hallucination_judge.md`](MODEL_CARD_hallucination_judge.md)).
- It is wired into the platform as the `deberta-hallucination` judge
  (`platform/api/evalforge/judges/deberta_judge.py`) — a cheap first-pass
  filter ahead of a paid judge, per the benchmark's findings above.
- A preference **reward model** (DeBERTa-v3 Bradley-Terry, calibrated,
  T=1.167) was trained on UltraFeedback-binarized and wired in as the
  `reward` judge — see `train_reward.py`, `calibrate_reward.py`,
  `eval_reward.py`, and its model card
  [`MODEL_CARD_preference_reward.md`](MODEL_CARD_preference_reward.md).
  ID pairwise accuracy 0.7026 against a 0.5000 chance floor and 0.6009 for
  `OpenAssistant/reward-model-deberta-v3-large-v2` (435M) run on the same
  split through the same harness (`eval_reward_baseline.py`) — the public
  model is out-of-distribution on UltraFeedback, so that gap measures fit,
  not quality. The tiny human OOD probe sits at chance, so the rating room's
  real job is to accumulate the human votes that close that gap.

Still open:

- Re-train the reward model on accumulated rating-room votes once N is
  meaningful — the closed loop's whole point.
- Add resumable/checkpointed training (`train.py` currently has no
  optimizer-state resume — an interrupted run restarts from epoch 0; this
  bit twice during real runs on this project and is worth fixing before the
  next multi-hour training job).
