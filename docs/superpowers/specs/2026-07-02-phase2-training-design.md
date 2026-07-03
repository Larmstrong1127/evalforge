# EvalForge Phase 2 — PyTorch Training Package Design

Date: 2026-07-02
Status: Approved
Parent spec: `docs/superpowers/specs/2026-07-02-evalforge-design.md` (section 4)

## 1. Scope

The `training/` package only: fine-tune a hallucination-detection classifier,
evaluate it in- and out-of-distribution, and produce a benchmark comparing it
to LLM-as-judge on cost/latency/accuracy. Publishing to Hugging Face Hub and
wiring the model into `platform/` as a live judge are explicitly out of scope
— both become small follow-up tasks once the model is validated locally.

## 2. Structure

```
training/
  data/
    prepare.py        HaluEval download + preprocessing (train/val split)
    ragtruth.py        RAGTruth loading (held out, eval-only)
  models/
    classifier.py      DeBERTa + sequence-classification head wrapper
  train.py              hand-written training loop
  evaluate.py           metrics: precision/recall/F1, confusion matrix, ECE
  benchmark.py           fine-tuned judge vs Claude/GPT/Gemini judges
  configs/
    lr-2e5.yaml
    lr-5e5.yaml
    lr-1e4.yaml
  checkpoints/           gitignored, local only
  runs/                  gitignored, one TensorBoard subdir per experiment
  README.md              results table, loss curves, "what didn't work"
```

## 3. Data pipeline

`data/prepare.py` uses `datasets.load_dataset()` to fetch HaluEval
(QA/dialogue/summarization subsets combined, ~35K examples), builds
`(question, context, answer) -> {faithful, hallucinated}` pairs, and performs
a 90/10 train/val split. RAGTruth is loaded separately in `data/ragtruth.py`
and used only by `evaluate.py` — it is never touched during training or
hyperparameter selection, so the cross-dataset generalization number is honest.

Auto-download via the `datasets` library (not manual) so a fresh clone can run
`python train.py --config configs/lr-2e5.yaml` with no manual data setup.

## 4. Model and dependencies

`microsoft/deberta-v3-base` (~184M params) via
`AutoModelForSequenceClassification(num_labels=2)`. Tokenizer max_length=512.

DeBERTa-v3 uses a SentencePiece tokenizer, which requires `sentencepiece` and
`protobuf` as explicit pinned dependencies (not always pulled in automatically
by `transformers` extras) — both are added to `training/pyproject.toml` (or
the relevant requirements file) to avoid a "works on my machine" tokenizer
load failure on a fresh clone.

## 5. Training loop (`train.py`)

Hand-written PyTorch loop (not `Trainer`, not `accelerate`):
- `AdamW` optimizer, linear warmup (10% of steps) + linear decay schedule
- Mixed precision via `torch.amp.autocast("cuda")` / `torch.amp.GradScaler("cuda")`
  (the modern PyTorch 2.x APIs — `torch.cuda.amp.*` must not appear)
- Gradient clipping at `max_norm=1.0`
- Early stopping: patience 2 epochs on validation F1, max 6 epochs
- TensorBoard logging to `training/runs/<experiment-name>/`, named from the
  config file, so multiple runs overlay cleanly for comparison screenshots

**RTX-3090-tuned defaults:** batch size 64, no gradient accumulation needed
(24GB VRAM has substantial headroom at this model size and sequence length).

**Three experiment configs** (a learning-rate sweep: 2e-5, 5e-5, 1e-4) —
chosen to reliably surface a real "what worked / what didn't" comparison
(e.g. an LR that diverges or overfits fast) for the training README's honesty
section, which is part of the portfolio-quality bar from the parent spec.

## 6. Evaluation (`evaluate.py`)

Runs a checkpoint against:
- **Val split of HaluEval** (in-distribution)
- **RAGTruth** (out-of-distribution, held out entirely from training)

Reports precision, recall, F1, confusion matrix, and Expected Calibration
Error (ECE) for both, side by side. The in-vs-out-of-distribution gap is
itself the interesting result to report.

## 7. Benchmark (`benchmark.py`)

Takes the RAGTruth eval set and scores it two ways:
1. **Locally** with the best fine-tuned checkpoint
2. **Via LLM-as-judge** using Claude, GPT, and Gemini — reusing
   `platform/api/evalforge/providers/*` directly (the first real integration
   between `training/` and `platform/`)

Produces the flagship comparison table: agreement with ground truth, cost per
1K evaluations, p50/p95 latency. This is the ~$10-15 of cloud API spend
budgeted in the parent design.

## 8. Testing

- **Smoke test:** `train.py` runs 10 steps on CPU with a tiny synthetic
  sample; asserts loss decreases. No GPU required — matches the `train-check.yml`
  CI workflow planned for Phase 4.
- **`evaluate.py` metric math:** tested against hand-computed fixtures (known
  predictions/labels -> known precision/recall/F1/ECE), not against real
  model output, to keep tests fast and deterministic.

## 9. Explicitly deferred

- Publishing to Hugging Face Hub (follow-up task once the model is validated)
- Wiring the fine-tuned judge into `platform/api/evalforge/judges/` as a live
  `Judge` implementation (follow-up task, straightforward once the model is
  saved and loadable via `from_pretrained`)
- `deberta-v3-large` or other architecture variants (noted as a future idea,
  not part of the approved 2-3 experiment runs)
- DPO / reward-model training on collected preference pairs (Phase 2 in the
  parent spec's roadmap, post-ship)
