# Phase 5: Preference Reward Model — Design

**Date:** 2026-07-08
**Status:** Approved (pending user review)
**Depends on:** Phase 2 (training package), Phase 3 (ratings API + rating room), Phase 4 (published repo)

## Goal

Close the platform's preference loop: train a DeBERTa-v3-base reward model on
public human-preference data with a hand-written Bradley-Terry pairwise loop,
evaluate it honestly (held-out ID split, real rating-room votes as a tiny OOD
probe), publish the checkpoint to Hugging Face, and register it in the
platform as a `reward` judge behind an optional pip extra.

The story this completes: the platform collects blind A/B human preferences →
the training stack turns preferences into a reward model → the reward model
scores future runs inside the same platform.

## Decisions (from brainstorm, incl. external review feedback)

1. **Training objective: Bradley-Terry pairwise loss.**
   `L = -log σ(r(x, y_chosen) − r(x, y_rejected))`, two forward passes through
   one shared model per pair. Regression on absolute scores and independent
   binary classification were rejected (weaker formulations; don't match the
   pair-shaped data the platform collects).

2. **Base model: `microsoft/deberta-v3-base` with a 1-dim head**
   (`AutoModelForSequenceClassification`, `num_labels=1`). Reuses the Phase 2
   training stack (AMP, dynamic padding, TF32, cosine schedule) and makes the
   portfolio narrative "one training stack, two specialized judges."

3. **Training data: `HuggingFaceH4/ultrafeedback_binarized`** —
   `train_prefs` split for training, `test_prefs` held out for ID evaluation
   and temperature calibration. Rating-room votes are NEVER trained on; they
   are a deliberately tiny real-world OOD probe (report N honestly).

4. **Sequence budget: 1024 tokens** (not 512). Right-truncating completions
   under a small budget can leave chosen/rejected pairs textually identical —
   training on zero-variance pairs is pure noise. Mitigations:
   - prompt is preserved; only the completion tail is truncated;
   - the data loader AUDITS the processed set and reports the fraction of
     pairs whose (prompt, chosen) and (prompt, rejected) encodings are
     identical after truncation; identical pairs are dropped;
   - 1024 tokens × batch 8–16 with AMP fits the RTX 3090 at 184M params.
     (Note: DeBERTa has no FlashAttention path; AMP + dynamic padding +
     TF32 are the levers, same as Phase 2.)

5. **Input encoding: tokenizer-native text pairs.**
   `tokenizer(prompt, completion, ...)` — DeBERTa inserts its own special
   tokens. No hand-packed `"[CLS] User: ..."` template strings (fragile,
   duplicates tokenizer responsibility). Multi-turn UltraFeedback prompts are
   flattened to a single prompt string (the messages' text content joined
   with role prefixes) by the data loader, documented in its docstring.

6. **Score calibration: post-hoc temperature, fit on the ID validation
   split.** Raw Bradley-Terry logits are arbitrarily scaled; a bare sigmoid
   saturates (rewards of ±8 → 0.999/0.001) and destroys granularity. First
   training run lets logits float; afterwards a scalar temperature T is fit
   on `test_prefs` (minimizing NLL of σ((r_c−r_r)/T)), stored in the
   published checkpoint's config. NOT fit on the OOD probe (N too small —
   that would overfit the calibrator to noise). Fitting T on the same split
   used for ID evaluation is safe for the headline metric: pairwise accuracy
   is invariant under any positive temperature (monotone transform); T only
   affects the calibrated score's granularity.

7. **Judge protocol integration.** `reward` judge returns the standard
   `Judgment`: `score = σ(raw_reward / T)` in [0,1] (higher = better),
   `justification = f"raw_reward={r:.3f}, temperature={T:.2f}"` so the
   uncompressed value stays visible without changing the protocol. The judge
   needs no `expected` value — it scores any (prompt, output) pair, making it
   the platform's first judge usable on suites without golden answers.
   torch/transformers dependency ships behind an optional `reward` extra —
   in practice the same dependency set as the existing `deberta` extra
   (both extras list torch+transformers; pip dedupes).

## Components

All new training code follows the existing package layout:

| File | Responsibility |
|---|---|
| `training/training/data/preference.py` | Load + format UltraFeedback binarized; multi-turn flattening; truncation audit (drop identical pairs, report fraction); returns `PreferencePair(prompt, chosen, rejected)` records |
| `training/training/models/reward.py` | `build_reward_model()` / tokenizer builders (num_labels=1), mirrors `models/classifier.py` |
| `training/training/reward_loss.py` | Pure Bradley-Terry loss + pairwise-accuracy functions (unit-testable without a GPU) |
| `training/train_reward.py` | Training loop script: dual forward pass, AMP, dynamic padding, per-epoch eval on test_prefs, best-checkpoint save (same real-path discipline as Phase 2's train.py) |
| `training/calibrate_reward.py` | Fits temperature T on test_prefs from a trained checkpoint; writes T into the checkpoint config |
| `training/export_rating_pairs.py` | Extracts (prompt, chosen, rejected) from an EvalForge DB (`HumanRating` ⋈ `Result` ⋈ `PromptVersion`); JSONL out; skipped votes (`chosen_result_id IS NULL`) excluded |
| `training/eval_reward.py` | Reports pairwise accuracy on: ID test split, exported rating-room pairs (OOD probe, N reported) |
| `platform/api/evalforge/judges/reward_judge.py` | `reward` judge plugin; lazy weight load; module-eager imports for mockability (same pattern as deberta_judge per ADR-004) |
| configs | `training/configs/reward-lr{2e5,5e5}.yaml` — small lr sweep, same schema style as Phase 2 |

## Testing

TDD throughout, mirroring Phase 2:

- `reward_loss.py`: hand-computed loss values; **flip-symmetry test** — swapping
  chosen/rejected in a batch must produce mirrored loss (guards against
  ordering bias in the implementation);
- `preference.py`: tiny fixture dataset; multi-turn flattening; truncation
  audit drops a deliberately-identical pair;
- `export_rating_pairs.py`: in-memory SQLite seeded via ORM (reuses the
  platform's models); skipped votes excluded; pair orientation follows
  `chosen_result_id`;
- `reward_judge.py`: mocked tokenizer/model (`@patch`), returns-None-without-
  loading when output empty, score in [0,1], justification carries raw reward;
- training loop smoke test: 2 steps on fixture data, CPU, asserts loss is
  finite and checkpoint path exists (same shape as Phase 2's train smoke).

## Operational plan (after code ships)

1. Dry-run first per [[feedback_dry_run_before_real_run]]: 100-example
   training subset end-to-end on the real stack before the full run.
2. Full runs: lr sweep (2 configs), each ~2–4 h on the 3090 at 1024 tokens.
3. Calibrate temperature on test_prefs; evaluate on ID + OOD probe.
4. Publish best checkpoint (by ID pairwise accuracy, sanity-checked against
   the OOD probe — document if they disagree, Phase 2 style) to
   `DantheMan124/` on Hugging Face with an honest model card.
5. Wire checkpoint ID into `reward_judge.py`, live smoke test through the
   platform, update root README (roadmap item → shipped, honest numbers).

## Non-goals (YAGNI)

- No RLHF/PPO fine-tuning of a generator — the reward model itself is the product.
- No dashboard changes — the judge appears wherever judges already appear.
- No training resume-from-checkpoint (still the documented v1 limitation).
- No decoder-based reward model (rejected in brainstorm — risk without story gain).

## Risks

- **UltraFeedback is GPT-4-annotated**, not purely human preferences — the
  model card and README must say "trained on AI-feedback preference data,
  evaluated against my platform's human votes" honestly.
- **OOD probe is tiny** (likely N < 20 at first) — reported as a probe, never
  as a benchmark; more votes accumulate over time and the eval script can be
  re-run.
- Known Phase 2 environment traps (CPU-wheel torch, fp16 from_pretrained,
  CWD-relative .env) are documented in training/README and apply here.
