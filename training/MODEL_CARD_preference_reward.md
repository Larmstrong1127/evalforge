---
license: mit
base_model: microsoft/deberta-v3-base
tags:
  - reward-model
  - preference-learning
  - bradley-terry
  - deberta-v2
datasets:
  - HuggingFaceH4/ultrafeedback_binarized
pipeline_tag: text-classification
---

# DeBERTa-v3 Preference Reward Model

Bradley-Terry reward model for scoring LLM response quality: given a
`(prompt, response)` text pair, outputs a scalar reward (higher = more
preferred). Built as part of
[EvalForge](https://github.com/Larmstrong1127/evalforge), where it runs as
the `reward` judge — the platform's first judge that needs no golden answer.

## Intended use

- **Primary:** rank or score LLM responses when there is no reference answer
  (open-ended generation), inside EvalForge's `reward` judge or any pipeline
  that needs a cheap, local relative-quality signal.
- **Out of scope:** a stand-in for a specific human's preferences. It
  predicts *UltraFeedback-style* (AI-feedback) preferences and carries that
  data's known length/elaboration bias — see Evaluation.

## Training

- `microsoft/deberta-v3-base` with a 1-dim regression head, hand-written
  PyTorch Bradley-Terry loop: `L = -log sigmoid(r_chosen - r_rejected)`.
- Single epoch on `HuggingFaceH4/ultrafeedback_binarized` `train_prefs`
  (60,700 pairs), 512-token budget with an audited truncation safety net
  (any pair whose chosen/rejected encodings become identical after
  truncation is dropped and counted: 1 of 62,688 across train+eval).
- AMP + dynamic per-side padding + TF32 on a single RTX 3090; lr 2e-5,
  linear warmup 6%. An lr 5e-5 run collapsed to chance (0.51 pairwise
  accuracy) and was discarded.

## Calibration

Raw Bradley-Terry logits are arbitrarily scaled, so a scalar temperature
**T = 1.167** was fit post-hoc on the held-out split (NLL of
`sigmoid(margin / T)`) at the same 512-token budget used for training, and
stored in `config.json` as `reward_temperature` (with the budget it was fit
under alongside it as `reward_train_max_length`; see Correction below).
Recommended normalized score: `sigmoid(logit / T)`. Pairwise accuracy is
invariant to T; only score granularity depends on it.

## Evaluation, honestly

| Split | N | Pairwise accuracy |
|---|---|---|
| UltraFeedback `test_prefs` (in-distribution) | 1,987 | **0.7026** |
| Human OOD probe (EvalForge rating room) | 15 | 0.4000 |

The OOD probe is 15 genuine blind A/B votes by one human rater on real
llama3.2-vs-qwen2.5:14b outputs collected in EvalForge's rating room. At
N=15 the result is statistically indistinguishable from chance (95% CI
roughly 0.16–0.68), and it is reported as a probe, not a benchmark — but
the direction is consistent with the documented length/elaboration bias of
AI-feedback preference data: **this model predicts UltraFeedback-style
preferences, not any individual human's.**

## Correction (2026-07-26): train/serve sequence-length mismatch

An audit flagged that this model trains and is configured at **512** tokens
while three shipped code paths defaulted to **1024**:
`training/eval_reward.py`, `training/calibrate_reward.py`, and the platform's
`reward_judge.py`. The suspicion was that the headline metrics had been
measured off-regime.

**Both numbers were re-measured on the same held-out split, and they hold.**
The published figures were produced at 512 all along — the operator had
passed `--max-length 512` explicitly; only the *defaults* were stale. The
re-run reproduces the stored temperature bit-for-bit
(`1.166796088218689`), which is conclusive.

| Metric | Published | Re-measured @512 | Off-regime @1024 |
|---|---|---|---|
| ID pairwise accuracy (`test_prefs`, N=1,987) | 0.7026 | **0.7026** | 0.7046 |
| Calibration temperature T | 1.167 | **1.1668** | 1.1395 |
| Pairs dropped by truncation audit | 1 / 62,688 | **1 / 62,688** | 1 / 62,688 |
| OOD probe (N=15) | 0.400 | **0.400** | — |

So the *documentation* was correct and the *code* was wrong. The real defect
was in serving, not in reporting: the platform judge scored live traffic at
1024 while applying a temperature fit at 512. That is not hypothetical —
**39% of `test_prefs` pairs (776 / 1,988) have at least one side exceeding
512 tokens**, so the judge routinely fed the model context it never saw in
training. DeBERTa-v3 uses relative position embeddings, so it degrades
gracefully instead of erroring, which is exactly why the mismatch survived
review. The measured cost of the off-regime setting is small (+0.20pt
accuracy, T off by 0.027) but it was unmeasured, and an unmeasured
difference is not a small one.

**Fix:** the sequence budget is no longer restated anywhere. It is derived
from the checkpoint's own `config.json` — now carrying an explicit
`reward_train_max_length: 512` next to `reward_temperature`, so the constant
and the regime it was fit under travel together with the weights.

## Limitations

- **In-distribution accuracy is modest** (0.70 pairwise) even before any
  distribution shift — this is a base-size model trained for one epoch.
- **Inherits UltraFeedback's length/elaboration bias**; longer, more
  elaborate answers are systematically favored regardless of correctness.
- **Not personalized** — it does not model any individual rater; the OOD
  probe is at chance.
- **512-token cap**; longer `(prompt, response)` pairs truncate.

## What didn't work

- Training at 1024 tokens measured 9–12 h/epoch on the 3090 and died at
  hour 8.5 with `cudaErrorIllegalAddress` before its first checkpoint.
  Stepped down to 512 tokens (~1.7 h/epoch) with the truncation audit as
  the guardrail (data loss at 512: 1 pair in 62,688).
- The lr sweep's higher setting (5e-5) destabilized training outright —
  eval accuracy 0.5098, i.e. the model learned nothing.

## License

MIT — same as the base model (`microsoft/deberta-v3-base`) and the EvalForge
repository.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "DantheMan124/deberta-preference-reward"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)
T = model.config.reward_temperature

enc = tok("What causes seasons?", "The tilt of Earth's axis...",
          truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    raw = model(**enc).logits.squeeze().item()
score = torch.sigmoid(torch.tensor(raw / T)).item()  # 0..1, higher = better
```
