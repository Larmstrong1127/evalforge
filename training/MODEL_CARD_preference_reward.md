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
`(prompt, response)` text pair, outputs a scalar reward that is meaningful
**only relative to another response to the same prompt** (higher = more
preferred; the scale has an arbitrary additive offset — see
[Usage](#usage)). Built as part of
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

Because T was fit on **margins**, the quantity it calibrates is
`sigmoid((r_a - r_b) / T)` — the probability that A is preferred to B for the
same prompt. It does **not** calibrate `sigmoid(r / T)` for a single response;
see the warning under [Usage](#usage). Pairwise accuracy is invariant to T;
only the sharpness of the probability depends on it.

## Evaluation, honestly

All rows below are the **same split** (UltraFeedback `test_prefs`, N=1,987
after the truncation audit) run through the **same harness**
(`training/eval_reward.py` and `training/eval_reward_baseline.py`, which share
`evaluate_pairs`), at each model's own 512-token budget — except where noted.

| Model / split | Params | N | Pairwise accuracy |
|---|---|---|---|
| Chance floor (balanced binary choice) | — | — | 0.5000 |
| `OpenAssistant/reward-model-deberta-v3-large-v2` (public baseline) | 435M | 1,987 | 0.6009 |
| lr 5e-5 run (collapsed, discarded) | 184M | 1,987 | 0.5098 |
| **This model** — UltraFeedback `test_prefs` (in-distribution) | 184M | 1,987 | **0.7026** |

_All three rows are committed as [`reward_results.json`](reward_results.json) with the command that produced each. The baseline row was re-measured on CPU on 2026-08-17 — after an audit found it existed only in prose — and reproduced 0.6009 exactly._
| Human OOD probe (EvalForge rating room) | 184M | 15 | 0.4000 |

### Reading the baseline row honestly

This model beats a public reward model 2.4x its size by **+10.2 points**, and
that comparison is **not** a claim that it is the better reward model. It is
in-distribution and the baseline is out-of-distribution:

- This model was trained on UltraFeedback `train_prefs` and is being scored on
  UltraFeedback `test_prefs`. Same annotator (an LLM), same prompt mix, same
  elaboration conventions.
- `reward-model-deberta-v3-large-v2` was trained on a different preference
  mixture entirely (WebGPT, summarize-from-feedback, synthetic-instruct,
  Anthropic HH). UltraFeedback is a distribution shift for it.

So the correct reading is: **0.7026 is a real number, not a collapsed one**
(the floor is 0.5000 and a lr-sweep failure sat at 0.5098), and a strong public
model transferred onto this distribution lands at 0.6009. The honest inverse of
this result is already reported above — on the human OOD probe *this* model
drops to chance. Neither model generalizes for free; each is good on the
distribution it was fit to.

The tradeoff this project deliberately explored is a **small, local, free**
judge (184M, ~40ms/response on CPU, no API key, no per-call cost) against
larger models and hosted LLM judges. The baseline row exists so that tradeoff
is stated with a number instead of asserted.

Reproduce:

```bash
python training/eval_reward.py --checkpoint checkpoints/reward-lr2e5
python training/eval_reward_baseline.py \
    --model OpenAssistant/reward-model-deberta-v3-large-v2
```

The OOD probe is 15 genuine blind A/B votes by one human rater on real
llama3.2-vs-qwen2.5:14b outputs collected in EvalForge's rating room. At
N=15 the result is statistically indistinguishable from chance (95% CI
roughly 0.16–0.68), and it is reported as a probe, not a benchmark — but
the direction is consistent with the documented length/elaboration bias of
AI-feedback preference data: **this model predicts UltraFeedback-style
preferences, not any individual human's.**

## RewardBench 2 (2026-08-13): a third-party number, and it is the floor

I ran the official RewardBench 2 harness (`allenai/reward-bench` @ `05a9005`,
dataset @ `7ff0885`, 1,865 prompts, best-of-4, random baseline **25%** for the
five accuracy domains) on this model, unmodified except for a registered
dialogue template that reproduces the two-segment training encoding
token-for-token (the stock `raw` template drops the `[SEP]` boundary and
merges subwords across it). Full protocol and per-domain scores:
`training/rewardbench2_results.json`. Device: CPU, float32, 1h18m.

| Domain | This model (184M) | OA deberta-v3-large-v2 (435M, official leaderboard) |
|---|---:|---:|
| Factuality | 28.8 | 38.5 |
| Focus | 15.8 | 27.7 |
| Math | **47.1** | 50.3 |
| Precise IF | 23.1 | 26.9 |
| Safety | 35.8 | 36.7 |
| Ties* | 1.4 | 12.0 |
| **Average** | **25.3** | **32.0** |

\* Ties uses a margin-based metric with a chance level well below 25%; do not
read it against the 25% floor.

**Reading this honestly: out of distribution, this model is at the random
floor.** That is not a surprise — it is the strongest evidence yet for what
this card already says: the model predicts UltraFeedback-style preferences
and does not transfer. The official 435M OpenAssistant DeBERTa — the baseline
this model beats by 10 points in-distribution — manages 32.0 here, and
encoder-class reward models as a category sit near the floor on this
benchmark (the strong entries, 61–84, are all modern decoder-based
classifiers). The one domain where a 184M encoder holds up is Math: 47.1,
within three points of the 435M baseline at 40% of the size.

If you need a general-purpose reward model, use one from the RewardBench 2
leaderboard. If you need a small, free, CPU-viable judge for
UltraFeedback-distribution comparisons, that is the niche this model
occupies, and these numbers mark its boundary precisely.

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
- **No absolute scale.** Bradley-Terry identifies rewards only up to an
  additive constant, so a single score cannot be thresholded, averaged across
  a dataset, or compared across prompts. Only within-prompt comparisons are
  validated.

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

This model is validated for **pairwise comparison**. Score two candidate
responses to the same prompt and compare them; the calibrated temperature
converts the *margin* into a preference probability.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "DantheMan124/deberta-preference-reward"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
T = model.config.reward_temperature            # 1.1668
MAX_LEN = model.config.reward_train_max_length  # 512


def reward(prompt: str, response: str) -> float:
    """Raw Bradley-Terry score.

    Meaningful ONLY relative to another response to the SAME prompt -- the
    scale carries an arbitrary additive offset. Encoded exactly as training
    pairs were: the prompt and the response as the two segments of one
    sequence pair, right-truncated to the training budget.
    """
    enc = tok(prompt, response, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    with torch.no_grad():
        return model(**enc).logits.squeeze().item()


prompt = "What causes seasons?"
a = ("Earth's axis is tilted about 23.5 degrees relative to its orbital "
     "plane, so each hemisphere receives sunlight at a steeper angle for "
     "part of the year.")
b = "Because the Earth gets closer to the Sun in summer."

r_a, r_b = reward(prompt, a), reward(prompt, b)
p_a = torch.sigmoid(torch.tensor((r_a - r_b) / T)).item()
print(f"r_a={r_a:.4f}  r_b={r_b:.4f}  margin={r_a - r_b:.4f}")
print(f"P(A preferred over B) = {p_a:.3f}")
```

Verified output on this checkpoint:

```text
r_a=-1.0902  r_b=-2.3133  margin=1.2231
P(A preferred over B) = 0.740
```

> ### ⚠️ Do not use a single score as an absolute quality measure
>
> `reward(prompt, response)` on its own is **not** a calibrated 0-1 quality
> score, and `sigmoid(reward / T)` is not the probability of anything.
>
> - Bradley-Terry training only ever sees `r_chosen - r_rejected`, so the
>   objective is **invariant to adding a constant to every reward**. The zero
>   point is arbitrary. Note that both scores in the example above are
>   *negative* even though A is the good answer — the sign carries no meaning.
> - **T was fit on pairwise margins** (minimizing NLL of
>   `sigmoid(margin / T)`), so applying it to a bare logit uses a calibration
>   constant outside the quantity it was calibrated on.
> - The model's only validated metric is **pairwise accuracy**. Comparisons
>   between two responses to the same prompt are in-distribution for how it
>   was trained, evaluated, and calibrated; absolute scores are not.
>
> Ranking N candidates for one prompt is fine (the scores are a valid ordering
> within a prompt). Comparing scores *across different prompts*, thresholding
> them, or averaging them over a dataset is not.
