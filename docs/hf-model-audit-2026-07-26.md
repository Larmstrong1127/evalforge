# HF Model Audit — `DantheMan124/deberta-preference-reward`

**Date:** 2026-07-26
**Auditor persona:** principal MLE / hiring manager, 10-minute skim
**Model:** https://huggingface.co/DantheMan124/deberta-preference-reward
**Status:** **PUBLIC and live.** Loads, renders, MIT-licensed, tagged. The blurb's link works — no automatic disqualifier.

## Repo contents (verified)

| File | Size | Verdict |
|---|---|---|
| `model.safetensors` | 738 MB | Good — safetensors, not `pytorch_model.bin` |
| `config.json` | 1.09 kB | `num_labels=1`, `max_position_embeddings=512`, custom `reward_temperature=1.1668` |
| `tokenizer.json` | 8.34 MB | Fast tokenizer present |
| `tokenizer_config.json` | 673 B | Present |
| `README.md` | 3.24 kB | Model card with YAML header |
| `.gitattributes` | 1.52 kB | — |

Missing: `special_tokens_map.json` (works without it, but non-standard for a DeBERTa-v3 upload), no `eval_results.json` / no metrics in card metadata, no `.py` example file.
Tags present: `reward-model`, `preference-learning`, `bradley-terry`, `deberta-v2`, `text-classification`, dataset + `base_model` declared. This is above the median for a junior upload.

## Grades

| Axis | Grade | One-line justification |
|---|---|---|
| First-impression clarity (30s test) | **B** | The first sentence correctly says what it is and what it outputs, but there is no worked example with real numbers and no baseline anywhere near the headline metric, so "0.7026" lands as an unanchored number. |
| Reproducibility | **C+** | Training recipe is described in accurate prose (lr, warmup, epochs, hardware, loss form) but there is no published seed, no link to the exact config file or commit, and no eval command a reader could run. |
| Evaluation rigor | **D+** | Zero external baseline (no random floor stated, no off-the-shelf reward model, no RewardBench slice), and the only OOD number is N=15. *(The original "measured at 1024" charge was re-measured on 2026-07-26 and does not hold — see the correction under finding 2. The remaining rigor gaps stand.)* |
| Usability | **C-** | The snippet loads and formats the text pair the same way training did, but it demonstrates an *absolute* score the model was never validated to produce, never shows the pairwise comparison that is the model's only validated use, and omits `model.eval()`. |
| Honesty & limitations | **A-** | Genuinely strong and rare: names the AI-feedback length/elaboration bias, calls its own ID accuracy "modest," reports the chance-level OOD probe with a CI, labels it a probe not a benchmark, and has a "What didn't work" section. Docks only for the undisclosed eval-length mismatch. |
| Professional polish | **B+** | License, tags, `base_model`, dataset, `pipeline_tag` all correct; GitHub link resolves to the right org (`Larmstrong1127/evalforge` — no wrong-org bug on the public card); loses points for no `metrics`/`model-index` block and an HF handle (`DantheMan124`) that matches nothing else on the application. |

## Overall grade: **B-**

**Verdict:** This artifact **helps** — but it is currently helping less than it easily could, and it contains one landmine. The honesty of the card is the strongest signal in the whole application; a hiring manager who reads "the OOD probe is at chance, this predicts UltraFeedback-style preferences, not any individual human's" concludes the candidate has ML judgment, which is exactly what "no professional ML experience" fails to establish otherwise. The landmine is that a reviewer who clicks through to `eval_reward.py` finds `default=1024` against a 512-trained, 512-configured model. Re-measurement (2026-07-26) shows the published numbers were *not* affected — they were produced at 512 — but the reviewer cannot know that from the repo, and the *judge* really was serving at 1024. Both are now fixed by deriving the budget from the checkpoint. Fix the usage snippet too and this is a solid A- artifact and a legitimate differentiator for the Airbnb "automate LLM evaluation" role.

---

## Top findings, ranked by how badly they land with a hiring manager

### 1. The usage snippet demonstrates the one thing the model isn't calibrated for (WORST)
The card's only example scores a *single* response and presents `sigmoid(raw / T)` as a 0–1 quality score. Two problems, both of which a principal MLE spots in under a minute:

- Bradley-Terry logits are identified only up to an **additive offset**. A single response's logit has no absolute meaning; `sigmoid(logit / T)` is not a probability of anything.
- `T` was fit by minimizing NLL of `sigmoid(margin / T)` — i.e. on **differences**. Applying it to a bare logit uses a calibration constant outside the quantity it was calibrated on.

The card even elevates this to a recommendation: "Recommended normalized score: `sigmoid(logit / T)`." And the model's *only* validated metric is pairwise accuracy — so the example does not exercise the validated capability at all. This same error is baked into production at `platform/api/evalforge/judges/reward_judge.py:59`.

### 2. Published metrics were measured at a max_length the model does not use — ❌ **NOT UPHELD, see correction**

> **Correction appended 2026-07-26, after re-measurement.** This finding was
> inferred from the code defaults, not measured, and it is **wrong**. Both
> headline numbers were re-run on the same held-out split: at 512 the model
> scores **0.7026** and fits **T = 1.1668**, reproducing the published values
> and the stored `reward_temperature` (`1.166796088218689`) bit-for-bit. At
> 1024 it scores 0.7046 and fits T = 1.1395 — neither of which was ever
> published. The operator had passed `--max-length 512` explicitly; only the
> script *defaults* were stale.
>
> The truncation-audit claim below is also wrong: the audit drops **1 pair in
> 62,688 at both 512 and 1024**, and the published figure was already the 512
> measurement (see `training/runs/reward-lr2e5-512-stdout.log`). Drops at 512
> are ≥ drops at 1024 in principle, but here both are 1.
>
> **What was genuinely broken** is narrower and worse-placed: `reward_judge.py`
> *served* live traffic at 1024 while applying a temperature fit at 512. 39%
> of `test_prefs` pairs (776 / 1,988) exceed 512 tokens on at least one side,
> so this was load-bearing in production, not theoretical. Fixed by deriving
> the budget from the checkpoint config rather than restating it in three
> places. Card numbers stand unchanged; no Hub metric edit is required.

The original finding, as written:

- `training/configs/reward-lr2e5.yaml:3` → `max_length: 512`
- `config.json` → `max_position_embeddings: 512`
- `training/eval_reward.py:41` → `--max-length` default **1024**
- `training/calibrate_reward.py:23` → `--max-length` default **1024**
- `platform/api/evalforge/judges/reward_judge.py:30` → `MAX_LENGTH = 1024`

So **0.7026, T=1.167, and the "1 of 62,688 pairs dropped" truncation-audit figure are all 1024-measurements published under a card that states a 512-token budget.** The data-loss claim is the most concretely wrong of the three: truncation drops at 512 must be ≥ drops at 1024, so "1 in 62,688" understates it. A fix + re-measurement is running concurrently; see the landing checklist below.

### 3. No baseline of any kind next to the headline number
`0.7026` appears with nothing to compare against. The card mentions the 5e-5 collapse at 0.5098, which implies chance, but never states the random floor explicitly and never compares to an off-the-shelf reward model. A reviewer cannot tell whether 70% is respectable for deberta-v3-base/1-epoch/UltraFeedback (it is roughly in range) or embarrassing. Adding one external row converts an unreadable number into evidence of benchmarking literacy — which is literally the Airbnb job.

### 4. Reproducibility stops at prose
Seed 42 exists in the config file but is never published. There is no link to `training/configs/reward-lr2e5.yaml`, no commit SHA, no runnable eval command. The training story is credible but not checkable, and "checkable" is the whole point of the artifact.

### 5. Identity fragmentation
HF handle `DantheMan124`, GitHub `Larmstrong1127`, resume presumably a legal name. A reviewer cross-referencing three identities gets a small friction hit and, worse, a moment of "is this actually his model?" Low effort to at least resolve on the card.

### 6. Minor polish
No `metrics:` / `model-index:` in the YAML header, so the model gets no metric badge and does not surface in HF's evaluation index. Missing `special_tokens_map.json`. No `library_name: transformers`. No `model.eval()` in the snippet (dropout active at inference — a real correctness bug in a copy-pasted example, not just style).

---

## Prioritized fix list

### (a) MUST-FIX before any referral goes out

**A1. Replace the usage snippet with a pairwise example.** This is the single highest-leverage edit in the audit. Exact replacement for the `## Usage` section:

````markdown
## Usage

This model is validated for **pairwise comparison**. Score two candidate
responses to the same prompt and compare; the calibrated temperature converts
the *margin* into a preference probability.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "DantheMan124/deberta-preference-reward"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
T = model.config.reward_temperature  # 1.167

def reward(prompt: str, response: str) -> float:
    """Raw Bradley-Terry score. Meaningful only relative to another response
    for the SAME prompt -- the scale has an arbitrary additive offset."""
    enc = tok(prompt, response, truncation=True, max_length=512,
              return_tensors="pt")
    with torch.no_grad():
        return model(**enc).logits.squeeze().item()

prompt = "What causes seasons?"
a = "The tilt of Earth's axis relative to its orbital plane."
b = "Because the Earth gets closer to the Sun in summer."

r_a, r_b = reward(prompt, a), reward(prompt, b)
p_a = torch.sigmoid(torch.tensor((r_a - r_b) / T)).item()
print(f"P(A preferred over B) = {p_a:.3f}")
```

**Do not** interpret a single `reward()` value as an absolute quality score:
Bradley-Terry training identifies rewards only up to an additive constant, and
`T` was fit on pairwise *margins*, not on individual logits.
````

**A2. Correct every 512-vs-1024 number on the public card.** When the concurrent re-measurement lands, the following must be updated **on the Hub README** (not only locally) — the Hub card is what the reviewer reads:
- Evaluation table: ID pairwise accuracy (currently `0.7026`)
- Calibration section: `T = 1.167` and the same value in `config.json` → requires a **new `config.json` commit**, since the stale T is baked into the weights repo and read by `reward_judge.py`
- Training section: the truncation-audit count (currently "1 of 62,688") re-run at 512
- `docs/`-side: `README.md:158` in the EvalForge repo carries the same `0.7026`, and `training/MODEL_CARD_preference_reward.md` is the card source — all three must move together.
Also fix the three code defaults so the mismatch cannot recur: `training/eval_reward.py:41`, `training/calibrate_reward.py:23`, `platform/api/evalforge/judges/reward_judge.py:30` → `512`.

**A3. Add a baseline row to the evaluation table.** Replace the table with:

```markdown
| Split | N | Pairwise accuracy |
|---|---|---|
| Random / chance floor | — | 0.500 |
| lr 5e-5 run (collapsed, discarded) | 1,987 | 0.510 |
| **This model** — UltraFeedback `test_prefs` (in-distribution) | 1,987 | **0.70xx** |
| Human OOD probe (EvalForge rating room) | 15 | 0.400 |
```

If time permits before the referral, add one row for an off-the-shelf reward model evaluated on the *same* `test_prefs` split with the *same* harness — `OpenAssistant/reward-model-deberta-v3-large-v2` is the natural comparator and the honest framing is "a 3x-larger public model scores X; this is a base-size single-epoch reproduction." Measuring yourself against a public baseline in your own harness is the exact skill the Airbnb project needs, and demonstrating it is worth more than the number.

### (b) SHOULD-FIX within the week

**B1. Publish reproducibility hooks.** Add to the Training section:

```markdown
- **Seed:** 42 (single run; no seed-variance study — treat ±1pt as noise).
- **Config:** [`training/configs/reward-lr2e5.yaml`](https://github.com/Larmstrong1127/evalforge/blob/main/training/configs/reward-lr2e5.yaml)
- **Training loop:** [`training/training/train_reward.py`](https://github.com/Larmstrong1127/evalforge/blob/main/training/training/train_reward.py)
- **Reproduce the eval:**
  ```bash
  python training/eval_reward.py \
    --checkpoint DantheMan124/deberta-preference-reward \
    --max-length 512
  ```
```

**B2. Ship the corrected YAML header** with metrics so the model gets a badge and shows in HF's index:

```yaml
---
license: mit
library_name: transformers
base_model: microsoft/deberta-v3-base
datasets:
  - HuggingFaceH4/ultrafeedback_binarized
language:
  - en
pipeline_tag: text-classification
tags:
  - reward-model
  - preference-learning
  - bradley-terry
  - rlhf
  - deberta-v2
  - llm-evaluation
model-index:
  - name: deberta-preference-reward
    results:
      - task:
          type: text-classification
          name: Pairwise Preference Prediction
        dataset:
          type: HuggingFaceH4/ultrafeedback_binarized
          name: UltraFeedback Binarized (test_prefs)
          split: test_prefs
        metrics:
          - type: accuracy
            name: Pairwise Accuracy
            value: 0.70XX   # <-- fill from the 512 re-measurement
---
```

**B3. Rewrite the opening paragraph** to pass the 30-second test with a number and a use case up front:

```markdown
# DeBERTa-v3 Preference Reward Model

A Bradley-Terry pairwise reward model (`microsoft/deberta-v3-base`, 184M params)
that scores which of two LLM responses a human-preference dataset would favor —
**no reference answer required**. 70% pairwise accuracy on UltraFeedback
`test_prefs` (chance = 50%), ~184M params, runs on CPU in ~40ms per response.

Built and trained from scratch (hand-written dual-forward PyTorch BT loop, no
TRL) as the reference-free `reward` judge inside
[EvalForge](https://github.com/Larmstrong1127/evalforge), an LLM evaluation
platform. Read the [Evaluation](#evaluation-honestly) section before using it —
it is deliberately blunt about where this model fails.
```

The "no TRL, hand-written loop" detail is worth foregrounding: for a junior candidate it is the difference between "ran a script" and "implemented the objective."

**B4. Resolve the identity gap.** Add a one-line footer to the card: `Author: <legal name> · [GitHub](https://github.com/Larmstrong1127) · [EvalForge](...)`, and set the HF profile display name to match the resume.

**B5. Add `model.eval()`** — done in A1, but audit `reward_judge.py` for the same omission.

### (c) NICE-TO-HAVE

- **C1.** Add `special_tokens_map.json` to the repo for tokenizer-loading robustness across `transformers` versions.
- **C2.** Add a length-bias probe to the card: bucket `test_prefs` by response-length delta and report accuracy per bucket. The card *asserts* length bias; measuring it turns an acknowledged limitation into a demonstrated analysis, and it is ~20 lines of code. This is the highest-value item in this tier for interview purposes.
- **C3.** Report ID accuracy with a 95% CI (at N=1,987, roughly ±2pt) so the headline number carries its own uncertainty.
- **C4.** Grow the human OOD probe past N=15. At N≈100 it becomes a finding instead of a caveat; the rating room already exists to collect it.
- **C5.** Enable the HF inference widget with 2–3 canned pairs so a reviewer can see it work without cloning anything.
- **C6.** Add a "Why base-size, one epoch?" line stating the compute budget (single 3090) explicitly as a constraint rather than leaving the modest number unexplained.
- **C7.** Consider `bfloat16` weights alongside fp32 to halve the 738 MB download.

---

## Landing checklist for the 512 re-measurement — **resolved 2026-07-26**

The re-measurement is done. Because the published numbers were already 512
measurements, **no metric anywhere needs to change.** Revised checklist:

1. ~~Hub `README.md` — evaluation table, calibration T, truncation-audit count~~ — **no change needed**; all three values verified correct.
2. Hub `config.json` — no metric change, but worth a commit to add `reward_train_max_length: 512` so the Hub copy self-describes its regime (the local checkpoint now carries it).
3. ~~Hub YAML `model-index` metric value~~ — value is `0.7026` as published, once B2 lands.
4. `training/MODEL_CARD_preference_reward.md` — **done**: correction section added, calibration section notes the 512 regime.
5. ~~`README.md:158`~~ — **no change needed**; `0.7026` / `T=1.167` stand.

Code defaults from A2 — **done**, and better than the literal `512` originally
prescribed: the budget is now *derived* from the checkpoint config
(`training/reward_metadata.py`, `_resolve_max_length` in `reward_judge.py`),
so it cannot drift again. What A2 got right was the diagnosis of *where*;
what it got wrong was assuming the reported numbers were the casualty.

The one genuinely user-visible change: the platform judge now truncates at
512 instead of 1024, so live `reward` scores on long inputs will shift
slightly. That is the intended correction — those scores were previously
off-regime.
