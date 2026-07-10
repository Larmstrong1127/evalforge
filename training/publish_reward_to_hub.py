"""One-off script: publishes the reward-lr2e5 checkpoint to Hugging Face Hub.

Manually-invoked operational script (same pattern as publish_to_hub.py).
Requires HF_TOKEN in training/.env.
"""
import io

from dotenv import dotenv_values
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "checkpoints/reward-lr2e5"
REPO_ID = "DantheMan124/deberta-preference-reward"

MODEL_CARD = """---
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
`sigmoid(margin / T)`) and stored in `config.json` as `reward_temperature`.
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

## What didn't work

- Training at 1024 tokens measured 9–12 h/epoch on the 3090 and died at
  hour 8.5 with `cudaErrorIllegalAddress` before its first checkpoint.
  Stepped down to 512 tokens (~1.7 h/epoch) with the truncation audit as
  the guardrail (data loss at 512: 1 pair in 62,688).
- The lr sweep's higher setting (5e-5) destabilized training outright —
  eval accuracy 0.5098, i.e. the model learned nothing.

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
"""


def main() -> None:
    env = dotenv_values(".env")
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("error: HF_TOKEN not found in training/.env")

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model.push_to_hub(REPO_ID, token=token, private=False)
    tokenizer.push_to_hub(REPO_ID, token=token, private=False)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(MODEL_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )
    print(f"published {CHECKPOINT} -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
