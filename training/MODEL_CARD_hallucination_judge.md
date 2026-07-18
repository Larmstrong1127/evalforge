---
license: mit
base_model: microsoft/deberta-v3-base
tags:
  - text-classification
  - hallucination-detection
  - deberta-v2
datasets:
  - HaluEval
  - RAGTruth
metrics:
  - f1
  - precision
  - recall
pipeline_tag: text-classification
---

# DeBERTa Hallucination Judge

Fine-tuned `microsoft/deberta-v3-base` binary classifier: given a
`(question, context, answer)` triple, predicts whether the answer is
**faithful** (label 0) or **hallucinated** (label 1) with respect to the
context.

Trained as part of [EvalForge](https://github.com/Larmstrong1127/evalforge),
an open-source LLM evaluation platform, as a fast/free local alternative to
LLM-as-judge for hallucination detection.

## Intended use

- **Primary:** a cheap, fast first-pass hallucination filter inside an
  evaluation or RAG pipeline — score every `(question, context, answer)`
  triple for free, resolve the confident cases, and escalate only the
  uncertain middle to a stronger (paid) LLM judge.
- **Out of scope:** a standalone verdict on arbitrary real-world content. As
  the evaluation below shows, accuracy collapses on out-of-distribution
  hallucinations, so it should not be the only judge on data unlike HaluEval.

## Training data

Fine-tuned on [HaluEval](https://github.com/RUCAIBox/HaluEval) (QA subset,
~35K examples): synthetically generated faithful/hallucinated answer pairs
for the same question and context. Trained via a hand-written PyTorch loop
(AdamW, linear warmup+decay, mixed precision, gradient clipping, early
stopping on validation F1) — see the
[training code and full writeup](https://github.com/Larmstrong1127/evalforge/tree/master/training)
for the complete methodology, including a three-run learning-rate sweep and
a documented list of real bugs hit while training this model.

## Evaluation — please read before using this model

| Distribution | F1 | Precision | Recall | Expected Calibration Error |
|---|---|---|---|---|
| **In-distribution** (held-out HaluEval val) | 0.9937 | 0.999 | 0.989 | 0.0044 |
| **Out-of-distribution** ([RAGTruth](https://github.com/ParticleMedia/RAGTruth), real RAG hallucinations, never seen in training) | **0.5067** | — | — | 0.4010 |

**This is the headline finding, not a footnote.** In-distribution
performance is excellent, but the model was trained only on HaluEval's
*synthetically generated* hallucinations, which have a detectable stylistic
signature (an LLM deliberately prompted to produce a plausible-but-wrong
answer). That signature does not transfer to RAGTruth's real-world RAG
failures — F1 drops to ~0.51 and calibration collapses (ECE 0.40) on
genuinely out-of-distribution hallucinations.

**Recommended use:** a cheap, fast first-pass filter (route confidently-
faithful and confidently-hallucinated cases for free; send uncertain or
out-of-domain cases to a stronger judge), not a standalone replacement for
LLM-as-judge on arbitrary real-world content. A benchmark comparing this
model against Claude/GPT-4o/Gemini-as-judge on cost, latency, and accuracy
is in the [training README](https://github.com/Larmstrong1127/evalforge/tree/master/training#benchmark-local-judge-vs-llm-as-judge).

## Limitations

- **Does not generalize out-of-distribution.** F1 ≈ 0.51 and ECE ≈ 0.40 on
  RAGTruth; it over-predicts "hallucinated" on real RAG failures.
- **Learns a dataset artifact, not "hallucination" in general** — it keys on
  HaluEval's synthetic-hallucination style, which real systems do not share.
- **English QA only**, and capped at 512 tokens (longer contexts truncate).
- **Uncalibrated on real data** — do not treat its probabilities as reliable
  confidences outside HaluEval-like inputs.

## License

MIT — same as the base model (`microsoft/deberta-v3-base`) and the EvalForge
repository.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("DantheMan124/deberta-hallucination-judge")
model = AutoModelForSequenceClassification.from_pretrained("DantheMan124/deberta-hallucination-judge")

text = "Q: What is the capital of France? C: France is in Europe. A: Paris"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
logits = model(**inputs).logits
pred = logits.argmax(dim=-1).item()  # 0 = faithful, 1 = hallucinated
```

## Labels

- `0`: faithful
- `1`: hallucinated
