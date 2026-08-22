---
license: mit
language:
  - en
tags:
  - hallucination-detection
  - rag
  - evaluation
  - reproducibility
  - diagnostics
task_categories:
  - text-classification
pretty_name: RAGTruth Diagnostic Sample (200, both encodings)
size_categories:
  - n<1K
source_datasets:
  - wandb/RAGTruth-processed
---

# RAGTruth Diagnostic Sample — 200 examples, both encodings

The exact 200-example RAGTruth sample used to find, measure, and then re-measure
an **encoding defect** in [`DantheMan124/deberta-hallucination-judge`](https://huggingface.co/DantheMan124/deberta-hallucination-judge),
part of [EvalForge](https://github.com/Larmstrong1127/evalforge).

Unlike the [UltraFeedback split](https://huggingface.co/datasets/DantheMan124/ultrafeedback-eval-split)
in this collection, **the rows are shipped here** — RAGTruth carries an explicit
MIT LICENSE from Particle Media with a named copyright holder. See
[License](#license).

## The finding this sample is keyed to

The judge encoded its input as one flat string, truncated from the right:

```python
tokenizer(f"Q: {question} C: {context} A: {answer}", truncation=True, max_length=512)
```

The answer sits at the **tail** of that string. Right-truncation eats the tail.
So on any input longer than 512 tokens, the model was being asked to classify an
answer it had never been shown. Measured on these 200 examples:

| | legacy encoding | answer-preserving |
|---|---:|---:|
| Answer **fully** deleted before the model saw it | **101 / 200 (50.5%)** | **0 / 200** |
| Answer partially deleted | **31 / 200 (15.5%)** | **0 / 200** |
| Examples exceeding the 512-token budget | 132 / 200 (66%) | 132 / 200 (66%) |

The fix budgets the **context** — the one segment that is legitimately
compressible, since a retrieval context is a bag of evidence — and keeps the
question and the answer whole. When the whole triple already fits, the fix takes
the single-string path and produces a byte-identical encoding, so in-distribution
behaviour is unchanged. Model weights were never retrained; this is an
inference-time encoding change.

### What the fix bought, and what it did not

Re-measured on this identical sample (n=200, seed 42, CPU, same checkpoint):

| | legacy | answer-preserving |
|---|---:|---:|
| **ROC-AUC** | **0.444** | **0.603** |
| F1 at the best-accuracy operating point | 0.092 | 0.615 |
| Best F1 at any threshold | 0.566 | 0.627 |
| Best achievable accuracy | 0.605 | 0.605 |
| Accuracy at a naive 0.5 threshold | 0.475 | **0.385** |
| Majority-class (all-faithful) baseline | 0.610 | 0.610 |

**Both halves of this are real and they have to travel together.** ROC-AUC
moved from *below chance* (0.444) to meaningfully above it (0.603) — the model
can now rank examples by hallucination risk. Headline accuracy at the naive 0.5
cut got **worse** (0.475 → 0.385), because the corrected encoding compresses
nearly every score above 0.99, leaving 0.5 in the wrong place entirely.

And the conclusion the fix did **not** change: best achievable accuracy is
0.605, still *below* the 0.610 all-faithful baseline. A classifier that cannot
beat "predict faithful for everything" has no operating point worth deploying,
however well it ranks. **RAGTruth-like traffic remains out of scope for this
judge.** What the fix changed is that 0.603 now measures the model instead of
measuring a measurement bug.

## The sample, exactly

```python
random.Random(42).sample(examples, 200)
```

over [`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed)
split `test` in its native row order. That is the identical call made by
`training/run_benchmark.py` and `training/scripts/diagnose_ragtruth_agreement.py`
in the EvalForge repository, so these are the same 200 examples every published
diagnostic number was measured on.

| | |
|---|---|
| Source rows | 2,700 |
| Sampled | 200 |
| Labels | 78 hallucinated (1), 122 faithful (0) — positive rate 0.39 |
| Tokenizer for the encoding columns | `microsoft/deberta-v3-base` |
| Budget | 512 tokens |
| SHA-256 of the sample | `db98a0226b28783daa88f9dd3407749e4e684ed96310eb63cf1a7ae6b571e5a6` |

**A labelling trap worth repeating:** upstream `hallucination_labels` is a
JSON-encoded **string** (e.g. `"[]"`), not a list. `bool("[]")` is `True` in
Python, so testing the raw field for truthiness labels **every single row** as
hallucinated. It must be `json.loads()`'d first. `regenerate_sample.py` does;
the label balance above is the check that it did.

## Schema

`data/diagnostic-200.jsonl`, one JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `sample_index` | int | 0–199, position in the seeded sample. The sample's identity. |
| `question` | str | The user query. |
| `context` | str | The retrieval context the answer is judged against. |
| `answer` | str | The model output being judged. |
| `label` | int | 1 = hallucinated (non-empty span list), 0 = faithful. |
| `encoding.n_tokens_flat` | int | Tokens in the flat `Q:/C:/A:` string, with specials. |
| `encoding.n_tokens_answer` | int | Tokens in `" A: " + answer`. |
| `encoding.truncated` | bool | Whether the flat encoding exceeds the budget. |
| `encoding.legacy.*` | | `n_tokens_kept`, `answer_tokens_lost`, `answer_fully_dropped`, `answer_partially_dropped`. |
| `encoding.answer_preserving.*` | | The same, plus `context_tokens_lost`. |

The encoding columns are **derived**, and deliberately excluded from the
SHA-256: a tokenizer version that shifts one token count by one must not change
the sample's identity. The digest covers `question`, `context`, `answer`,
`label` and their order only.

## Reproduce

```bash
pip install -r requirements.txt

# Rebuild from source and check against the manifest
python regenerate_sample.py --verify
# VERIFY OK: 200 rows, sha256 db98a0226b28783d...

# Re-run the full diagnostic under both encodings (EvalForge repo, training/)
python scripts/diagnose_ragtruth_agreement.py \
    --checkpoint checkpoints/lr-2e5 --device cpu --out ragtruth_diagnostic.json
```

`--verify` also checks `n_source_examples` (2,700). The sample is a function of
upstream row order *and count*, so a reupload of `wandb/RAGTruth-processed`
would silently produce a different 200 under the same seed. The digest turns
that into a failed command instead of a quietly different benchmark.

## Files

| File | What it is |
|---|---|
| `data/diagnostic-200.jsonl` | The 200 rows with both encodings' truncation accounting. |
| `regenerate_sample.py` | Rebuilds from source. `--verify` checks the digest. |
| `manifest.json` | Counts, label balance, per-encoding totals, SHA-256. |
| `LICENSE` | The upstream MIT notice this redistribution rests on. |
| `requirements.txt` | `datasets` + `transformers` + `sentencepiece`. |

## License

**Verdict: redistributable. MIT, with a named copyright holder and a real
LICENSE file — this one we checked rather than assumed.**

[`ParticleMedia/RAGTruth`](https://github.com/ParticleMedia/RAGTruth) carries a
[LICENSE file](https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE)
headed "MIT License" and "Copyright (c) 2023 Particle Media", and GitHub's
license API reports `spdx_id: MIT` for the repository. The dataset files live in
that same repository and are covered by the repository-level grant. MIT permits
redistribution of derived subsets with the notice retained, which is why
`LICENSE` ships here.

The HuggingFace mirror this sample is actually loaded from,
[`wandb/RAGTruth-processed`](https://huggingface.co/datasets/wandb/RAGTruth-processed),
has **no `license:` field in its card metadata** — its card body states MIT in
prose. We treat the ParticleMedia LICENSE as the authority rather than the
mirror's prose, since that is where the grant actually originates.

**One caveat, stated rather than buried.** RAGTruth's responses were generated
by commercial models including the GPT family, and its source contexts draw on
external corpora (news articles for the summarization task, business-review data
for data-to-text). Particle Media's MIT grant covers their compilation and their
span annotations; it does not independently relicense third-party source text
that appears verbatim in the `context` field. This is a materially smaller
exposure than the UltraFeedback case — there is a real grant from a real
identified party here, which is the thing that was missing there — but it is not
zero, and if you redistribute this further, that field is the one to look at.

### Contact details present in the source text

We scanned the 200 rows rather than assuming. The `context` field contains, all
of it verbatim upstream text carried over from RAGTruth's source corpora:

| | Count | Where |
|---|---:|---|
| Email addresses | 1 | a business address in a news article (`sample_index` 184) |
| Phone numbers | 3 (2 distinct) | a toll-free business line and a business number (`sample_index` 54, 77) |

All are **business** contact details published in news and business-listing
copy, not personal data about private individuals, which is why they are left
in place — redacting them would silently desynchronise this sample from the
RAGTruth rows the published metrics were measured on, and the digest would stop
matching. They are documented here instead so the decision is visible rather
than implicit. If your use requires a PII-free corpus, filter those three
`sample_index` values and note that your split no longer reproduces the numbers
above.

`hf-release/leak_scan.py` in the EvalForge repository re-derives this list on
every publish and prints it as an advisory that has to be acknowledged.

The RAGTruth README requests citation. That is a norm rather than a license
condition; honor it anyway:

```bibtex
@inproceedings{niu2024ragtruth,
  title={RAGTruth: A Hallucination Corpus for Developing Trustworthy
         Retrieval-Augmented Language Models},
  author={Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and
          Shum, Kashun and Zhong, Randy and Song, Juntong and Zhang, Tong},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for
             Computational Linguistics},
  year={2024}
}
```
