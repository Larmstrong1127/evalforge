---
license: mit
language:
  - en
tags:
  - preference-learning
  - reward-model
  - evaluation
  - reproducibility
task_categories:
  - text-classification
pretty_name: UltraFeedback Evaluation Split (regeneration script)
size_categories:
  - 1K<n<10K
source_datasets:
  - HuggingFaceH4/ultrafeedback_binarized
configs: []
---

# UltraFeedback Evaluation Split — 1,987 pairs

> ## This repository ships a SCRIPT, not the data.
>
> There are no parquet or jsonl rows here. `regenerate_split.py` rebuilds the
> exact split from the source dataset on your machine, and `manifest.json`
> carries a SHA-256 so you can prove your copy is byte-identical to the one the
> published metrics were measured on.
>
> **Why:** see [License and why the rows are not here](#license-and-why-the-rows-are-not-here).
> The short version is that `HuggingFaceH4/ultrafeedback_binarized` is labelled
> MIT, but roughly a third of UltraFeedback's prompts trace to sources that
> either carry no license at all or have no valid upstream grant to pass along.
> An MIT label at the top of that stack does not cure what is underneath it, and
> re-uploading the rows would be relying on someone else's optimism. Rebuilding
> them from the source you are already entitled to download costs one command
> and skips the question entirely.

This is the held-out evaluation split behind every in-distribution number
reported for [`DantheMan124/deberta-preference-reward`](https://huggingface.co/DantheMan124/deberta-preference-reward),
built as part of [EvalForge](https://github.com/Larmstrong1127/evalforge).

## Source

| | |
|---|---|
| Dataset | [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) |
| Split | `test_prefs` |
| Upstream | [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback) (Cui et al., 2023) |
| Tokenizer for the audit filter | `microsoft/deberta-v3-base` |
| Sequence budget | 512 tokens |

## The filter, exactly

Two filters, applied in order, no sampling and no shuffling — the split is a
deterministic function of the source rows in their native order.

1. **Empty-preference filter.** Drop any row whose last `chosen` turn equals its
   last `rejected` turn: there is no preference signal to learn from. **Removes 0
   rows on `test_prefs`.**

2. **Truncation audit.** Encode `(prompt, chosen)` and `(prompt, rejected)` as
   the two segments of one DeBERTa sequence pair at `max_length=512`. Drop the
   pair if the cap was **actually hit** *and* the two encodings came out
   **identical**.

   This is the filter that matters. A long shared prompt with two differing
   tails can have both tails cut off, leaving two byte-identical encodings.
   Training or evaluating on such a pair asks the model to separate two
   identical inputs — pure gradient noise in training, and a coin flip in
   evaluation. The "cap was actually hit" guard is load-bearing: if both
   encodings come in *under* the budget then nothing was cut, so equal encodings
   are impossible (identical completions were already removed by filter 1) and
   the check would be measuring nothing.

   **Removes exactly 1 row.**

```
1,988 loaded  ->  1 dropped  ->  1,987 evaluated
```

Prompts are flattened before encoding. Multi-turn prompts become a single
`"role: content"` transcript joined by newlines — DeBERTa has no chat template,
and the reward head only needs the textual context, not the turn structure.
Single-turn rows use the dataset's own `prompt` column verbatim.

## The numbers this split reproduces

All three rows are the **same 1,987 pairs** through the **same harness**
(`training/eval_reward.py` and `training/eval_reward_baseline.py`, which share
`evaluate_pairs`), each at its own 512-token budget. Chance floor is 0.5000.

| Model | Params | Pairwise accuracy | Distribution |
|---|---:|---:|---|
| Chance floor (balanced binary choice) | — | 0.5000 | — |
| `OpenAssistant/reward-model-deberta-v3-large-v2` | 435M | **0.6009** | out-of-distribution for that model |
| lr 5e-5 run (collapsed, discarded) | 184M | **0.5098** | in-distribution |
| `DantheMan124/deberta-preference-reward` | 184M | **0.7026** | in-distribution |

### Read the 0.7026 with its counterweight attached

The 10.2-point gap over a public model 2.4x the size is **not** a claim that the
smaller model is better. It is in-distribution for our model and
out-of-distribution for the baseline: ours was fit on UltraFeedback
`train_prefs` and is scored on `test_prefs` — same annotator (an LLM), same
prompt mix, same elaboration conventions — while the OpenAssistant model was fit
on a different preference mixture entirely (WebGPT, summarize-from-feedback,
synthetic-instruct, Anthropic HH).

What the number actually says is narrower and still worth something: **0.7026 is
a real result and not a collapsed one** (the floor is 0.5000 and a
learning-rate-sweep failure sat at 0.5098), and a strong public model
transferred onto this distribution lands at 0.6009.

**The out-of-distribution counterweight:** on RewardBench 2 the same model
averages **25.3** against a **25.0** random baseline. It is at the floor. This
split measures how well the model fits one preference distribution; it does not
measure whether the model is good, and the two numbers have to travel together.

## Harness commands

```bash
# 1. Rebuild the split and prove it matches
pip install -r requirements.txt
python regenerate_split.py --verify
# VERIFY OK: 1987 pairs, sha256 9c392052da72518e...

# 2. Reproduce 0.7026 and 0.5098 (EvalForge repo, training/)
python eval_reward.py --checkpoint checkpoints/reward-lr2e5 --max-length 512

# 3. Reproduce the 0.6009 baseline row
python eval_reward_baseline.py \
    --model OpenAssistant/reward-model-deberta-v3-large-v2 --batch-size 8
```

Both eval commands rebuild this split internally through the same
`load_ultrafeedback_pairs` / `audit_and_filter_pairs` pair that
`regenerate_split.py` reimplements, so the script is a mirror of the harness
rather than a second opinion about it.

`--verify` is a real gate, not a formality: the split is a function of upstream
row order and content, so a reupload or reorder of
`HuggingFaceH4/ultrafeedback_binarized` would silently change what "1,987 pairs"
means. The digest turns that into a failed command instead of a quietly
different number.

## Files

| File | What it is |
|---|---|
| `regenerate_split.py` | Rebuilds the split. `--verify` checks against the manifest. |
| `manifest.json` | Counts, dropped index, and the SHA-256 of the regenerated split. |
| `requirements.txt` | `datasets` + `transformers` + `sentencepiece`. |

## License and why the rows are not here

**Verdict: the split is regenerated locally, not redistributed.**

The layer we would be copying from is permissive. Both
[`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
and upstream [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback)
declare `license: mit` in their card metadata, and MIT permits redistribution of
derived subsets with the notice retained. Taken at face value, we could ship the
rows.

We are not taking it at face value, for three reasons:

1. **~34% of the prompts have no clean upstream grant.** UltraFeedback pools
   63,967 prompts from six sources. **ShareGPT (19,949, ~31%)** is scraped
   ChatGPT conversation data with no upstream license to pass along.
   **FalseQA (2,339)** comes from [`thunlp/FalseQA`](https://github.com/thunlp/FalseQA),
   which has **no LICENSE file at all** — and absence of a license is absence of
   permission, not permission by default. **Evol-Instruct (10,000)** descends
   from the WizardLM line, distributed under non-commercial terms in some
   releases. The clean remainder is FLAN (Apache-2.0), TruthfulQA (Apache-2.0),
   and UltraChat (MIT). OpenBMB's MIT label sits on top of that stack; it does
   not launder it.

2. **The completions were generated by commercial models.** The UltraFeedback
   card names GPT-4, GPT-3.5 Turbo and Bard among the generators, and GPT-4 as
   the scorer. Neither card carries an OpenAI or Google terms notice, and
   OpenBMB cannot license away output terms that were never theirs. That is a
   contractual question, separate from copyright, and it is not one this
   repository should quietly answer on a reader's behalf.

3. **We do not need to.** Anyone who can run the evaluation can already
   download the source dataset. Shipping a script and a digest gives them the
   same split with the same verifiability and asks nothing of the license.

The `license: mit` in this repository's own metadata covers **this repository's
own contents** — the script, the manifest, and this card — not the UltraFeedback
data, which is never present here.

If you want the strictly-clean subset instead of the full split, filter the
source rows by provenance to FLAN + TruthfulQA + UltraChat before running the
audit. That is a different split and reproduces none of the numbers above; it is
noted here because it is the honest answer to "can I redistribute *any* of it".

### Citation

```bibtex
@article{cui2023ultrafeedback,
  title={UltraFeedback: Boosting Language Models with High-quality Feedback},
  author={Cui, Ganqu and Yuan, Lifan and Ding, Ning and Yao, Guanming and
          Zhu, Wei and Ni, Yuan and Xie, Guotong and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2310.01377},
  year={2023}
}
```
