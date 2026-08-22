# hf-release — staged, not published

Everything under this directory is built, tested, and ready to upload. **None of
it has been uploaded.** No Hub repo was created, no file pushed, no Space
deployed, no collection made, and `hf auth` state was never touched.

The owner runs the commands. Start at [`PUBLISH.md`](PUBLISH.md).

## Layout

| Path | What it is | Ships as |
|---|---|---|
| [`PUBLISH.md`](PUBLISH.md) | The runbook: exact commands in order, leak-check gate, rollback limits. | — |
| [`collection.md`](collection.md) | Collection title, description, ordered items, `huggingface_hub` commands. | — |
| [`leak_scan.py`](leak_scan.py) | Pre-publish scanner. 21 rules, two tiers. Step 1 of the runbook. | — |
| `datasets/ultrafeedback-eval-split/` | The 1,987-pair eval split — **as a regeneration script + SHA-256 manifest, not rows.** | HF dataset repo |
| `datasets/ragtruth-diagnostic-200/` | The 200-example diagnostic sample with both encodings — **rows included.** | HF dataset repo |
| `space/reward-serve-demo/` | Gradio Space serving the reward model for pairwise scoring. | HF Space |
| `model-card-sync/` | Staged replacements for both live Hub model cards, plus the drift report. | Hub `README.md` ×2 |
| `tests/` | Determinism and filter-semantics tests for both regeneration scripts. | — |

## The two license verdicts, in one line each

- **UltraFeedback — script only.** Labelled MIT at the layer we would copy from,
  but ~34% of its prompts trace to ShareGPT (no upstream grant), FalseQA (no
  LICENSE file at all), and Evol-Instruct (non-commercial in places). We
  regenerate instead of redistribute.
- **RAGTruth — rows shipped.** A real `LICENSE` file, "MIT License",
  "Copyright (c) 2023 Particle Media", verified against the repository rather
  than assumed from a mirror's prose. The notice ships with the rows.

Full reasoning is in each dataset's `README.md` under "License".

## Verify everything locally

```bash
cd /path/to/evalforge

python hf-release/leak_scan.py                                  # expect CLEAN + 2 advisories
python -m pytest hf-release/tests/test_split_determinism.py -q  # expect 25 passed
cd hf-release/space/reward-serve-demo
python -m pytest test_scoring.py -q                             # expect 14 passed
CUDA_VISIBLE_DEVICES=-1 python smoke_test.py --port 7860        # boots + serves one real request
```

`smoke_proof.json` records an end-to-end run that reproduces the model card's
published worked example exactly (`margin = +1.2231`, `P(A) = 0.740`), which is
the check that the Space serves the model the card documents.

## One item to back-port into the repo

`training/MODEL_CARD_preference_reward.md` places the `reward_results.json` note
**inside** its evaluation table, so the blank line terminates the table and the
Human OOD probe row — the honest counterweight to the 0.7026 headline — renders
as loose text. The staged Hub card fixes it; the repo copy still has it. See
`model-card-sync/DRIFT.md` §2b.
