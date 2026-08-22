---
title: Pairwise Preference Scoring
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
short_description: Rank two LLM responses with a 184M reward model. Honest about its floor.
models:
  - DantheMan124/deberta-preference-reward
datasets:
  - DantheMan124/ultrafeedback-eval-split
---

# Pairwise Preference Scoring

Two responses to the same prompt in, a calibrated preference probability out.
184M parameters, free CPU tier, no API key. Serves
[`DantheMan124/deberta-preference-reward`](https://huggingface.co/DantheMan124/deberta-preference-reward)
from [EvalForge](https://github.com/Larmstrong1127/evalforge).

## What this Space is careful about

The four facts in the UI are there because any one of them alone is misleading:

- **Ranking only.** Bradley-Terry identifies rewards only up to an additive
  constant. A single score is not a quality measure, cannot be thresholded, and
  its sign means nothing — the demo's own example returns two *negative* scores
  for a pair where A is clearly the better answer. Only the margin is real.
- **T = 1.1668**, fit post-hoc on *margins* at the 512-token training budget.
  It converts `r_A − r_B` into a probability. Applied to a lone score it
  calibrates nothing, so this Space never exposes one.
- **In-distribution accuracy 0.7026** on UltraFeedback `test_prefs`
  (N = 1,987, chance 0.5000); a 435M public baseline transfers onto the same
  split at 0.6009.
- **RewardBench 2 average 25.3 against a 25.0 random floor.** Out of
  distribution the model does not transfer. The same 435M OpenAssistant DeBERTa
  manages 32.0 there.

A demo that let a visitor type two arbitrary responses and read the output as a
quality verdict would be actively misleading, so the honest numbers sit next to
the input box rather than in a collapsed section, and the caveats are attached
to the result table itself.

## Running it CPU-tier

- **Lazy load.** The model is not loaded at import. A CPU-tier Space that loads
  184M float32 parameters during module import blocks its own health check and
  gets restarted before it answers anything. The first caller pays ~9s; nobody
  else does.
- **Single thread, single concurrency.** `torch.set_num_threads(1)` and
  `default_concurrency_limit=1` — on a shared vCPU, thread thrash costs more
  than the parallelism buys. A warm request is ~0.5s end to end.
- **Bounded input.** 20,000 characters per field, rejected before tokenizing.
  The 512-token budget makes anything longer meaningless anyway, and one worker
  should not be held by a multi-megabyte paste.
- **Bounded queue.** `max_size=12`; a burst sheds load rather than piling up
  behind a cold model load.

## Local development

```bash
pip install -r requirements.txt

# Handler tests against a stub -- no weights, no Hub call, no Gradio process
python -m pytest test_scoring.py -q

# End-to-end: boot the real app, POST one real request, write smoke_proof.json
CUDA_VISIBLE_DEVICES=-1 python smoke_test.py --port 7860

# Point at a local checkpoint directory instead of the Hub
CUDA_VISIBLE_DEVICES=-1 REWARD_MODEL_REPO=/path/to/checkpoint python smoke_test.py
```

`smoke_proof.json` in this directory is the recorded result of that end-to-end
run. It reproduces the worked example published in the model card exactly —
`r_a = −1.0902`, `r_b = −2.3133`, margin `+1.2231`, `P(A) = 0.740` — which is
the check that the Space serves the same model the card documents.

## Files

| File | What it is |
|---|---|
| `app.py` | Gradio layout and copy. No model logic. |
| `scoring.py` | The entire model-facing surface: lazy loader, encoding, margin, formatting. |
| `test_scoring.py` | 14 stub-model tests, including Bradley-Terry shift invariance and swap symmetry. |
| `smoke_test.py` | Boots the app and POSTs a real HTTP request. |
| `smoke_proof.json` | Recorded proof of one served request. |

## API

The Space exposes `/gradio_api/call/score`:

```python
from gradio_client import Client

client = Client("DantheMan124/reward-serve-demo")
print(client.predict(
    "What causes seasons?",
    "Earth's axis is tilted about 23.5 degrees relative to its orbital plane.",
    "Because the Earth gets closer to the Sun in summer.",
    api_name="/score",
))
```

Nothing typed into this Space is logged or stored.
