"""Gradio Space: pairwise preference scoring with a 184M DeBERTa reward model.

Layout and copy only -- all model behaviour lives in `scoring.py`, which is
unit-tested against a stub in `test_scoring.py`.

The framing in the UI text is not decoration. This model is at the RANDOM FLOOR
out of distribution (RewardBench 2 average 25.3 vs a 25.0 random baseline), and
a demo that let a visitor type two arbitrary responses and read the number as a
quality verdict would be actively misleading. The honest numbers therefore
appear next to the input box, not buried in a collapsed section, and the four
facts that have to travel together -- ranking-only semantics, T = 1.1668, ID
accuracy 0.7026, RB2 25.3 -- are stated as one block.
"""
from __future__ import annotations

import os

import gradio as gr

from scoring import MODEL_REPO, RewardScorer, ScoringError, format_result

SCORER = RewardScorer()

# One worker on a CPU-tier Space: two ~40ms forward passes each, so the queue
# drains fast, but a burst should shed load rather than pile up behind a cold
# model load.
QUEUE_SIZE = 12
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "90"))

HEADER = """
# Pairwise Preference Scoring

Two responses to the **same prompt** go in. A calibrated preference probability
comes out. 184M parameters, runs on a free CPU Space, no API key.
"""

HONEST_FRAMING = """
### What this model is, stated with numbers

| | |
|---|---|
| **Ranking only** | It scores a *pair*. A single score is not a quality measure -- Bradley-Terry identifies rewards only up to an additive constant, so the zero point is arbitrary and the sign is meaningless. |
| **Calibration T = 1.1668** | Fit post-hoc on *margins*, at the same 512-token budget used for training. It converts `r_A - r_B` into a probability. It does not calibrate a lone score. |
| **In-distribution accuracy 0.7026** | UltraFeedback `test_prefs`, N = 1,987, chance floor 0.5000. A 435M public baseline transfers onto the same split at 0.6009. |
| **Out of distribution: 25.3 = the random floor** | RewardBench 2, unweighted six-domain average. Random is 25.0. **The model does not transfer.** The 435M OpenAssistant DeBERTa manages 32.0 on the same benchmark. |

**Those four facts travel together.** The 0.7026 is real, and it is real *only*
on UltraFeedback-style AI-feedback preferences -- the distribution it was fit
to. Off that distribution this demo is a random number generator with a
confident-looking table. Judge its answers accordingly, and use a
RewardBench 2 leaderboard model if you need a general-purpose reward signal.

It also inherits UltraFeedback's **length and elaboration bias**: longer, more
structured answers are systematically favoured regardless of whether they are
correct. The demo below will happily demonstrate this on you.
"""

FOOTER = f"""
---
Model: [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}) &middot;
Code and full evaluation: [EvalForge](https://github.com/Larmstrong1127/evalforge) &middot;
Eval split: [`ultrafeedback-eval-split`](https://huggingface.co/datasets/DantheMan124/ultrafeedback-eval-split)

Inputs are truncated to the model's 512-token training budget. Nothing typed here is logged or stored.
"""

EXAMPLES = [
    [
        "What causes seasons?",
        "Earth's axis is tilted about 23.5 degrees relative to its orbital plane, "
        "so each hemisphere receives sunlight at a steeper angle for part of the year.",
        "Because the Earth gets closer to the Sun in summer.",
    ],
    [
        "What is 17 * 24?",
        "408.",
        "Let me work through this carefully. We can decompose 17 * 24 as "
        "17 * (20 + 4) = 340 + 68 = 408. So the answer is 408. This method of "
        "breaking a multiplication into a sum of easier products is called the "
        "distributive property, and it is often faster than long multiplication.",
    ],
]


def score(prompt: str, response_a: str, response_b: str) -> str:
    try:
        return format_result(SCORER.score(prompt, response_a, response_b))
    except ScoringError as exc:
        return f"**Cannot score this.** {exc}"
    except Exception:  # noqa: BLE001 -- a Space must not hand a visitor a traceback
        return (
            "**Something went wrong scoring that pair.** The model may still be "
            "loading on this Space's first request; wait a few seconds and try again."
        )


def build() -> gr.Blocks:
    with gr.Blocks(title="Pairwise Preference Scoring", analytics_enabled=False) as demo:
        gr.Markdown(HEADER)
        gr.Markdown(HONEST_FRAMING)

        prompt = gr.Textbox(
            label="Prompt",
            placeholder="The question both responses are answering",
            lines=2,
        )
        with gr.Row():
            response_a = gr.Textbox(label="Response A", lines=8)
            response_b = gr.Textbox(label="Response B", lines=8)

        run = gr.Button("Score the pair", variant="primary")
        out = gr.Markdown(label="Result")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt, response_a, response_b],
            label="Try these (the second one demonstrates the length bias)",
        )
        gr.Markdown(FOOTER)

        run.click(
            score,
            inputs=[prompt, response_a, response_b],
            outputs=out,
            api_name="score",
        )
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(max_size=QUEUE_SIZE, default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        max_file_size="1mb",
    )
