"""Pairwise preference scoring -- the Space's entire model-facing surface.

Kept separate from `app.py` so the handler can be unit-tested against a stub
model with no torch download, no Hub call, and no Gradio process. `app.py` owns
only layout and copy.

Semantics this module is careful about (see MODEL_CARD_preference_reward.md):

  * The model is Bradley-Terry. Its objective only ever sees `r_chosen -
    r_rejected`, so it is invariant to adding a constant to every reward. A
    single score is NOT a quality measure; only the MARGIN between two
    responses to the SAME prompt is meaningful.
  * The calibration temperature T = 1.1668 was fit on MARGINS. It converts a
    margin into P(A preferred over B). Applying it to a bare logit calibrates
    nothing. This module therefore never exposes a single-response probability.
  * T and the 512-token budget are read from the checkpoint's own config
    (`reward_temperature`, `reward_train_max_length`), never restated here. A
    hardcoded copy is exactly how the train/serve mismatch this project already
    corrected got in.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

# Overridable so the Space can be smoke-tested against a local checkpoint
# directory without a Hub round-trip. Unset in the deployed Space.
MODEL_REPO = os.environ.get("REWARD_MODEL_REPO", "DantheMan124/deberta-preference-reward")

# Refuse absurd inputs before tokenizing them. A CPU-tier Space has one worker;
# a multi-megabyte paste would hold it for the whole request timeout.
MAX_CHARS = 20_000

# Documented fallbacks, used only if a checkpoint somehow lacks the keys. The
# published checkpoint carries both, so these should never fire.
FALLBACK_TEMPERATURE = 1.1668
FALLBACK_MAX_LENGTH = 512


class ScoringError(ValueError):
    """Bad user input. Surfaced to the UI as a message, not a stack trace."""


@dataclass
class PairScore:
    reward_a: float
    reward_b: float
    margin: float
    prob_a_preferred: float
    temperature: float
    max_length: int
    truncated: dict[str, bool] = field(default_factory=dict)

    @property
    def winner(self) -> str:
        if self.margin > 0:
            return "A"
        if self.margin < 0:
            return "B"
        return "tie"


class RewardScorer:
    """Lazy, thread-safe, single-instance model holder.

    The model is NOT loaded at import time. A CPU-tier Space that loads 184M
    float32 parameters during module import blocks its own health check and
    gets restarted before it ever answers a request; loading on the first real
    call costs that one caller a few seconds and nobody else.
    """

    def __init__(self, repo: str = MODEL_REPO) -> None:
        self.repo = repo
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._temperature: float = FALLBACK_TEMPERATURE
        self._max_length: int = FALLBACK_MAX_LENGTH

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # another thread won the race
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.repo)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.repo, torch_dtype=torch.float32
            )
            model.eval()
            # CPU-tier Space: one thread avoids oversubscribing a shared vCPU,
            # where thread thrash costs more than the parallelism buys.
            torch.set_num_threads(1)

            self._temperature = float(
                getattr(model.config, "reward_temperature", FALLBACK_TEMPERATURE)
            )
            self._max_length = int(
                getattr(model.config, "reward_train_max_length", FALLBACK_MAX_LENGTH)
            )
            self._tokenizer, self._model = tokenizer, model

    def _reward(self, prompt: str, response: str) -> tuple[float, bool]:
        """Raw Bradley-Terry score, plus whether the encoding hit the budget.

        Encoded exactly as training pairs were: prompt and response as the two
        segments of one sequence pair, right-truncated to the training budget.
        """
        import torch

        enc = self._tokenizer(
            prompt,
            response,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        truncated = int(enc["input_ids"].shape[-1]) >= self._max_length
        with torch.no_grad():
            logits = self._model(**enc).logits
        return float(logits.squeeze().item()), truncated

    def score(self, prompt: str, response_a: str, response_b: str) -> PairScore:
        validate(prompt, response_a, response_b)
        self.load()
        import torch

        r_a, trunc_a = self._reward(prompt, response_a)
        r_b, trunc_b = self._reward(prompt, response_b)
        margin = r_a - r_b
        p_a = float(torch.sigmoid(torch.tensor(margin / self._temperature)).item())
        return PairScore(
            reward_a=r_a,
            reward_b=r_b,
            margin=margin,
            prob_a_preferred=p_a,
            temperature=self._temperature,
            max_length=self._max_length,
            truncated={"a": trunc_a, "b": trunc_b},
        )


def validate(prompt: str, response_a: str, response_b: str) -> None:
    """Reject inputs the model cannot say anything meaningful about.

    An empty response is not a low-quality response -- it is a missing one, and
    the resulting margin would be a comparison against nothing.
    """
    fields = {"Prompt": prompt, "Response A": response_a, "Response B": response_b}
    for name, value in fields.items():
        if not value or not value.strip():
            raise ScoringError(f"{name} is empty. All three fields are required.")
        if len(value) > MAX_CHARS:
            raise ScoringError(
                f"{name} is {len(value):,} characters; the limit is {MAX_CHARS:,}. "
                "The model's 512-token budget makes longer inputs meaningless anyway."
            )


def format_result(score: PairScore) -> str:
    """Markdown verdict. Every framing caveat that could mislead a reader is
    attached to the number itself, not left to the page copy above it."""
    if score.winner == "tie":
        headline = "**Tie.** The two responses received identical scores."
    else:
        loser = "B" if score.winner == "A" else "A"
        headline = (
            f"**Response {score.winner} is preferred over Response {loser}** "
            f"with probability **{max(score.prob_a_preferred, 1 - score.prob_a_preferred):.3f}**."
        )

    lines = [
        headline,
        "",
        "| Quantity | Value | What it means |",
        "|---|---:|---|",
        f"| Raw score A | `{score.reward_a:+.4f}` | Meaningless alone -- arbitrary zero point |",
        f"| Raw score B | `{score.reward_b:+.4f}` | Meaningless alone -- arbitrary zero point |",
        f"| **Margin (A - B)** | **`{score.margin:+.4f}`** | The only quantity the model was trained on |",
        f"| P(A preferred) | `{score.prob_a_preferred:.4f}` | `sigmoid(margin / T)`, T = {score.temperature:.4f} |",
    ]

    if score.truncated.get("a") or score.truncated.get("b"):
        which = [k.upper() for k in ("a", "b") if score.truncated.get(k)]
        lines += [
            "",
            f"> **Truncated:** response {' and '.join(which)} hit the "
            f"{score.max_length}-token budget and was cut from the right. The score "
            "reflects only the part the model saw.",
        ]

    lines += [
        "",
        "> The two raw scores are both often negative even when one answer is clearly "
        "good. That is not a bug: Bradley-Terry identifies rewards only up to an additive "
        "constant, so the sign carries no meaning. Subtract them, never read them.",
    ]
    return "\n".join(lines)
