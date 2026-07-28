"""Preference reward judge: a fine-tuned DeBERTa reward model (Bradley-Terry
trained on UltraFeedback-binarized) scoring how preferred a response is,
published to Hugging Face Hub and run locally.

Requires the optional `reward` extra (same torch/transformers set as the
`deberta` extra). Module-level imports + lazy weight loading: identical
pattern and rationale as deberta_judge.py (see ADR-004).

Unlike every other judge, this one needs NO expected output — it scores any
(prompt, output) pair, so it works on suites without golden answers.
`expected` is accepted and ignored (protocol uniformity).

SCORE SEMANTICS — read before comparing these numbers to any other judge's.

`Judgment.score` here is the **temperature-scaled Bradley-Terry reward**,
`raw_reward / T`. It is a *relative* preference score, not a 0-1 quality
probability:

- It is unbounded (both signs occur) and has an **arbitrary additive
  offset**. Bradley-Terry training only ever sees `r_chosen - r_rejected`, so
  the objective is invariant to adding a constant to every reward. Nothing
  pins the zero point.
- It is therefore valid **only for ranking candidate outputs against each
  other on the same prompt**. A single value in isolation says nothing about
  absolute quality, and averaging it across a suite is meaningless.
- Differences *are* meaningful and *are* calibrated: for two outputs A and B
  on one prompt, `sigmoid(score_B - score_A)` is the model's calibrated
  P(B preferred over A). That is exactly the quantity T was fit for — T
  minimizes NLL of `sigmoid(margin / T)` on the held-out split, i.e. it was
  calibrated on *margins*. Dividing by T here means `/compare`'s
  `score_delta` is already the calibrated margin, so a consumer can pass it
  straight to a sigmoid.

This judge previously returned `sigmoid(raw_reward / T)` and presented it as
a calibrated 0-1 quality score. That was a statistical error on two counts:
it applied a margin-fit temperature to a bare logit, and it dressed an
offset-free relative score up as an absolute probability. The sigmoid was
also pure loss — being monotone it changed no ranking, while squashing away
the margin information that is the model's only validated output.

T comes from the checkpoint config key `reward_temperature` (fallback 1.0 for
uncalibrated checkpoints); the raw pre-scaling reward is preserved in the
justification string.

The tokenizer's sequence budget is likewise derived from the checkpoint
config rather than hardcoded (see `_resolve_max_length`): train and serve
must agree on it, and a literal here is a silent drift waiting to happen.
"""
import asyncio

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evalforge.config import Settings
from evalforge.judges import Judgment

MODEL_ID = "DantheMan124/deberta-preference-reward"

# Keys the sequence budget is derived from, in priority order. Mirrors
# training/reward_metadata.py — this package cannot import the training
# package, but both read the same checkpoint config, which is the single
# source of truth that travels with the weights.
TRAIN_MAX_LENGTH_KEY = "reward_train_max_length"
POSITION_LIMIT_KEY = "max_position_embeddings"


def _resolve_max_length(config: object) -> int:
    """Derive the training sequence budget from the loaded checkpoint config.

    Serving a reward model above the length it was trained at feeds it an
    off-regime input (DeBERTa-v3 uses relative positions so it degrades
    gracefully rather than crashing, which is exactly what let a 1024
    constant sit here unnoticed against a 512-token checkpoint) and applies
    a temperature fit under a different regime. Derived, never restated.
    """
    for key in (TRAIN_MAX_LENGTH_KEY, POSITION_LIMIT_KEY):
        value = getattr(config, key, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError(
        f"reward checkpoint config carries neither {TRAIN_MAX_LENGTH_KEY!r} nor "
        f"{POSITION_LIMIT_KEY!r}; cannot determine its training sequence budget"
    )


class RewardJudge:
    name = "reward"

    def __init__(self, settings: Settings) -> None:
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None
        self._temperature: float = 1.0
        self._max_length: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self._model = model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()
        self._temperature = float(getattr(self._model.config, "reward_temperature", 1.0))
        self._max_length = _resolve_max_length(self._model.config)

    def _score_sync(self, prompt: str, output: str) -> tuple[float, float]:
        """Return (temperature-scaled reward, raw reward).

        The scaled value is the score; see the module docstring for why it is
        deliberately NOT squashed through a sigmoid.
        """
        assert self._tokenizer is not None and self._model is not None
        assert self._max_length is not None
        encoding = self._tokenizer(
            prompt, output, truncation=True, max_length=self._max_length, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            raw = self._model(**encoding).logits.squeeze().item()
        return raw / self._temperature, raw

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        if not output.strip():
            return None  # nothing to score; don't load the model for this
        # Load on the event loop BEFORE handing off to a worker thread: the
        # runner judges items concurrently, and lazy-loading inside to_thread
        # would let several threads race the first load (redundant multi-GB
        # downloads + duplicate models on the GPU). Same pattern as
        # deberta_judge.py.
        self._ensure_loaded()
        scaled, raw = await asyncio.to_thread(self._score_sync, prompt, output)
        return Judgment(
            score=scaled,
            justification=(
                f"relative preference score (Bradley-Terry, ranking within the same "
                f"prompt only — not an absolute quality probability); "
                f"raw_reward={raw:.3f}, temperature={self._temperature:.2f}"
            ),
        )
