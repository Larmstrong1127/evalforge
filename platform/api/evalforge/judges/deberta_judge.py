"""Local hallucination-detection judge: a fine-tuned DeBERTa classifier
published to Hugging Face Hub, run entirely on this machine (free, ~30ms
per item on GPU) as an alternative to LLM-as-judge.

Requires the optional `deberta` extra (`pip install -e ".[deberta]"`) — see
platform/api/pyproject.toml. This module itself imports torch/transformers
at the top level (normal imports, will raise ImportError with a clear
message if the extra isn't installed) — that's fine because
`evalforge.judges.get_judge()` only imports this module when a caller
explicitly asks for "deberta-hallucination" by name (see judges/__init__.py),
so the base platform never pays this import cost unless someone actually
requests this judge. What IS deferred is the expensive part: the model
weights are not downloaded/loaded until the first real `score()` call that
needs them, not merely on import.

IMPORTANT — read the model card before relying on this for anything beyond
a first-pass filter: https://huggingface.co/DantheMan124/deberta-hallucination-judge
It scores F1 ~0.99 in-distribution (HaluEval) but only ~0.51 out-of-distribution
(RAGTruth, real-world hallucinations) — the benchmark in training/README.md
shows it trailing Claude/GPT-4o/Gemini meaningfully on real content. Use it to
cheaply pre-filter confidently-faithful/confidently-hallucinated cases, not
as a standalone replacement for a paid judge. Confirmed firsthand while
wiring this up: a hand-typed example outside HaluEval's exact phrasing style
("capital of France" — not from any training set) was misclassified with
high confidence, while a real HaluEval example was classified correctly
with equally high confidence — a live demonstration of the documented
generalization gap, not a bug in this file (both were run through the exact
same code path).

ENCODING (fixed 2026-08-13): this judge previously encoded each item as one
string, `Q: … C: … A: …`, truncated from the right at a hardcoded 512 tokens.
Because the answer sits at the tail, right-truncation deleted the very span
being classified — on 51% of the 200-example RAGTruth benchmark sample,
entirely. Encoding now goes through `hallucination_encoding.encode_qca`,
which budgets the CONTEXT and keeps question and answer whole, and the budget
itself is derived from the checkpoint config rather than hardcoded. Inputs
that already fit are encoded identically to before. Re-measured 2026-08-13 on
the same 200-example RAGTruth sample: ROC-AUC 0.444 → 0.603, and F1 at the
best-accuracy operating point 0.09 → 0.61. That is a RANKING-quality gain,
not an accuracy gain — best achievable accuracy is 0.605 either way against a
0.610 majority-class baseline, and accuracy at the naive 0.5 threshold gets
*worse* (0.475 → 0.385) because the fixed encoding pushes almost every score
above 0.99. RAGTruth-like traffic remains out of scope for this judge; see
training/MODEL_CARD_hallucination_judge.md.

Field mapping onto the shared (prompt, expected, output) Judge signature:
this model was trained on (question, context, answer) triples. `prompt` maps
to the question, `expected` maps to the CONTEXT to check groundedness
against (not a golden reference answer, unlike exact_match's use of
`expected`), and `output` maps to the candidate's answer. If `expected` is
None there is no context to check faithfulness against, so this judge
returns None (cannot judge), matching exact_match's convention for its own
inapplicable case — and it does so before ever loading the model.
"""
import asyncio

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evalforge.config import Settings
from evalforge.judges import Judgment
from evalforge.judges.hallucination_encoding import encode_qca

MODEL_ID = "DantheMan124/deberta-hallucination-judge"

# Sequence budget keys, in priority order — derived from the checkpoint that
# ships with the weights rather than restated as a literal here. Same
# principle as reward_judge._resolve_max_length / training/reward_metadata.py.
TRAIN_MAX_LENGTH_KEY = "train_max_length"
POSITION_LIMIT_KEY = "max_position_embeddings"
FALLBACK_MAX_LENGTH = 512


def _resolve_max_length(config: object) -> int:
    """Derive the trained sequence budget from the loaded checkpoint config."""
    for key in (TRAIN_MAX_LENGTH_KEY, POSITION_LIMIT_KEY):
        value = getattr(config, key, None)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 4096:
            return value
    return FALLBACK_MAX_LENGTH


class DebertaJudge:
    name = "deberta-hallucination"

    def __init__(self, settings: Settings) -> None:
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None
        self._max_length: int = FALLBACK_MAX_LENGTH

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        # transformers' stub for PreTrainedModel.to() appears to erase the
        # real (device | dtype | Tensor) overload down to something mypy
        # resolves incorrectly — this is standard, correct PyTorch usage
        # (move the model to the target device), not a real type error.
        self._model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()
        self._max_length = _resolve_max_length(self._model.config)

    def _score_sync(self, question: str, context: str, answer: str) -> float:
        """Synchronous inference — kept separate so it can be offloaded to a
        worker thread via asyncio.to_thread, per the platform design's ADR:
        local model inference must never block the event loop while
        concurrent cloud-judge/provider network calls are in flight. PyTorch
        releases the GIL during its C++ tensor ops, so to_thread (not a
        process pool) is sufficient here.

        Only ever called after `_ensure_loaded()` (see `score()`) — the
        asserts below are for mypy's benefit (it can't prove that ordering
        across two methods) and double as a runtime guard against a future
        refactor breaking that invariant.
        """
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        encoding = encode_qca(
            self._tokenizer,
            question,
            context,
            answer,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        p_hallucinated = float(probs[1].item())
        return p_hallucinated

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        if expected is None:
            return None
        self._ensure_loaded()
        p_hallucinated = await asyncio.to_thread(self._score_sync, prompt, expected, output)
        # Judgment.score follows the same convention as every other judge:
        # 1.0 = good/faithful, 0.0 = bad/hallucinated — the inverse of the
        # model's raw hallucination probability.
        return Judgment(score=1.0 - p_hallucinated)
