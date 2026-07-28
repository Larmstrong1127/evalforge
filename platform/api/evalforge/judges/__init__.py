"""Judge plugin interface.

A judge scores one (prompt, expected, output) triple. Returning None means
"cannot judge this item" (e.g. exact_match with no expected answer) — the
runner records nothing rather than a fake zero.
"""
from dataclasses import dataclass
from typing import Protocol

from evalforge.config import Settings


@dataclass(frozen=True)
class Judgment:
    """One judge's verdict on one item.

    `score` is 0.0-1.0 by convention for *absolute* judges (exact_match,
    llm_judge, deberta-hallucination): higher is better and the number means
    something on its own.

    A judge whose model only produces a *relative* signal must not fake that
    scale. It keeps the scalar (the runner, `/compare`'s `score_delta` and the
    dashboard all want one number per output) but documents its own range and
    interpretation in its module docstring — see `reward_judge.py`, whose
    Bradley-Terry score is unbounded, has an arbitrary additive offset, and is
    only meaningful when compared against another output on the same prompt.
    Do not average scores across judges or assume a 0-1 range without checking.
    """

    score: float
    justification: str | None = None


class Judge(Protocol):
    name: str

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None: ...


def get_judge(name: str, settings: Settings) -> Judge:
    from evalforge.judges.exact_match import ExactMatchJudge
    from evalforge.judges.llm_judge import LlmJudge

    registry: dict[str, type] = {
        "exact_match": ExactMatchJudge,
        "llm_judge": LlmJudge,
    }
    if name == "deberta-hallucination":
        # Imported only on request: this judge's module needs the optional
        # `deberta` extra (torch/transformers) installed, which most users
        # of this platform won't have — importing it eagerly alongside the
        # other judges would break `get_judge` for everyone else.
        from evalforge.judges.deberta_judge import DebertaJudge

        return DebertaJudge(settings)
    if name == "reward":
        # Same lazy-import rationale as deberta-hallucination above: needs
        # the optional `reward` extra (torch/transformers) installed.
        from evalforge.judges.reward_judge import RewardJudge

        return RewardJudge(settings)
    return registry[name](settings)  # type: ignore[no-any-return]
