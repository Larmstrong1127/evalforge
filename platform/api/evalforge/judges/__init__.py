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
    score: float  # 0.0 - 1.0
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
    return registry[name](settings)  # type: ignore[no-any-return]
