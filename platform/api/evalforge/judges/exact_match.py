from evalforge.config import Settings
from evalforge.judges import Judgment


class ExactMatchJudge:
    name = "exact_match"

    def __init__(self, settings: Settings) -> None:
        pass

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        if expected is None:
            return None
        match = expected.strip().casefold() == output.strip().casefold()
        return Judgment(score=1.0 if match else 0.0)
