"""LLM-as-judge using Claude.

The judge model returns strict JSON {"score": float, "justification": str}.
Malformed judge output raises ValueError — a judge that silently returns a
default score would corrupt the dataset.
"""
import json

from evalforge.config import Settings
from evalforge.judges import Judgment
from evalforge.providers.anthropic import AnthropicProvider

_JUDGE_MODEL = "claude-sonnet-5"


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop opening fence (with optional language tag)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()

_PROMPT_TEMPLATE = """You are an evaluation judge. Score the RESPONSE to the QUESTION \
on a scale of 0.0 to 1.0.
{expected_block}
QUESTION:
{prompt}

RESPONSE:
{output}

Reply with strict JSON only, no markdown fences: \
{{"score": <float 0-1>, "justification": "<one sentence>"}}"""


class LlmJudge:
    name = "llm_judge"

    def __init__(self, settings: Settings) -> None:
        self._provider = AnthropicProvider(settings)

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        expected_block = f"\nREFERENCE ANSWER:\n{expected}\n" if expected else ""
        judge_prompt = _PROMPT_TEMPLATE.format(
            expected_block=expected_block, prompt=prompt, output=output
        )
        completion = await self._provider.generate(model=_JUDGE_MODEL, prompt=judge_prompt)
        try:
            data = json.loads(_strip_markdown_fence(completion.text))
            score = float(data["score"])
            justification = str(data["justification"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"judge returned malformed output: {completion.text[:200]}") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"judge score out of range: {score}")
        return Judgment(score=score, justification=justification)
