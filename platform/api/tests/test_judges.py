import httpx
import pytest
import respx

from evalforge.config import Settings
from evalforge.judges import get_judge
from evalforge.judges.exact_match import ExactMatchJudge
from evalforge.judges.llm_judge import LlmJudge

SETTINGS = Settings(anthropic_api_key="test-key")


def test_registry():
    assert isinstance(get_judge("exact_match", SETTINGS), ExactMatchJudge)
    assert isinstance(get_judge("llm_judge", SETTINGS), LlmJudge)
    with pytest.raises(KeyError):
        get_judge("nope", SETTINGS)


async def test_exact_match_scores_one_for_match():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="2+2?", expected="4", output="4")
    assert judgment.score == 1.0


async def test_exact_match_is_case_and_whitespace_insensitive():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected="Paris", output="  paris \n")
    assert judgment.score == 1.0


async def test_exact_match_scores_zero_for_mismatch():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected="4", output="5")
    assert judgment.score == 0.0


async def test_exact_match_without_expected_returns_none():
    judge = ExactMatchJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected=None, output="anything")
    assert judgment is None


@respx.mock
async def test_llm_judge_parses_score_and_justification():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": '{"score": 0.8, "justification": "mostly correct"}'}
                ],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )
    )
    judge = LlmJudge(SETTINGS)
    judgment = await judge.score(prompt="2+2?", expected="4", output="four")
    assert judgment is not None
    assert judgment.score == 0.8
    assert judgment.justification == "mostly correct"


@respx.mock
async def test_llm_judge_malformed_json_raises():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "I think it's pretty good!"}],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )
    )
    judge = LlmJudge(SETTINGS)
    with pytest.raises(ValueError):
        await judge.score(prompt="q", expected="4", output="four")
