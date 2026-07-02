import httpx
import pytest
import respx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError, get_provider
from evalforge.providers.ollama import OllamaProvider

SETTINGS = Settings(
    anthropic_api_key="test-key", openai_api_key="test-key", gemini_api_key="test-key"
)


def test_registry_returns_known_providers():
    assert isinstance(get_provider("ollama", SETTINGS), OllamaProvider)


def test_registry_rejects_unknown_provider():
    with pytest.raises(KeyError):
        get_provider("nonexistent", SETTINGS)


@respx.mock
async def test_ollama_generate_returns_completion():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "The answer is 4.",
                "prompt_eval_count": 12,
                "eval_count": 6,
            },
        )
    )
    provider = OllamaProvider(SETTINGS)
    completion = await provider.generate(model="llama3.2", prompt="What is 2+2?")
    assert isinstance(completion, Completion)
    assert completion.text == "The answer is 4."
    assert completion.input_tokens == 12
    assert completion.output_tokens == 6


@respx.mock
async def test_ollama_http_error_raises_provider_error():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(500, text="boom")
    )
    provider = OllamaProvider(SETTINGS)
    with pytest.raises(ProviderError):
        await provider.generate(model="llama3.2", prompt="q")
