import httpx
import pytest
import respx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError, get_provider
from evalforge.providers.anthropic import AnthropicProvider
from evalforge.providers.gemini import GeminiProvider
from evalforge.providers.ollama import OllamaProvider
from evalforge.providers.openai import OpenAIProvider

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


@respx.mock
async def test_anthropic_generate():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "4"}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )
    )
    provider = AnthropicProvider(SETTINGS)
    completion = await provider.generate(model="claude-sonnet-5", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.input_tokens == 12
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "test-key"


@respx.mock
async def test_anthropic_429_is_retryable():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    provider = AnthropicProvider(SETTINGS)
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(model="claude-sonnet-5", prompt="q")
    assert excinfo.value.retryable is True


@respx.mock
async def test_anthropic_401_is_not_retryable():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    provider = AnthropicProvider(SETTINGS)
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(model="claude-sonnet-5", prompt="q")
    assert excinfo.value.retryable is False


@respx.mock
async def test_openai_generate():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "4"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )
    )
    provider = OpenAIProvider(SETTINGS)
    completion = await provider.generate(model="gpt-4o-mini", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.output_tokens == 3


@respx.mock
async def test_gemini_generate():
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "4"}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
            },
        )
    )
    provider = GeminiProvider(SETTINGS)
    completion = await provider.generate(model="gemini-2.0-flash", prompt="What is 2+2?")
    assert completion.text == "4"
    assert completion.input_tokens == 12
