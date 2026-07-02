"""Provider plugin interface.

Every provider is a single file implementing `Provider.generate()`. The
registry maps a provider name to its class; adding a fifth provider is one
file plus one registry line.
"""
from dataclasses import dataclass
from typing import Protocol

from evalforge.config import Settings


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderError(Exception):
    """A provider call failed after exhausting its own error handling."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    name: str

    async def generate(self, model: str, prompt: str) -> Completion: ...


def get_provider(name: str, settings: Settings) -> Provider:
    from evalforge.providers.anthropic import AnthropicProvider
    from evalforge.providers.gemini import GeminiProvider
    from evalforge.providers.ollama import OllamaProvider
    from evalforge.providers.openai import OpenAIProvider

    registry: dict[str, type] = {
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    return registry[name](settings)  # type: ignore[no-any-return]
