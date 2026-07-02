import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 502, 503, 529}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.anthropic_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"anthropic {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"anthropic transport error: {exc}") from exc
        data = response.json()
        try:
            text = "".join(
                block["text"] for block in data["content"] if block["type"] == "text"
            )
            return Completion(
                text=text,
                input_tokens=data["usage"]["input_tokens"],
                output_tokens=data["usage"]["output_tokens"],
            )
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"anthropic returned unexpected response shape: {exc}", retryable=False
            ) from exc
