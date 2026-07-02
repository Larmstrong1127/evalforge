import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 502, 503}


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openai_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"openai {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"openai transport error: {exc}") from exc
        data = response.json()
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
        )
