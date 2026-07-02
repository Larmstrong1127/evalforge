import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError

_RETRYABLE = {429, 500, 503}


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key

    async def generate(self, model: str, prompt: str) -> Completion:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"gemini {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=exc.response.status_code in _RETRYABLE,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"gemini transport error: {exc}") from exc
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            usage = data.get("usageMetadata", {})
            return Completion(
                text="".join(p.get("text", "") for p in parts),
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
            )
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"gemini returned unexpected response shape: {exc}", retryable=False
            ) from exc
