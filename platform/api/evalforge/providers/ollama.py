import httpx

from evalforge.config import Settings
from evalforge.providers import Completion, ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url

    async def generate(self, model: str, prompt: str) -> Completion:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code >= 500 or exc.response.status_code == 429
                raise ProviderError(
                    f"ollama {exc.response.status_code}: {exc.response.text[:200]}",
                    retryable=retryable,
                ) from exc
            except httpx.TransportError as exc:
                raise ProviderError(f"ollama transport error: {exc}") from exc
        data = response.json()
        return Completion(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
