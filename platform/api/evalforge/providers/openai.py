# TEMPORARY stub, replaced in Task 4
from evalforge.config import Settings
from evalforge.providers import Completion


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, model: str, prompt: str) -> Completion:
        raise NotImplementedError
