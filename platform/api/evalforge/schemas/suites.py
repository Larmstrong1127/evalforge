from uuid import UUID

from pydantic import BaseModel


class PromptCreate(BaseModel):
    input_text: str
    expected_output: str | None = None


class SuiteCreate(BaseModel):
    name: str
    description: str | None = None
    prompts: list[PromptCreate] = []


class SuiteResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    prompt_count: int
