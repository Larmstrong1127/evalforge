from uuid import UUID

from pydantic import BaseModel


class PromptVersionResponse(BaseModel):
    prompt_id: UUID
    version_number: int
