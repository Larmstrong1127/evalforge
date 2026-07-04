from uuid import UUID

from pydantic import BaseModel


class RatingCreate(BaseModel):
    prompt_version_id: UUID
    result_a_id: UUID
    result_b_id: UUID
    chosen_result_id: UUID | None = None
    skipped: bool = False
    rater_session: str | None = None


class RatingResponse(BaseModel):
    id: UUID
