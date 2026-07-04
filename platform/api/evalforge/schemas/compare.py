from uuid import UUID

from pydantic import BaseModel

from evalforge.schemas.runs import ResultResponse, RunStatusResponse


class CompareRow(BaseModel):
    prompt_version_id: UUID
    candidate_model: str
    run_a_result: ResultResponse | None
    run_b_result: ResultResponse | None
    score_delta: dict[str, float]


class CompareResponse(BaseModel):
    run_a: RunStatusResponse
    run_b: RunStatusResponse
    rows: list[CompareRow]
