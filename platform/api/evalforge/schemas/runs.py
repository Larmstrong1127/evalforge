from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    suite_id: UUID
    candidates: list[str]
    judges: list[str] = ["exact_match"]
    concurrency: int = Field(default=3, ge=1, le=20)


class RunAccepted(BaseModel):
    run_id: UUID


class RunStatusResponse(BaseModel):
    id: UUID
    status: str
    completed_steps: int
    total_steps: int
    started_at: datetime | None
    finished_at: datetime | None


class JudgeEvaluationResponse(BaseModel):
    judge_name: str
    score: float
    justification: str | None


class ResultResponse(BaseModel):
    id: UUID
    prompt_version_id: UUID
    candidate_model: str
    status: str
    generated_text: str
    error: str | None
    latency_ms: int
    cost_usd: float
    judge_evaluations: list[JudgeEvaluationResponse]


class CostResponse(BaseModel):
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    by_candidate: dict[str, float]
