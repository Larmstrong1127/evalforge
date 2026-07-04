from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import CandidateModel, JudgeEvaluation, Result, Run
from evalforge.db.session import get_session
from evalforge.schemas.compare import CompareResponse, CompareRow
from evalforge.schemas.runs import JudgeEvaluationResponse, ResultResponse, RunStatusResponse

router = APIRouter(tags=["compare"])


def _run_status_response(run: Run) -> RunStatusResponse:
    return RunStatusResponse(
        id=run.id,
        status=run.status.value,
        completed_steps=run.completed_steps,
        total_steps=run.total_steps,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


async def _result_response(
    session: AsyncSession, result: Result, candidates_by_id: dict[UUID, CandidateModel]
) -> ResultResponse:
    candidate = candidates_by_id[result.candidate_model_id]
    evals = (
        await session.execute(
            select(JudgeEvaluation).where(JudgeEvaluation.result_id == result.id)
        )
    ).scalars().all()
    return ResultResponse(
        id=result.id,
        prompt_version_id=result.prompt_version_id,
        candidate_model=f"{candidate.provider}:{candidate.name}",
        status=result.status.value,
        generated_text=result.generated_text,
        error=result.error,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        judge_evaluations=[
            JudgeEvaluationResponse(
                judge_name=e.judge_name, score=e.score, justification=e.justification
            )
            for e in evals
        ],
    )


@router.get("/compare", response_model=CompareResponse)
async def compare_runs(
    run_id: list[UUID] = Query(..., min_length=2, max_length=2),  # noqa: B008
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> CompareResponse:
    run_a_id, run_b_id = run_id
    run_a = await session.get(Run, run_a_id)
    run_b = await session.get(Run, run_b_id)
    if run_a is None:
        raise HTTPException(status_code=404, detail=f"run {run_a_id} not found")
    if run_b is None:
        raise HTTPException(status_code=404, detail=f"run {run_b_id} not found")

    results_a = (
        await session.execute(select(Result).where(Result.run_id == run_a_id))
    ).scalars().all()
    results_b = (
        await session.execute(select(Result).where(Result.run_id == run_b_id))
    ).scalars().all()

    by_key_a = {(r.prompt_version_id, r.candidate_model_id): r for r in results_a}
    by_key_b = {(r.prompt_version_id, r.candidate_model_id): r for r in results_b}
    all_keys = set(by_key_a) | set(by_key_b)

    candidate_ids = {candidate_model_id for _, candidate_model_id in all_keys}
    candidates_by_id = {
        c.id: c
        for c in (
            await session.execute(
                select(CandidateModel).where(CandidateModel.id.in_(candidate_ids))
            )
        )
        .scalars()
        .all()
    }

    rows: list[CompareRow] = []
    for prompt_version_id, candidate_model_id in all_keys:
        result_a = by_key_a.get((prompt_version_id, candidate_model_id))
        result_b = by_key_b.get((prompt_version_id, candidate_model_id))
        candidate = candidates_by_id[candidate_model_id]

        response_a = (
            await _result_response(session, result_a, candidates_by_id) if result_a else None
        )
        response_b = (
            await _result_response(session, result_b, candidates_by_id) if result_b else None
        )

        score_delta: dict[str, float] = {}
        if response_a and response_b:
            scores_a = {e.judge_name: e.score for e in response_a.judge_evaluations}
            scores_b = {e.judge_name: e.score for e in response_b.judge_evaluations}
            for judge_name in set(scores_a) & set(scores_b):
                score_delta[judge_name] = scores_b[judge_name] - scores_a[judge_name]

        rows.append(
            CompareRow(
                prompt_version_id=prompt_version_id,
                candidate_model=f"{candidate.provider}:{candidate.name}",
                run_a_result=response_a,
                run_b_result=response_b,
                score_delta=score_delta,
            )
        )

    return CompareResponse(
        run_a=_run_status_response(run_a), run_b=_run_status_response(run_b), rows=rows
    )
