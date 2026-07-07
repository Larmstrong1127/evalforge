"""POST /runs must never hand its request-scoped session to the background
task — see the module-level note on `_run_in_background` below and
docs/superpowers/specs/2026-07-03-phase3a-fastapi-backend-design.md section
4 for why: the request's AsyncSession is torn down as part of the request/
response lifecycle, and BackgroundTasks execute after the response is sent.
Sharing it produces a closed-session/detached-instance error the first time
the background task touches the ORM objects.
"""
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evalforge.api.params import parse_uuid_or_404
from evalforge.config import Settings
from evalforge.db.engine import make_engine, make_session_factory
from evalforge.db.models import CandidateModel, JudgeEvaluation, Result, Run, RunStatus, Suite
from evalforge.db.session import get_session
from evalforge.judges import get_judge
from evalforge.providers import get_provider
from evalforge.runner import RunConfig, execute_run
from evalforge.schemas.runs import (
    CostResponse,
    JudgeEvaluationResponse,
    ResultResponse,
    RunAccepted,
    RunCreate,
    RunStatusResponse,
)

router = APIRouter(tags=["runs"])
logger = logging.getLogger(__name__)


def _make_background_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Extracted into its own function (rather than inlined in
    _run_in_background) purely so tests have a seam to monkeypatch: SQLite
    `:memory:` databases are unique per engine, so a real call to
    make_engine(settings) in a test would connect to an empty database the
    test's own session/api_client fixtures never touched. Tests monkeypatch
    this exact function to return the shared in-memory session_factory
    fixture instead — see tests/test_api_runs.py."""
    return make_session_factory(make_engine(settings))


async def _run_in_background(run_id: str, candidate_ids: list[str], judge_names: list[str]) -> None:
    """Opens its OWN session — never reuses the request's. Re-fetches the
    Run and CandidateModel rows by id inside that session."""
    settings = Settings()
    factory = _make_background_session_factory(settings)
    async with factory() as session:
        # session.get() requires an actual uuid.UUID (not a str) for
        # SQLAlchemy's Uuid column type here — it's emulated on SQLite via a
        # binary/hex representation, and the type processor calls .hex on
        # whatever value it's given, which a plain str doesn't have.
        run = await session.get(Run, uuid.UUID(run_id))
        assert run is not None  # the row was just created by create_run in the same DB
        candidates: list[CandidateModel] = []
        for cid in candidate_ids:
            candidate = await session.get(CandidateModel, uuid.UUID(cid))
            assert candidate is not None  # created moments ago by create_run
            candidates.append(candidate)
        providers = {c.provider: get_provider(c.provider, settings) for c in candidates}
        judges = [get_judge(name, settings) for name in judge_names]
        config = RunConfig(providers=providers, judges=judges)
        try:
            await execute_run(session, run, candidates, config)
        except Exception:
            # execute_run() already sets status=FAILED and commits before
            # re-raising (see runner.py). This catch exists purely so the
            # failure is visible in the API server's own logs — Starlette
            # would otherwise swallow an exception escaping a BackgroundTask
            # with only an easy-to-miss ASGI-server-level log line.
            logger.exception("run %s failed", run_id)
    # No explicit engine.dispose() here: _make_background_session_factory's
    # engine isn't exposed to this function (by design — see its docstring,
    # this keeps the test monkeypatch seam trivial, since a mocked version
    # only needs to return a factory, not a factory+engine pair to manage).
    # A short-lived per-run engine is GC'd rather than explicitly disposed;
    # an accepted minor tradeoff, not a resource leak in the connection-pool
    # sense (SQLite/aiosqlite hold no persistent server-side connection slot).


@router.post("/runs", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: RunCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> RunAccepted:
    settings = Settings()
    suite = await session.get(Suite, body.suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {body.suite_id} not found")

    candidates: list[CandidateModel] = []
    for spec in body.candidates:
        provider_name, _, model_name = spec.partition(":")
        if not model_name:
            raise HTTPException(
                status_code=400, detail=f"candidate '{spec}' must be provider:model"
            )
        try:
            get_provider(provider_name, settings)
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown provider '{provider_name}'"
            ) from exc
        candidates.append(CandidateModel(name=model_name, provider=provider_name))
    session.add_all(candidates)

    for name in body.judges:
        try:
            get_judge(name, settings)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"unknown judge '{name}'") from exc

    run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=body.concurrency)
    session.add(run)
    # Must be commit(), not flush(): BackgroundTasks execute inside the same
    # ASGI middleware layer that later closes the get_session dependency's
    # AsyncExitStack, so the dependency's post-yield commit runs AFTER the
    # background task has already started — a flush()-only row is visible
    # only within this session's own uncommitted transaction, and the
    # background task's freshly-opened session sees nothing. Confirmed via a
    # real end-to-end run against local Ollama (this bug does not reproduce
    # under the mocked test suite, since tests intentionally share one
    # session_factory between the request and the "background task").
    await session.commit()

    candidate_ids = [str(c.id) for c in candidates]
    background_tasks.add_task(_run_in_background, str(run.id), candidate_ids, body.judges)

    return RunAccepted(run_id=run.id)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)) -> RunStatusResponse:  # noqa: B008
    run_uuid = parse_uuid_or_404(run_id, "run")
    run = await session.get(Run, run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return RunStatusResponse(
        id=run.id,
        status=run.status.value,
        completed_steps=run.completed_steps,
        total_steps=run.total_steps,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.get("/runs/{run_id}/results", response_model=list[ResultResponse])
async def get_run_results(
    run_id: str,
    judge_name: str | None = None,
    status_filter: str | None = None,  # query param name is "status_filter", not "status" —
    # avoids shadowing the `status` module imported from fastapi at the top of this file.
    limit: int = 1000,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> list[ResultResponse]:
    if limit > 5000:
        limit = 5000
    run_uuid = parse_uuid_or_404(run_id, "run")
    run = await session.get(Run, run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    query = select(Result).where(Result.run_id == run_uuid)
    if status_filter is not None:
        query = query.where(Result.status == status_filter)
    query = query.limit(limit)
    results = (await session.execute(query)).scalars().all()

    candidate_ids = {r.candidate_model_id for r in results}
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

    responses: list[ResultResponse] = []
    for r in results:
        evals_query = select(JudgeEvaluation).where(JudgeEvaluation.result_id == r.id)
        if judge_name is not None:
            evals_query = evals_query.where(JudgeEvaluation.judge_name == judge_name)
        evals = (await session.execute(evals_query)).scalars().all()
        if judge_name is not None and not evals:
            continue
        candidate = candidates_by_id[r.candidate_model_id]
        responses.append(
            ResultResponse(
                id=r.id,
                prompt_version_id=r.prompt_version_id,
                candidate_model=f"{candidate.provider}:{candidate.name}",
                status=r.status.value,
                generated_text=r.generated_text,
                error=r.error,
                latency_ms=r.latency_ms,
                cost_usd=r.cost_usd,
                judge_evaluations=[
                    JudgeEvaluationResponse(
                        judge_name=e.judge_name, score=e.score, justification=e.justification
                    )
                    for e in evals
                ],
            )
        )
    return responses


@router.get("/runs/{run_id}/costs", response_model=CostResponse)
async def get_run_costs(run_id: str, session: AsyncSession = Depends(get_session)) -> CostResponse:  # noqa: B008
    run_uuid = parse_uuid_or_404(run_id, "run")
    run = await session.get(Run, run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    rows = (
        await session.execute(
            select(
                Result.candidate_model_id,
                Result.cost_usd,
                Result.input_tokens,
                Result.output_tokens,
            ).where(Result.run_id == run_uuid)
        )
    ).all()

    candidate_ids = {row.candidate_model_id for row in rows}
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

    by_candidate: dict[str, float] = {}
    total_cost = 0.0
    total_in = 0
    total_out = 0
    for row in rows:
        candidate = candidates_by_id[row.candidate_model_id]
        key = f"{candidate.provider}:{candidate.name}"
        by_candidate[key] = by_candidate.get(key, 0.0) + row.cost_usd
        total_cost += row.cost_usd
        total_in += row.input_tokens
        total_out += row.output_tokens

    return CostResponse(
        total_cost_usd=total_cost,
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        by_candidate=by_candidate,
    )


@router.get("/suites/{suite_id}/runs", response_model=list[RunStatusResponse])
async def list_suite_runs(
    suite_id: str, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> list[RunStatusResponse]:
    suite_uuid = parse_uuid_or_404(suite_id, "suite")
    suite = await session.get(Suite, suite_uuid)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id} not found")
    runs = (
        await session.execute(select(Run).where(Run.suite_id == suite_uuid))
    ).scalars().all()
    return [
        RunStatusResponse(
            id=r.id,
            status=r.status.value,
            completed_steps=r.completed_steps,
            total_steps=r.total_steps,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in runs
    ]
