"""Async eval runner.

Design (ADR-001): plain asyncio, no Celery/Redis. An eval run is I/O-bound
fan-out over provider APIs; a semaphore per run bounds concurrency, and the
job state lives in the runs table. Per-result failures never fail the whole
run — partial data is still data. Interrupted runs (process killed mid-flight)
are not resumed — completed_steps/status stay wherever they were left; this
is an accepted v1 tradeoff of the asyncio-not-Celery design (ADR-001).
"""
import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
)
from evalforge.judges import Judge, Judgment
from evalforge.pricing import cost_usd
from evalforge.providers import Provider, ProviderError


@dataclass
class RunConfig:
    providers: dict[str, Provider]
    judges: list[Judge] = field(default_factory=list)
    max_retries: int = 3
    retry_base_delay: float = 1.0


async def _latest_versions(session: AsyncSession, suite_id: object) -> list[PromptVersion]:
    """Latest version of each prompt in the suite."""
    rows = (
        (
            await session.execute(
                select(PromptVersion)
                .join(PromptVersion.prompt)
                .where(PromptVersion.prompt.has(suite_id=suite_id))
                .order_by(PromptVersion.prompt_id, PromptVersion.version_number.desc())
            )
        )
        .scalars()
        .all()
    )
    latest: dict[object, PromptVersion] = {}
    for v in rows:
        latest.setdefault(v.prompt_id, v)
    return list(latest.values())


async def _generate_with_retries(
    provider: Provider, model: str, prompt: str, config: RunConfig
) -> tuple[str, int, int, int, str | None]:
    """Returns (text, input_tokens, output_tokens, latency_ms, error)."""
    attempt = 0
    while True:
        start = asyncio.get_running_loop().time()
        try:
            completion = await provider.generate(model=model, prompt=prompt)
            latency_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return (
                completion.text,
                completion.input_tokens,
                completion.output_tokens,
                latency_ms,
                None,
            )
        except ProviderError as exc:
            if not exc.retryable or attempt >= config.max_retries:
                return "", 0, 0, 0, str(exc)
            await asyncio.sleep(config.retry_base_delay * (2**attempt))
            attempt += 1


async def _process_one(
    session: AsyncSession,
    run: Run,
    version: PromptVersion,
    candidate: CandidateModel,
    config: RunConfig,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
) -> None:
    try:
        async with semaphore:
            provider = config.providers[candidate.provider]
            text, tokens_in, tokens_out, latency_ms, error = await _generate_with_retries(
                provider, candidate.name, version.input_text, config
            )

        judgments: list[tuple[Judge, Judgment]] = []
        if not error:
            for judge in config.judges:
                try:
                    judgment = await judge.score(
                        prompt=version.input_text,
                        expected=version.expected_output,
                        output=text,
                    )
                except Exception:
                    # A misbehaving judge must not fail the whole run or drop the
                    # underlying Result — just skip recording this judge's evaluation.
                    continue
                if judgment is not None:
                    judgments.append((judge, judgment))

        # ORM object construction mutates shared relationship collections (e.g.
        # run.results via backref), so it must happen under the same lock as the
        # commit — constructing concurrently with another task's flush/commit
        # triggers SQLAlchemy's "collection append during flush" race.
        async with lock:  # AsyncSession is not concurrency-safe
            result = Result(
                run=run,
                prompt_version=version,
                candidate_model=candidate,
                status=ResultStatus.FAILED if error else ResultStatus.OK,
                generated_text=text,
                error=error,
                latency_ms=latency_ms,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=cost_usd(candidate.name, tokens_in, tokens_out),
            )
            evaluations = [
                JudgeEvaluation(
                    result=result,
                    judge_name=judge.name,
                    score=judgment.score,
                    justification=judgment.justification,
                )
                for judge, judgment in judgments
            ]
            session.add(result)
            session.add_all(evaluations)
            run.completed_steps += 1
            await session.commit()
    except Exception as exc:
        # Safety net: anything unexpected that slips past the fine-grained
        # handling above (unregistered provider, a non-ProviderError bug in a
        # provider, a pricing lookup blowing up, an ORM construction error,
        # etc.) must still yield exactly one FAILED Result for this item and
        # must never propagate out of _process_one — a single item failing
        # must never fail the whole run (see module docstring).
        #
        # We don't know whether `lock` is already held at the point the
        # exception was raised (e.g. cost_usd() can raise from inside the
        # `async with lock:` block above), so re-entering it here would
        # deadlock. asyncio.Lock releases automatically when its `async with`
        # block is exited via exception, so by the time we're in this except
        # clause the lock is guaranteed to be free — acquiring it again here
        # is always safe and never double-acquires.
        try:
            async with lock:
                # If the happy-path commit above raised (DB constraint,
                # serialization failure, etc.), the session is left in a
                # failed-transaction/pending-rollback state. Roll it back
                # before touching it again, otherwise the fallback commit
                # below will itself raise on a poisoned session.
                #
                # rollback() expires every ORM object in the session,
                # including `run`. A plain `run.completed_steps += 1`
                # after that would trigger an implicit (unawaited) lazy
                # refresh outside of any SQLAlchemy-awaited call, which
                # raises MissingGreenlet. Explicitly await a refresh first
                # so `run`'s attributes are safe to read/mutate again.
                await session.rollback()
                await session.refresh(run)
                result = Result(
                    run=run,
                    prompt_version=version,
                    candidate_model=candidate,
                    status=ResultStatus.FAILED,
                    generated_text="",
                    error=str(exc)[:500],
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                )
                session.add(result)
                run.completed_steps += 1
                await session.commit()
        except Exception:
            # Even the fallback commit failed (session unrecoverable for
            # this item). Roll back again so sibling tasks sharing this
            # session aren't left poisoned, and give up recording this one
            # result rather than crashing the whole run.
            async with lock:
                await session.rollback()


async def execute_run(
    session: AsyncSession, run: Run, candidates: list[CandidateModel], config: RunConfig
) -> None:
    versions = await _latest_versions(session, run.suite_id)
    run.total_steps = len(versions) * len(candidates)
    run.status = RunStatus.RUNNING
    run.started_at = datetime.now(UTC)
    await session.commit()

    semaphore = asyncio.Semaphore(run.concurrency_limit)
    lock = asyncio.Lock()
    tasks = [
        _process_one(session, run, version, candidate, config, semaphore, lock)
        for version in versions
        for candidate in candidates
    ]
    try:
        await asyncio.gather(*tasks)
        run.status = RunStatus.COMPLETED
    except Exception:
        run.status = RunStatus.FAILED
        raise
    finally:
        run.finished_at = datetime.now(UTC)
        await session.commit()
