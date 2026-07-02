import asyncio

from sqlalchemy import select

from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)
from evalforge.judges import Judgment
from evalforge.providers import Completion, ProviderError
from evalforge.runner import RunConfig, execute_run


class FakeProvider:
    name = "fake"

    def __init__(self, fail_times: int = 0, track: list | None = None) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.track = track
        self.active = 0
        self.max_active = 0

    async def generate(self, model: str, prompt: str) -> Completion:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.calls <= self.fail_times:
                raise ProviderError("transient", retryable=True)
            return Completion(text=f"answer to {prompt}", input_tokens=10, output_tokens=5)
        finally:
            self.active -= 1


class FakeJudge:
    name = "fake_judge"

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        return Judgment(score=0.5)


async def make_fixture(session, n_prompts: int = 4):
    suite = Suite(name="s")
    model = CandidateModel(name="fake-model", provider="fake")
    versions = []
    for i in range(n_prompts):
        p = Prompt(suite=suite)
        v = PromptVersion(prompt=p, version_number=1, input_text=f"q{i}", expected_output=f"a{i}")
        versions.append(v)
        session.add_all([p, v])
    run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=2)
    session.add_all([suite, model, run])
    await session.commit()
    return run, model, versions


async def test_run_produces_results_and_evaluations(session):
    run, model, versions = await make_fixture(session)
    provider = FakeProvider()
    config = RunConfig(providers={"fake": provider}, judges=[FakeJudge()], max_retries=2)
    await execute_run(session, run, [model], config)

    results = (await session.execute(select(Result))).scalars().all()
    evals = (await session.execute(select(JudgeEvaluation))).scalars().all()
    assert len(results) == 4
    assert all(r.status is ResultStatus.OK for r in results)
    assert len(evals) == 4
    assert run.status is RunStatus.COMPLETED
    assert run.completed_steps == run.total_steps == 4


async def test_concurrency_is_bounded(session):
    run, model, _ = await make_fixture(session, n_prompts=6)
    provider = FakeProvider()
    config = RunConfig(providers={"fake": provider}, judges=[], max_retries=0)
    await execute_run(session, run, [model], config)
    assert provider.max_active <= run.concurrency_limit


async def test_transient_failure_is_retried(session):
    run, model, _ = await make_fixture(session, n_prompts=1)
    provider = FakeProvider(fail_times=2)
    config = RunConfig(
        providers={"fake": provider}, judges=[], max_retries=3, retry_base_delay=0.0
    )
    await execute_run(session, run, [model], config)
    results = (await session.execute(select(Result))).scalars().all()
    assert results[0].status is ResultStatus.OK
    assert provider.calls == 3  # 2 failures + 1 success


async def test_exhausted_retries_mark_result_failed_not_run(session):
    run, model, _ = await make_fixture(session, n_prompts=2)
    provider = FakeProvider(fail_times=100)
    config = RunConfig(
        providers={"fake": provider}, judges=[], max_retries=1, retry_base_delay=0.0
    )
    await execute_run(session, run, [model], config)
    results = (await session.execute(select(Result))).scalars().all()
    assert all(r.status is ResultStatus.FAILED for r in results)
    assert all("transient" in (r.error or "") for r in results)
    assert run.status is RunStatus.COMPLETED  # partial failure ≠ run failure
