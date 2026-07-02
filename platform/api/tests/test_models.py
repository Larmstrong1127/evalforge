import uuid

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


async def test_suite_prompt_version_chain(session):
    suite = Suite(name="qa-basics")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(
        prompt=prompt, version_number=1, input_text="What is 2+2?", expected_output="4"
    )
    session.add_all([suite, prompt, v1])
    await session.commit()

    fetched = (await session.execute(select(PromptVersion))).scalar_one()
    assert fetched.input_text == "What is 2+2?"
    assert fetched.prompt.suite.name == "qa-basics"


async def test_editing_prompt_appends_new_version(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="old")
    v2 = PromptVersion(prompt=prompt, version_number=2, input_text="new")
    session.add_all([suite, prompt, v1, v2])
    await session.commit()

    versions = (
        await session.execute(select(PromptVersion).order_by(PromptVersion.version_number))
    ).scalars().all()
    assert [v.input_text for v in versions] == ["old", "new"]
    assert v1.input_text == "old"  # old version untouched


async def test_result_and_judge_evaluations_are_separate(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="llama3.2", provider="ollama")
    run = Run(suite=suite, status=RunStatus.RUNNING, concurrency_limit=3)
    result = Result(
        run=run, prompt_version=v1, candidate_model=model,
        status=ResultStatus.OK, generated_text="a", latency_ms=120,
        input_tokens=10, output_tokens=5, cost_usd=0.0,
    )
    j1 = JudgeEvaluation(result=result, judge_name="exact_match", score=1.0)
    j2 = JudgeEvaluation(
        result=result, judge_name="llm_judge:claude", score=0.9, justification="matches"
    )
    session.add_all([suite, prompt, v1, model, run, result, j1, j2])
    await session.commit()

    evals = (await session.execute(select(JudgeEvaluation))).scalars().all()
    assert len(evals) == 2
    assert {e.result_id for e in evals} == {result.id}


async def test_failed_result_records_error(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    v1 = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="gpt-4o", provider="openai")
    run = Run(suite=suite, status=RunStatus.RUNNING, concurrency_limit=3)
    result = Result(
        run=run, prompt_version=v1, candidate_model=model,
        status=ResultStatus.FAILED, generated_text="", latency_ms=0,
        input_tokens=0, output_tokens=0, cost_usd=0.0, error="429 rate limited",
    )
    session.add_all([suite, prompt, v1, model, run, result])
    await session.commit()
    fetched = (await session.execute(select(Result))).scalar_one()
    assert fetched.status is ResultStatus.FAILED
    assert "429" in fetched.error


async def test_ids_are_uuids(session):
    suite = Suite(name="s")
    session.add(suite)
    await session.commit()
    assert isinstance(suite.id, uuid.UUID)
