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


async def test_compare_two_runs_matches_by_prompt_version_and_candidate(api_client, session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    version = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model = CandidateModel(name="model-a", provider="fake")

    run_a = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    run_b = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)

    result_a = Result(
        run=run_a, prompt_version=version, candidate_model=model,
        status=ResultStatus.OK, generated_text="old answer", latency_ms=10,
    )
    result_b = Result(
        run=run_b, prompt_version=version, candidate_model=model,
        status=ResultStatus.OK, generated_text="new answer", latency_ms=10,
    )
    eval_a = JudgeEvaluation(result=result_a, judge_name="exact_match", score=1.0)
    eval_b = JudgeEvaluation(result=result_b, judge_name="exact_match", score=0.0)

    session.add_all(
        [suite, prompt, version, model, run_a, run_b, result_a, result_b, eval_a, eval_b]
    )
    await session.commit()

    response = await api_client.get(
        f"/api/v1/compare?run_id={run_a.id}&run_id={run_b.id}"
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["run_a_result"]["generated_text"] == "old answer"
    assert row["run_b_result"]["generated_text"] == "new answer"
    assert row["score_delta"]["exact_match"] == -1.0


async def test_compare_requires_exactly_two_run_ids(api_client):
    response = await api_client.get("/api/v1/compare?run_id=00000000-0000-0000-0000-000000000000")
    assert response.status_code == 422


async def test_compare_with_missing_run_returns_404(api_client, session):
    suite = Suite(name="s")
    run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    session.add_all([suite, run])
    await session.commit()

    response = await api_client.get(
        f"/api/v1/compare?run_id={run.id}&run_id=00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404
