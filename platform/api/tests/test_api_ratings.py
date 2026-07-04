from evalforge.db.models import (
    CandidateModel,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)


async def _make_two_results(session):
    suite = Suite(name="s")
    prompt = Prompt(suite=suite)
    version = PromptVersion(prompt=prompt, version_number=1, input_text="q")
    model_a = CandidateModel(name="model-a", provider="fake")
    model_b = CandidateModel(name="model-b", provider="fake")
    run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1)
    result_a = Result(
        run=run, prompt_version=version, candidate_model=model_a,
        status=ResultStatus.OK, generated_text="answer A", latency_ms=10,
    )
    result_b = Result(
        run=run, prompt_version=version, candidate_model=model_b,
        status=ResultStatus.OK, generated_text="answer B", latency_ms=10,
    )
    session.add_all([suite, prompt, version, model_a, model_b, run, result_a, result_b])
    await session.commit()
    return version, result_a, result_b


async def test_create_rating_records_chosen_result(api_client, session):
    version, result_a, result_b = await _make_two_results(session)
    response = await api_client.post(
        "/api/v1/ratings",
        json={
            "prompt_version_id": str(version.id),
            "result_a_id": str(result_a.id),
            "result_b_id": str(result_b.id),
            "chosen_result_id": str(result_a.id),
        },
    )
    assert response.status_code == 201
    assert "id" in response.json()


async def test_create_rating_allows_skip_with_no_choice(api_client, session):
    version, result_a, result_b = await _make_two_results(session)
    response = await api_client.post(
        "/api/v1/ratings",
        json={
            "prompt_version_id": str(version.id),
            "result_a_id": str(result_a.id),
            "result_b_id": str(result_b.id),
            "skipped": True,
        },
    )
    assert response.status_code == 201
