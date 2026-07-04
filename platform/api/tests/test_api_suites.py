async def test_create_suite_returns_id_and_prompt_count(api_client):
    response = await api_client.post(
        "/api/v1/suites",
        json={
            "name": "demo-qa",
            "description": "a demo suite",
            "prompts": [
                {"input_text": "What is 2+2?", "expected_output": "4"},
                {"input_text": "Capital of France?", "expected_output": "Paris"},
            ],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "demo-qa"
    assert body["description"] == "a demo suite"
    assert body["prompt_count"] == 2
    assert "id" in body


async def test_create_suite_requires_name(api_client):
    response = await api_client.post("/api/v1/suites", json={"prompts": []})
    assert response.status_code == 422


async def test_list_suites_returns_created_suites(api_client):
    await api_client.post("/api/v1/suites", json={"name": "suite-a", "prompts": []})
    await api_client.post("/api/v1/suites", json={"name": "suite-b", "prompts": []})

    response = await api_client.get("/api/v1/suites")
    assert response.status_code == 200
    names = {s["name"] for s in response.json()}
    assert names == {"suite-a", "suite-b"}


async def test_list_suites_reports_prompt_count(api_client):
    await api_client.post(
        "/api/v1/suites",
        json={
            "name": "suite-with-prompts",
            "prompts": [
                {"input_text": "a", "expected_output": None},
                {"input_text": "b", "expected_output": None},
            ],
        },
    )

    response = await api_client.get("/api/v1/suites")
    assert response.status_code == 200
    suite = next(s for s in response.json() if s["name"] == "suite-with-prompts")
    assert suite["prompt_count"] == 2
