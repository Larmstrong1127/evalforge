async def test_create_prompt_under_suite_starts_at_version_1(api_client):
    suite_resp = await api_client.post("/api/v1/suites", json={"name": "s", "prompts": []})
    suite_id = suite_resp.json()["id"]

    response = await api_client.post(
        f"/api/v1/suites/{suite_id}/prompts",
        json={"input_text": "q1", "expected_output": "a1"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["version_number"] == 1
    assert "prompt_id" in body


async def test_create_prompt_under_missing_suite_returns_404(api_client):
    response = await api_client.post(
        "/api/v1/suites/00000000-0000-0000-0000-000000000000/prompts",
        json={"input_text": "q"},
    )
    assert response.status_code == 404


async def test_add_version_increments_version_number(api_client):
    suite_resp = await api_client.post("/api/v1/suites", json={"name": "s", "prompts": []})
    suite_id = suite_resp.json()["id"]
    create_resp = await api_client.post(
        f"/api/v1/suites/{suite_id}/prompts", json={"input_text": "q1"}
    )
    prompt_id = create_resp.json()["prompt_id"]

    response = await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions", json={"input_text": "q1-edited"}
    )
    assert response.status_code == 201
    assert response.json()["version_number"] == 2


async def test_add_version_to_missing_prompt_returns_404(api_client):
    response = await api_client.post(
        "/api/v1/prompts/00000000-0000-0000-0000-000000000000/versions",
        json={"input_text": "q"},
    )
    assert response.status_code == 404
