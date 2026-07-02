import json

from typer.testing import CliRunner

from evalforge.cli import app

runner = CliRunner()


def test_suite_create_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "name": "cli-test",
                "prompts": [{"input_text": "q1", "expected_output": "a1"}],
            }
        )
    )
    result = runner.invoke(app, ["suite", "create", str(suite_file)])
    assert result.exit_code == 0
    assert "cli-test" in result.output

    result = runner.invoke(app, ["suite", "list"])
    assert result.exit_code == 0
    assert "cli-test" in result.output


def test_suite_create_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    result = runner.invoke(app, ["suite", "create", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
