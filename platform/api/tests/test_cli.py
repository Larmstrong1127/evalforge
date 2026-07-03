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


def _create_suite(tmp_path, monkeypatch):
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


def test_run_rejects_unknown_provider(tmp_path, monkeypatch):
    _create_suite(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "cli-test", "-c", "bogus:model"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "unknown provider" in result.output
    assert "bogus" in result.output


def test_run_rejects_unknown_judge(tmp_path, monkeypatch):
    _create_suite(tmp_path, monkeypatch)
    result = runner.invoke(app, ["run", "cli-test", "-c", "ollama:x", "-j", "bogus_judge"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "unknown judge" in result.output
    assert "bogus_judge" in result.output


def test_results_rejects_invalid_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    result = runner.invoke(app, ["results", "not-a-uuid"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "not a valid run id" in result.output


def test_results_accepts_valid_uuid_with_no_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALFORGE_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/t.db")
    result = runner.invoke(app, ["results", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 0
    assert "Traceback" not in result.output
