"""CI eval gate: runs a fixed suite against a pinned local Ollama model and
fails the build if the average exact_match score regresses past a tolerance
from the recorded baseline.

Usage: python .github/scripts/eval_gate.py [--update-baseline]

Requires Ollama running locally with the pinned model pulled (see
`PINNED_MODEL` below); the eval-gate.yml workflow sets this up in CI via a
service container. Uses a throwaway SQLite DB, never the dev/demo database.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform" / "api"))

from evalforge.config import Settings  # noqa: E402
from evalforge.db.engine import init_db, make_engine, make_session_factory  # noqa: E402
from evalforge.db.models import CandidateModel, JudgeEvaluation, Prompt, PromptVersion, Run, RunStatus, Suite  # noqa: E402
from evalforge.judges import get_judge  # noqa: E402
from evalforge.providers import get_provider  # noqa: E402
from evalforge.runner import RunConfig, execute_run  # noqa: E402

PINNED_MODEL = "ollama:llama3.2"
SUITE_PATH = Path(__file__).resolve().parent.parent / "eval-gate-suite.json"
BASELINE_PATH = Path(__file__).resolve().parent.parent / "eval-baseline.json"
REGRESSION_TOLERANCE = 0.15  # allow this much drop below baseline before failing


async def run_gate() -> float:
    spec = json.loads(SUITE_PATH.read_text(encoding="utf-8"))

    settings = Settings(database_url="sqlite+aiosqlite:///:memory:")
    engine = make_engine(settings)
    await init_db(engine)
    factory = make_session_factory(engine)

    async with factory() as session:
        suite = Suite(name=spec["name"], description=spec.get("description"))
        session.add(suite)
        for item in spec["prompts"]:
            prompt = Prompt(suite=suite)
            session.add(prompt)
            session.add(
                PromptVersion(
                    prompt=prompt,
                    version_number=1,
                    input_text=item["input_text"],
                    expected_output=item.get("expected_output"),
                )
            )
        await session.commit()

        provider_name, _, model_name = PINNED_MODEL.partition(":")
        candidate = CandidateModel(name=model_name, provider=provider_name)
        session.add(candidate)

        run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=3)
        session.add(run)
        await session.commit()

        config = RunConfig(
            providers={provider_name: get_provider(provider_name, settings)},
            judges=[get_judge("exact_match", settings)],
        )
        await execute_run(session, run, [candidate], config)

        scores = (
            await session.execute(select(JudgeEvaluation.score))
        ).scalars().all()

    await engine.dispose()

    if not scores:
        print("eval-gate: no scores recorded — treating as failure", file=sys.stderr)
        return 0.0
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    score = asyncio.run(run_gate())
    print(f"eval-gate: average exact_match score = {score:.4f}")

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps({"score": score}, indent=2) + "\n", encoding="utf-8")
        print(f"eval-gate: wrote new baseline ({score:.4f}) to {BASELINE_PATH}")
        return

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["score"]
    floor = baseline - REGRESSION_TOLERANCE
    print(f"eval-gate: baseline = {baseline:.4f}, floor = {floor:.4f}")
    if score < floor:
        print(
            f"eval-gate: FAIL — score {score:.4f} is below floor {floor:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("eval-gate: PASS")


if __name__ == "__main__":
    main()
