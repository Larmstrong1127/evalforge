"""Idempotent demo seeding for the zero-API-key Docker Compose demo.

Creates one suite with a completed run so every dashboard view (suites list,
run results/costs, rating room, compare) has real data to show without any
provider keys. Rows are written directly via the ORM — no providers are
called. Enabled by `EVALFORGE_SEED_DEMO=1` (checked in main.py's lifespan),
which is a demo hook rather than app config, so it deliberately does not
live in Settings.
"""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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

DEMO_SUITE_NAME = "demo-qa"

# (input_text, expected_output, model_a_answer, model_b_answer)
_DEMO_PROMPTS = [
    ("What is 2+2? Answer with the number only.", "4", "4", "4"),
    ("What is the capital of France? One word.", "Paris", "Paris", "Lyon"),
    ("What color is a stop sign? One word.", "Red", "Red", "Red"),
]


async def seed_demo(session: AsyncSession) -> None:
    existing = (
        await session.execute(select(Suite).where(Suite.name == DEMO_SUITE_NAME))
    ).scalar_one_or_none()
    if existing is not None:
        return

    suite = Suite(
        name=DEMO_SUITE_NAME,
        description="Seeded demo suite — browse results, cast A/B votes, compare runs.",
    )
    session.add(suite)

    model_a = CandidateModel(name="model-a", provider="demo")
    model_b = CandidateModel(name="model-b", provider="demo")
    session.add_all([model_a, model_b])

    started = datetime.now(UTC) - timedelta(minutes=5)
    run = Run(
        suite=suite,
        status=RunStatus.COMPLETED,
        concurrency_limit=3,
        completed_steps=len(_DEMO_PROMPTS) * 2,
        total_steps=len(_DEMO_PROMPTS) * 2,
        started_at=started,
        finished_at=started + timedelta(seconds=42),
    )
    session.add(run)

    latency = 300
    for input_text, expected, answer_a, answer_b in _DEMO_PROMPTS:
        prompt = Prompt(suite=suite)
        version = PromptVersion(
            prompt=prompt,
            version_number=1,
            input_text=input_text,
            expected_output=expected,
        )
        session.add_all([prompt, version])
        for candidate, answer in ((model_a, answer_a), (model_b, answer_b)):
            result = Result(
                run=run,
                prompt_version=version,
                candidate_model=candidate,
                status=ResultStatus.OK,
                generated_text=answer,
                latency_ms=latency,
                input_tokens=24,
                output_tokens=3,
                cost_usd=0.0,
            )
            session.add(result)
            session.add(
                JudgeEvaluation(
                    result=result,
                    judge_name="exact_match",
                    score=1.0 if answer == expected else 0.0,
                )
            )
            latency += 137  # varied but deterministic

    await session.commit()
