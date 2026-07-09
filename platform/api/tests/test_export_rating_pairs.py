"""Tests for exporting HumanRating rows as (prompt, chosen, rejected) JSONL."""
import json

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from evalforge.db.engine import init_db
from evalforge.db.models import (
    CandidateModel,
    HumanRating,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)
from evalforge.export_rating_pairs import export_pairs


async def test_export_pairs_orientation_and_skip_filtering(tmp_path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        suite = Suite(name="s")
        prompt = Prompt(suite=suite)
        version = PromptVersion(
            prompt=prompt, version_number=1, input_text="What is 2+2?", expected_output="4"
        )
        model = CandidateModel(name="m", provider="demo")
        run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1,
                  completed_steps=2, total_steps=2)
        res_a = Result(run=run, prompt_version=version, candidate_model=model,
                       status=ResultStatus.OK, generated_text="4", latency_ms=1,
                       input_tokens=1, output_tokens=1, cost_usd=0.0)
        res_b = Result(run=run, prompt_version=version, candidate_model=model,
                       status=ResultStatus.OK, generated_text="5", latency_ms=1,
                       input_tokens=1, output_tokens=1, cost_usd=0.0)
        session.add_all([suite, prompt, version, model, run, res_a, res_b])
        await session.flush()
        # vote where B was shown first but A won: orientation must follow chosen_result_id
        session.add(HumanRating(prompt_version_id=version.id, result_a_id=res_b.id,
                                result_b_id=res_a.id, chosen_result_id=res_a.id))
        # skipped vote: must be excluded
        session.add(HumanRating(prompt_version_id=version.id, result_a_id=res_a.id,
                                result_b_id=res_b.id, chosen_result_id=None, skipped=True))
        await session.commit()

        out = tmp_path / "pairs.jsonl"
        count = await export_pairs(session, out)

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}]
    await engine.dispose()
