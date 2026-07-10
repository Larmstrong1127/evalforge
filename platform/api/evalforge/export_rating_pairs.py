"""Export rating-room preference pairs as JSONL for the training package.

Each non-skipped HumanRating becomes one line:
    {"prompt": <input_text>, "chosen": <winning text>, "rejected": <losing text>}
oriented by chosen_result_id (NOT by the a/b display order, which is
shuffled per rater). Skipped votes (chosen_result_id IS NULL) are excluded —
a skip is "no preference", not a preference.

Run with the platform venv (needs the evalforge package):
    cd platform/api
    .venv/Scripts/python.exe -m evalforge.export_rating_pairs \
        --out ../../training/data/rating_pairs.jsonl
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.config import Settings
from evalforge.db.engine import make_engine, make_session_factory
from evalforge.db.models import HumanRating, PromptVersion, Result


async def export_pairs(session: AsyncSession, out_path: Path) -> int:
    ratings = (
        (
            await session.execute(
                select(HumanRating).where(HumanRating.chosen_result_id.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    # batch-fetch referenced rows (avoid per-rating N+1, same pattern as runs.py)
    result_ids = {r.result_a_id for r in ratings} | {r.result_b_id for r in ratings}
    results = {
        res.id: res
        for res in (await session.execute(select(Result).where(Result.id.in_(result_ids))))
        .scalars()
        .all()
    }
    version_ids = {r.prompt_version_id for r in ratings}
    versions = {
        v.id: v
        for v in (
            await session.execute(select(PromptVersion).where(PromptVersion.id.in_(version_ids)))
        )
        .scalars()
        .all()
    }

    lines: list[str] = []
    for rating in ratings:
        chosen_id = rating.chosen_result_id
        if chosen_id is None:
            continue  # filtered by the query above, but narrows type for mypy
        rejected_id = rating.result_b_id if chosen_id == rating.result_a_id else rating.result_a_id
        chosen = results.get(chosen_id)
        rejected = results.get(rejected_id)
        version = versions.get(rating.prompt_version_id)
        if chosen is None or rejected is None or version is None:
            continue  # dangling reference; skip rather than crash the export
        if chosen.generated_text is None or rejected.generated_text is None:
            continue
        lines.append(
            json.dumps(
                {
                    "prompt": version.input_text,
                    "chosen": chosen.generated_text,
                    "rejected": rejected.generated_text,
                },
                ensure_ascii=False,
            )
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


async def _main(out: Path) -> None:
    settings = Settings()
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    async with factory() as session:
        count = await export_pairs(session, out)
    print(f"exported {count} preference pairs -> {out}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(_main(parser.parse_args().out))
