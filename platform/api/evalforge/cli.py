"""CLI entry point: create suites, run evals, inspect results."""
import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import typer
from sqlalchemy import select

from evalforge.config import Settings
from evalforge.db.engine import init_db, make_engine, make_session_factory
from evalforge.db.models import (
    CandidateModel,
    JudgeEvaluation,
    Prompt,
    PromptVersion,
    Result,
    Run,
    RunStatus,
    Suite,
)
from evalforge.judges import get_judge
from evalforge.providers import get_provider
from evalforge.runner import RunConfig, execute_run

app = typer.Typer(help="EvalForge: multi-provider LLM evaluation engine")
suite_app = typer.Typer(help="Manage eval suites")
app.add_typer(suite_app, name="suite")


def _run(coro: Coroutine[Any, Any, None]) -> None:  # bridge typer's sync world to asyncio
    asyncio.run(coro)


@suite_app.command("create")
def suite_create(path: Path) -> None:
    """Create a suite from a JSON file."""
    if not path.exists():
        typer.echo(f"error: {path} not found", err=True)
        raise typer.Exit(code=1)
    spec = json.loads(path.read_text(encoding="utf-8"))

    async def _create() -> None:
        settings = Settings()
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
            count = len(spec["prompts"])
            typer.echo(f"created suite '{suite.name}' ({suite.id}) with {count} prompts")
        await engine.dispose()

    _run(_create())


@suite_app.command("list")
def suite_list() -> None:
    """List all suites."""

    async def _list() -> None:
        settings = Settings()
        engine = make_engine(settings)
        await init_db(engine)
        factory = make_session_factory(engine)
        async with factory() as session:
            suites = (await session.execute(select(Suite))).scalars().all()
            for s in suites:
                typer.echo(f"{s.id}  {s.name}")
        await engine.dispose()

    _run(_list())


@app.command("run")
def run_eval(
    suite_name: str,
    candidate: list[str] = typer.Option(  # noqa: B008
        ..., "--candidate", "-c", help="provider:model, e.g. ollama:llama3.2"
    ),
    judge: list[str] = typer.Option(  # noqa: B008
        ["exact_match"], "--judge", "-j", help="judge name, e.g. exact_match, llm_judge"
    ),
    concurrency: int = typer.Option(3, "--concurrency"),  # noqa: B008
) -> None:
    """Run a suite against candidate models."""

    async def _execute() -> None:
        settings = Settings()
        engine = make_engine(settings)
        await init_db(engine)
        factory = make_session_factory(engine)
        async with factory() as session:
            suite = (
                await session.execute(select(Suite).where(Suite.name == suite_name))
            ).scalar_one_or_none()
            if suite is None:
                typer.echo(f"error: suite '{suite_name}' not found", err=True)
                raise typer.Exit(code=1)

            candidates: list[CandidateModel] = []
            providers = {}
            for spec in candidate:
                provider_name, _, model_name = spec.partition(":")
                if not model_name:
                    typer.echo(f"error: candidate '{spec}' must be provider:model", err=True)
                    raise typer.Exit(code=1)
                providers[provider_name] = get_provider(provider_name, settings)
                candidates.append(CandidateModel(name=model_name, provider=provider_name))
            session.add_all(candidates)

            run = Run(suite=suite, status=RunStatus.QUEUED, concurrency_limit=concurrency)
            session.add(run)
            await session.commit()

            config = RunConfig(
                providers=providers,
                judges=[get_judge(name, settings) for name in judge],
            )
            typer.echo(f"run {run.id} started ({len(candidate)} candidates)")
            await execute_run(session, run, candidates, config)
            typer.echo(
                f"run {run.id} {run.status.value}: {run.completed_steps}/{run.total_steps} steps"
            )
        await engine.dispose()

    _run(_execute())


@app.command("results")
def show_results(run_id: str) -> None:
    """Show scored results for a run."""

    async def _show() -> None:
        settings = Settings()
        engine = make_engine(settings)
        factory = make_session_factory(engine)
        async with factory() as session:
            results = (
                await session.execute(select(Result).where(Result.run_id == run_id))
            ).scalars().all()
            for r in results:
                evals = (
                    await session.execute(
                        select(JudgeEvaluation).where(JudgeEvaluation.result_id == r.id)
                    )
                ).scalars().all()
                scores = ", ".join(f"{e.judge_name}={e.score:.2f}" for e in evals)
                status = r.status.value
                typer.echo(f"[{status}] {r.generated_text[:60]!r}  {scores}  ${r.cost_usd:.4f}")
        await engine.dispose()

    _run(_show())


if __name__ == "__main__":
    app()
