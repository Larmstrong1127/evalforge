# ADR-003: Explicit commit before scheduling a BackgroundTask

## Status
Accepted (2026-07)

## Context
`POST /runs` creates the `Run` row in the request-scoped session, returns
`202 Accepted`, and hands execution to a FastAPI `BackgroundTasks` wrapper
that deliberately opens its own database session (per the Phase 3a design:
the request session is torn down as part of the request/response lifecycle,
so sharing it with a background task is a known anti-pattern). The original
implementation called `session.flush()` and relied on the `get_session`
dependency's post-yield `commit()` to make the row durable.

## Decision
`create_run` calls `session.commit()` explicitly before
`background_tasks.add_task(...)`. The dependency's post-yield commit remains
as the general-purpose path for every other endpoint.

## Rationale
Starlette executes `BackgroundTasks` inside the same ASGI middleware layer
that later unwinds the dependency's `AsyncExitStack` — meaning the background
task starts running *before* the dependency's post-yield commit fires. A
flushed-but-uncommitted row is visible only inside the request session's own
transaction; the background task's freshly opened session saw nothing and
crashed on `assert run is not None`.

## How it was found
Not by the test suite. Every mocked test intentionally shares one in-memory
session factory between the request and the simulated background task (a
requirement of SQLite `:memory:` semantics), which masks the ordering bug by
construction. It surfaced on the first real end-to-end run against a live
server and local Ollama. A regression test now exists that uses two genuinely
separate engines against the same file-backed SQLite database, reproducing
the production topology (`test_create_run_row_is_visible_to_background_task_with_separate_engines`).

## Revisit when
Run execution moves out of the API process (see ADR-001's revisit note) — a
real queue makes the enqueue itself the durability boundary.
