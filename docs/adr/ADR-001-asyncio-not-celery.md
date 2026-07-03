# ADR-001: asyncio task execution, not Celery/Redis

## Status
Accepted (2026-07)

## Context
Eval runs are long-running jobs: N prompts x M candidate models, each an HTTP
call to a provider, plus judge scoring. A common reflex is Celery + Redis.

## Decision
Plain asyncio: a semaphore bounds per-run concurrency, run state persists in
the `runs` table, and per-result failures are recorded without failing the run.

## Rationale
The workload is I/O-bound fan-out, not CPU work needing worker processes. A
single API process comfortably drives hundreds of concurrent HTTP calls.
Celery would add a broker, a result backend, worker deployment, and serialization
constraints — none of which buy anything at this scale.

## Consequence observed during implementation
Because a single `AsyncSession` is shared across all concurrent tasks for a
run, isolating failures turned out to require more care than the initial
design assumed: individual judge exceptions, unexpected provider exceptions,
and even a failed `commit()` itself (which poisons the session's transaction
state until an explicit `rollback()`) all had to be defended against
separately to uphold "one bad item never fails the whole run." This is the
tradeoff of a single shared session instead of one connection per task — cheap
now, but the exception-isolation logic in `runner.py` is the direct cost of it.

## Revisit when
Multiple API replicas need to share a work queue, or runs must survive process
restarts mid-flight.
