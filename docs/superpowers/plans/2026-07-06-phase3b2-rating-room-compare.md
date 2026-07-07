# EvalForge Phase 3b-2: Rating Room + Compare View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the A/B blind-voting rating room and the run-vs-run compare view, the
two remaining pieces of the dashboard's originally-scoped feature set.

**Architecture:** One small additive backend endpoint (`GET /suites/{id}/runs`) to make
the rating room reachable; everything else is new Next.js pages/components consuming
already-shipped `POST /ratings` and `GET /compare` endpoints. Pairing logic (grouping
results, generating unique candidate pairs, seeded shuffle) lives in a pure, unit-tested
TypeScript module separate from the React component that renders it.

**Tech Stack:** FastAPI (backend endpoint), Next.js 16 + TypeScript + TanStack Query
(frontend, same stack as Phase 3b-1), Vitest for the pairing-logic unit tests.

**Conventions:** Backend commands run from `platform/api/`, frontend from
`platform/dashboard/`. Conventional commits, no AI attribution. TDD for the backend
endpoint and the pairing-logic module (the two pieces with real logic); presentational
components are not unit-tested, consistent with Phase 3b-1's established testing
philosophy. Apply Phase 3b-1's established conventions throughout: `htmlFor`/`id`
label association on every form input, an `if (submitting) return;` reentrancy guard on
every submit handler, try/catch + inline error message for server-component data
fetches, `isError` handling for `useQuery` calls, `aria-live`/`role="status"` on any
live-updating region.

---

## File structure (locked in by this plan)

```
platform/api/evalforge/
  api/runs.py                          # MODIFY: add GET /suites/{suite_id}/runs
  tests/test_api_suites.py             # MODIFY: add tests for the new endpoint
                                         # (lives here, not test_api_runs.py, since
                                         # it's a suite-scoped query even though the
                                         # route function lives in runs.py's router)

platform/dashboard/
  lib/
    types.ts                            # MODIFY: add RatingCreate, RatingResponse,
                                          #  CompareRow, CompareResponse
    api.ts                              # MODIFY: add createRating, getCompare,
                                          #  listSuiteRuns
    pairing.ts                          # NEW: pure pairing/shuffle logic
  __tests__/
    pairing.test.ts                     # NEW: unit tests for lib/pairing.ts
  app/
    suites/[suiteId]/
      page.tsx                          # MODIFY: list completed runs, link to rate
    suites/[suiteId]/rate/
      page.tsx                          # NEW: rating room
    compare/
      page.tsx                          # NEW: compare view
    layout.tsx                          # MODIFY: add "Compare" nav link
  components/
    RatingCard.tsx                      # NEW: one blind result card in the rating room
    CompareTable.tsx                    # NEW: the diff table
```

---

### Task 1: Backend — `GET /suites/{suite_id}/runs`

**Files:**
- Modify: `platform/api/evalforge/api/runs.py`
- Modify: `platform/api/tests/test_api_suites.py`

- [ ] **Step 1: Write the failing tests**

Append to `platform/api/tests/test_api_suites.py`:

```python
async def test_list_suite_runs_returns_runs_for_that_suite(api_client, session):
    from evalforge.db.models import Run, RunStatus, Suite

    suite_a = Suite(name="suite-a")
    suite_b = Suite(name="suite-b")
    run_a1 = Run(suite=suite_a, status=RunStatus.COMPLETED, concurrency_limit=1)
    run_a2 = Run(suite=suite_a, status=RunStatus.QUEUED, concurrency_limit=1)
    run_b1 = Run(suite=suite_b, status=RunStatus.COMPLETED, concurrency_limit=1)
    session.add_all([suite_a, suite_b, run_a1, run_a2, run_b1])
    await session.commit()

    response = await api_client.get(f"/api/v1/suites/{suite_a.id}/runs")
    assert response.status_code == 200
    body = response.json()
    returned_ids = {r["id"] for r in body}
    assert returned_ids == {str(run_a1.id), str(run_a2.id)}
    assert str(run_b1.id) not in returned_ids


async def test_list_suite_runs_returns_empty_list_for_suite_with_no_runs(api_client, session):
    from evalforge.db.models import Suite

    suite = Suite(name="empty-suite")
    session.add(suite)
    await session.commit()

    response = await api_client.get(f"/api/v1/suites/{suite.id}/runs")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_suite_runs_returns_404_for_missing_suite(api_client):
    response = await api_client.get(
        "/api/v1/suites/00000000-0000-0000-0000-000000000000/runs"
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run (from `platform/api/`): `.venv\Scripts\python -m pytest tests/test_api_suites.py -v`
Expected: the 3 new tests FAIL with 404/405 (route doesn't exist yet); all other
existing tests in the file still PASS.

- [ ] **Step 3: Add the route to `evalforge/api/runs.py`**

No new imports are needed: `Suite`, `Run`, `RunStatusResponse`, `parse_uuid_or_404`,
`HTTPException`, `Depends`, `AsyncSession`, `select` are all already imported in this
file (used by `create_run` and the other existing routes) — read the file's current
import block to confirm before writing, and only add anything genuinely missing. Append
this route to `evalforge/api/runs.py`, after the existing `get_run_costs` function:

```python
@router.get("/suites/{suite_id}/runs", response_model=list[RunStatusResponse])
async def list_suite_runs(
    suite_id: str, session: AsyncSession = Depends(get_session)  # noqa: B008
) -> list[RunStatusResponse]:
    suite_uuid = parse_uuid_or_404(suite_id, "suite")
    suite = await session.get(Suite, suite_uuid)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id} not found")
    runs = (
        await session.execute(select(Run).where(Run.suite_id == suite_uuid))
    ).scalars().all()
    return [
        RunStatusResponse(
            id=r.id,
            status=r.status.value,
            completed_steps=r.completed_steps,
            total_steps=r.total_steps,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in runs
    ]
```

This route lives on the `runs` router (not the `suites` router) even though its path is
under `/suites/{suite_id}/...`, matching how `evalforge/main.py` already mounts both
routers under the same `/api/v1` prefix with no path-based separation enforced — check
`evalforge/main.py` to confirm this is consistent with how the existing
`POST /suites/{suite_id}/prompts` route lives on the `suites` router instead. Since this
new route's query logic is entirely about `Run` (not `Suite`) and needs `Run`/`RunStatus`
already imported in `runs.py`, placing it here avoids a duplicate import of `Run` in
`suites.py`. `Suite`, `RunStatusResponse`, `parse_uuid_or_404`, `HTTPException`,
`Depends`, `AsyncSession`, `select` are all already imported in `runs.py` — verify this
by reading the file's current import block before writing, and only add what's missing.

- [ ] **Step 4: Run to verify pass**

Run: `.venv\Scripts\python -m pytest tests/test_api_suites.py -v`
Expected: all tests in the file PASS (previous 5 + new 3 = 8).

- [ ] **Step 5: Run the full suite, lint, typecheck**

```bash
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\ruff check evalforge tests
.venv\Scripts\mypy evalforge
```
Expected: all pass (69 + 3 = 72 total), ruff/mypy clean.

- [ ] **Step 6: Commit**

```bash
git add platform/api/evalforge/api/runs.py platform/api/tests/test_api_suites.py
git commit -m "feat: add GET /suites/{suite_id}/runs endpoint"
```

---

### Task 2: Frontend — types, API client, pairing logic (TDD)

**Files:**
- Modify: `platform/dashboard/lib/types.ts`
- Modify: `platform/dashboard/lib/api.ts`
- Create: `platform/dashboard/lib/pairing.ts`
- Create: `platform/dashboard/__tests__/pairing.test.ts`

- [ ] **Step 1: Add types to `lib/types.ts`**

Append to the existing file (after `CostResponse`):

```typescript
export interface RatingCreate {
  prompt_version_id: string;
  result_a_id: string;
  result_b_id: string;
  chosen_result_id: string | null;
  skipped: boolean;
  rater_session: string | null;
}

export interface RatingResponse {
  id: string;
}

export interface CompareRow {
  prompt_version_id: string;
  candidate_model: string;
  run_a_result: ResultResponse | null;
  run_b_result: ResultResponse | null;
  score_delta: Record<string, number>;
}

export interface CompareResponse {
  run_a: RunStatusResponse;
  run_b: RunStatusResponse;
  rows: CompareRow[];
}
```

- [ ] **Step 2: Add API functions to `lib/api.ts`**

Append to the existing file (before the final `export { ApiError };` line — move that
export to stay last, or just add these functions above it):

```typescript
export function createRating(body: RatingCreate): Promise<RatingResponse> {
  return request<RatingResponse>("/api/v1/ratings", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getCompare(runIdA: string, runIdB: string): Promise<CompareResponse> {
  const params = new URLSearchParams();
  params.append("run_id", runIdA);
  params.append("run_id", runIdB);
  return request<CompareResponse>(`/api/v1/compare?${params.toString()}`);
}

export function listSuiteRuns(suiteId: string): Promise<RunStatusResponse[]> {
  return request<RunStatusResponse[]>(`/api/v1/suites/${suiteId}/runs`);
}
```

Add `RatingCreate`, `RatingResponse`, `CompareResponse` to the existing `import type { ... } from "./types";` block at the top of the file (merge with what's already imported —
`RunStatusResponse` is likely already imported; don't duplicate).

- [ ] **Step 3: Write the failing test for pairing logic**

Create `platform/dashboard/__tests__/pairing.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { buildRatingPairs } from "@/lib/pairing";
import type { ResultResponse } from "@/lib/types";

function makeResult(id: string, promptVersionId: string, candidate: string): ResultResponse {
  return {
    id,
    prompt_version_id: promptVersionId,
    candidate_model: candidate,
    status: "ok",
    generated_text: `answer from ${candidate}`,
    error: null,
    latency_ms: 10,
    cost_usd: 0,
    judge_evaluations: [],
  };
}

describe("buildRatingPairs", () => {
  it("pairs results within the same prompt_version and never across prompt_versions", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-2", "a"),
      makeResult("r4", "pv-2", "b"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    expect(pairs).toHaveLength(2);
    for (const pair of pairs) {
      expect(pair.a.prompt_version_id).toBe(pair.b.prompt_version_id);
    }
  });

  it("generates each unique candidate pair exactly once, never a result with itself", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    // C(3,2) = 3 unique pairs for one prompt_version with 3 candidates
    expect(pairs).toHaveLength(3);
    const seenKeys = new Set<string>();
    for (const pair of pairs) {
      expect(pair.a.id).not.toBe(pair.b.id);
      const key = [pair.a.id, pair.b.id].sort().join(":");
      expect(seenKeys.has(key)).toBe(false);
      seenKeys.add(key);
    }
  });

  it("skips prompt_versions with fewer than 2 results", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-2", "a"),
      makeResult("r3", "pv-2", "b"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    expect(pairs).toHaveLength(1);
    expect(pairs[0].a.prompt_version_id).toBe("pv-2");
  });

  it("is deterministic for the same seed", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
      makeResult("r4", "pv-1", "d"),
    ];
    const first = buildRatingPairs(results, "same-seed");
    const second = buildRatingPairs(results, "same-seed");
    expect(first.map((p) => `${p.a.id}:${p.b.id}`)).toEqual(
      second.map((p) => `${p.a.id}:${p.b.id}`)
    );
  });

  it("produces a different order for a different seed (probabilistically, with a fixed fixture)", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
      makeResult("r4", "pv-1", "d"),
      makeResult("r5", "pv-1", "e"),
    ];
    const orderA = buildRatingPairs(results, "seed-a").map((p) => `${p.a.id}:${p.b.id}`);
    const orderB = buildRatingPairs(results, "seed-b").map((p) => `${p.a.id}:${p.b.id}`);
    expect(orderA).not.toEqual(orderB);
  });
});
```

- [ ] **Step 4: Run to verify failure**

Run: `npm run test`
Expected: FAIL — `Cannot find module '@/lib/pairing'`.

- [ ] **Step 5: Write `lib/pairing.ts`**

```typescript
import type { ResultResponse } from "./types";

export interface RatingPair {
  a: ResultResponse;
  b: ResultResponse;
}

/**
 * Deterministic string hash → 32-bit int, used to seed the PRNG. Not
 * cryptographic — this only needs to produce a stable, well-distributed
 * seed from a runId string, not resist adversarial input.
 */
function hashSeed(seed: string): number {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (Math.imul(31, hash) + seed.charCodeAt(i)) | 0;
  }
  return hash >>> 0;
}

/** Mulberry32 PRNG — small, seedable, sufficient for shuffle ordering (not
 * used for anything security-sensitive). */
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(items: T[], rand: () => number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Groups results by prompt_version_id, generates every unique unordered pair
 * of results within each group (skipping groups with fewer than 2 results),
 * then shuffles both the pair order and each pair's internal (a, b) side
 * assignment using a PRNG seeded from `seed` — stable across repeated calls
 * with the same seed (e.g. reloading the rating room mid-session), different
 * across different seeds (e.g. different runs).
 */
export function buildRatingPairs(results: ResultResponse[], seed: string): RatingPair[] {
  const byPromptVersion = new Map<string, ResultResponse[]>();
  for (const result of results) {
    const group = byPromptVersion.get(result.prompt_version_id) ?? [];
    group.push(result);
    byPromptVersion.set(result.prompt_version_id, group);
  }

  const rand = mulberry32(hashSeed(seed));
  const pairs: RatingPair[] = [];
  // Map iteration order for a JS Map is insertion order, which depends on
  // the input array's order — sort group keys for determinism independent
  // of the caller's array ordering.
  const sortedPromptVersionIds = [...byPromptVersion.keys()].sort();
  for (const promptVersionId of sortedPromptVersionIds) {
    const group = byPromptVersion.get(promptVersionId)!;
    if (group.length < 2) continue;
    const sortedGroup = [...group].sort((x, y) => x.id.localeCompare(y.id));
    for (let i = 0; i < sortedGroup.length; i++) {
      for (let j = i + 1; j < sortedGroup.length; j++) {
        const [a, b] = rand() < 0.5 ? [sortedGroup[i], sortedGroup[j]] : [sortedGroup[j], sortedGroup[i]];
        pairs.push({ a, b });
      }
    }
  }

  return seededShuffle(pairs, rand);
}
```

- [ ] **Step 6: Run to verify pass**

Run: `npm run test`
Expected: all 5 new tests PASS, plus the existing 4 `RunStatusPoll` tests still PASS (9
total).

- [ ] **Step 7: Lint and typecheck**

```bash
npm run lint
npm run build
```
Expected: both clean (build succeeds even though no page uses these new exports yet —
they're just unused-but-exported functions, which is fine for a library module, not a
lint violation the way an unused local variable would be).

- [ ] **Step 8: Commit**

```bash
git add lib/types.ts lib/api.ts lib/pairing.ts __tests__/pairing.test.ts
git commit -m "feat: add rating/compare types, API client functions, and pairing logic"
```

---

### Task 3: Suite detail page — list completed runs, link to rating room

**Files:**
- Modify: `platform/dashboard/app/suites/[suiteId]/page.tsx`

- [ ] **Step 1: Read the current file**

Read `platform/dashboard/app/suites/[suiteId]/page.tsx` in full — it currently fetches
`listSuites()` and finds the matching suite, then renders `LaunchRunForm`. This task
adds a second fetch (`listSuiteRuns`) and a new section listing completed runs with
links into the rating room.

- [ ] **Step 2: Modify the page**

Replace the full file with:

```typescript
import Link from "next/link";
import { notFound } from "next/navigation";
import { listSuiteRuns, listSuites } from "@/lib/api";
import { LaunchRunForm } from "@/components/LaunchRunForm";

export const dynamic = "force-dynamic";

export default async function SuiteDetailPage({
  params,
}: {
  params: Promise<{ suiteId: string }>;
}) {
  const { suiteId } = await params;

  let suites: Awaited<ReturnType<typeof listSuites>> = [];
  let loadError: string | null = null;
  try {
    suites = await listSuites();
  } catch (err) {
    loadError = err instanceof Error ? err.message : "Failed to load suite";
  }

  if (loadError) {
    return <p className="text-red-600 text-sm">{loadError}</p>;
  }

  const suite = suites.find((s) => s.id === suiteId);
  if (!suite) {
    notFound();
  }

  let completedRuns: Awaited<ReturnType<typeof listSuiteRuns>> = [];
  let runsLoadError: string | null = null;
  try {
    const runs = await listSuiteRuns(suiteId);
    completedRuns = runs.filter((r) => r.status === "completed");
  } catch (err) {
    runsLoadError = err instanceof Error ? err.message : "Failed to load runs";
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold">{suite.name}</h1>
        {suite.description && <p className="text-gray-600 mt-1">{suite.description}</p>}
        <p className="text-gray-500 text-sm mt-1">
          {suite.prompt_count} prompt{suite.prompt_count === 1 ? "" : "s"}
        </p>
      </div>
      <div>
        <h2 className="text-lg font-semibold mb-4">Launch a run</h2>
        <LaunchRunForm suiteId={suite.id} />
      </div>
      <div>
        <h2 className="text-lg font-semibold mb-4">Rate completed runs</h2>
        {runsLoadError ? (
          <p className="text-red-600 text-sm">{runsLoadError}</p>
        ) : completedRuns.length === 0 ? (
          <p className="text-gray-500">No completed runs yet.</p>
        ) : (
          <ul className="space-y-2">
            {completedRuns.map((run) => (
              <li key={run.id}>
                <Link
                  href={`/suites/${suiteId}/rate?runId=${run.id}`}
                  className="text-blue-600 hover:underline"
                >
                  Rate run {run.id.slice(0, 8)}
                </Link>
                <span className="text-gray-500 text-sm ml-2">
                  {run.completed_steps} / {run.total_steps} steps
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

Note: the two data fetches (`listSuites()` and `listSuiteRuns(suiteId)`) are kept as
separate try/catch blocks rather than one combined block — a failure fetching runs
shouldn't hide the suite's own name/description/launch-form section, which is still
fully renderable even if the runs list fails. This mirrors the existing page's
established pattern of failing narrowly rather than broadly.

- [ ] **Step 3: Verify build**

```bash
npm run build
```
Expected: succeeds, `/suites/[suiteId]` still marked dynamic (`ƒ`).

- [ ] **Step 4: Commit**

```bash
git add app/suites/[suiteId]/page.tsx
git commit -m "feat: list completed runs on suite detail page, link to rating room"
```

---

### Task 4: Rating room page + RatingCard component (TDD covered by Task 2's pairing tests; this task is integration, not new logic)

**Files:**
- Create: `platform/dashboard/components/RatingCard.tsx`
- Create: `platform/dashboard/app/suites/[suiteId]/rate/page.tsx`

- [ ] **Step 1: Write `components/RatingCard.tsx`**

```typescript
import type { ResultResponse } from "@/lib/types";

export function RatingCard({
  result,
  revealed,
  onChoose,
}: {
  result: ResultResponse;
  revealed: boolean;
  onChoose: () => void;
}) {
  return (
    <div className="flex-1 rounded border border-gray-300 p-4 space-y-3">
      <p className="whitespace-pre-wrap text-sm">
        {result.error ?? result.generated_text}
      </p>
      {revealed ? (
        <p className="text-xs text-gray-500 font-mono">{result.candidate_model}</p>
      ) : (
        <button
          type="button"
          onClick={onChoose}
          className="rounded bg-blue-600 px-4 py-2 text-white text-sm"
        >
          Choose this
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Write `app/suites/[suiteId]/rate/page.tsx`**

```typescript
"use client";

import { use, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { createRating, getRunResults } from "@/lib/api";
import { buildRatingPairs, type RatingPair } from "@/lib/pairing";
import { RatingCard } from "@/components/RatingCard";

export default function RatingRoomPage({
  params,
}: {
  params: Promise<{ suiteId: string }>;
}) {
  const { suiteId } = use(params);
  const searchParams = useSearchParams();
  const runId = searchParams.get("runId");

  const raterSessionRef = useRef<string>(crypto.randomUUID());
  const [pairs, setPairs] = useState<RatingPair[] | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [voteCount, setVoteCount] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const resultsQuery = useQuery({
    queryKey: ["run-results-for-rating", runId],
    queryFn: () => getRunResults(runId!),
    enabled: !!runId,
  });

  useEffect(() => {
    if (resultsQuery.data && pairs === null) {
      setPairs(buildRatingPairs(resultsQuery.data, runId ?? ""));
    }
  }, [resultsQuery.data, pairs, runId]);

  const currentPair = useMemo(
    () => (pairs && currentIndex < pairs.length ? pairs[currentIndex] : null),
    [pairs, currentIndex]
  );

  async function submitVote(chosenResultId: string | null, skipped: boolean) {
    if (submitting || !currentPair) return;
    setSubmitting(true);
    try {
      await createRating({
        prompt_version_id: currentPair.a.prompt_version_id,
        result_a_id: currentPair.a.id,
        result_b_id: currentPair.b.id,
        chosen_result_id: chosenResultId,
        skipped,
        rater_session: raterSessionRef.current,
      });
      setVoteCount((count) => count + 1);
      setRevealed(true);
    } finally {
      setSubmitting(false);
    }
  }

  function advance() {
    setRevealed(false);
    setCurrentIndex((i) => i + 1);
  }

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (revealed || submitting || !currentPair) return;
      if (e.key === "ArrowLeft") {
        submitVote(currentPair.a.id, false);
      } else if (e.key === "ArrowRight") {
        submitVote(currentPair.b.id, false);
      } else if (e.key === "s" || e.key === "S") {
        submitVote(null, true);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  if (!runId) {
    return <p className="text-red-600 text-sm">Missing runId query parameter.</p>;
  }

  if (resultsQuery.isError) {
    return (
      <p className="text-red-600 text-sm">
        {resultsQuery.error instanceof Error ? resultsQuery.error.message : "Failed to load results"}
      </p>
    );
  }

  if (!pairs) {
    return <p className="text-gray-500">Loading results to rate...</p>;
  }

  if (!currentPair) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Done rating</h1>
        <p>You cast {voteCount} vote{voteCount === 1 ? "" : "s"}.</p>
        <Link href={`/suites/${suiteId}`} className="text-blue-600 hover:underline">
          Back to suite
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4" role="region" aria-label="Rating room">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Rate this pair</h1>
        <p className="text-gray-500 text-sm" role="status" aria-live="polite">
          {currentIndex + 1} / {pairs.length} · {voteCount} vote{voteCount === 1 ? "" : "s"} cast
        </p>
      </div>
      <div className="flex gap-4">
        <RatingCard
          result={currentPair.a}
          revealed={revealed}
          onChoose={() => submitVote(currentPair.a.id, false)}
        />
        <RatingCard
          result={currentPair.b}
          revealed={revealed}
          onChoose={() => submitVote(currentPair.b.id, false)}
        />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => submitVote(null, true)}
          disabled={submitting || revealed}
          className="rounded border border-gray-300 px-4 py-2 text-sm disabled:opacity-50"
        >
          Skip (S)
        </button>
        {revealed && (
          <button
            type="button"
            onClick={advance}
            className="rounded bg-blue-600 px-4 py-2 text-white text-sm"
          >
            Next pair
          </button>
        )}
        <p className="text-gray-500 text-xs">
          Keyboard: ← left, → right, S to skip
        </p>
      </div>
    </div>
  );
}
```

Note on the `useEffect` for keyboard handling: it deliberately has no dependency array
(runs after every render) rather than `[revealed, submitting, currentPair]`, because
`submitVote`'s closure over `currentPair`/`submitting` would otherwise go stale between
renders if the dependency array were incomplete — removing and re-adding the listener
on every render is a small, acceptable cost here (this page has no expensive render
work) and guarantees the handler always sees current state. The cleanup function
correctly removes the exact same listener reference each time.

- [ ] **Step 3: Verify build**

```bash
npm run build
```
Expected: succeeds. `/suites/[suiteId]/rate` should be a dynamic route (client
component with a dynamic segment — same reasoning as `/runs/[runId]` from Phase 3b-1,
no explicit `force-dynamic` needed).

- [ ] **Step 4: Run the full test suite**

```bash
npm run test
```
Expected: all 9 tests still pass (nothing in this task touches tested logic directly,
but confirms no regression).

- [ ] **Step 5: Commit**

```bash
git add components/RatingCard.tsx app/suites/[suiteId]/rate/page.tsx
git commit -m "feat: add rating room page with keyboard-driven blind voting"
```

---

### Task 5: Compare view page + CompareTable component

**Files:**
- Create: `platform/dashboard/components/CompareTable.tsx`
- Create: `platform/dashboard/app/compare/page.tsx`
- Modify: `platform/dashboard/app/layout.tsx`

- [ ] **Step 1: Write `components/CompareTable.tsx`**

```typescript
import type { CompareResponse } from "@/lib/types";

function ScoreDelta({ judgeName, delta }: { judgeName: string; delta: number }) {
  const color = delta > 0 ? "text-green-600" : delta < 0 ? "text-red-600" : "text-gray-500";
  const sign = delta > 0 ? "+" : "";
  return (
    <div className={color}>
      {judgeName}: {sign}
      {delta.toFixed(2)}
    </div>
  );
}

export function CompareTable({ compare }: { compare: CompareResponse }) {
  if (compare.rows.length === 0) {
    return <p className="text-gray-500">No overlapping results between these two runs.</p>;
  }

  return (
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="border-b border-gray-200 text-left">
          <th className="py-2 pr-4">Prompt</th>
          <th className="py-2 pr-4">Candidate</th>
          <th className="py-2 pr-4">Run A output</th>
          <th className="py-2 pr-4">Run B output</th>
          <th className="py-2 pr-4">Score delta</th>
        </tr>
      </thead>
      <tbody>
        {compare.rows.map((row) => (
          <tr
            key={`${row.prompt_version_id}:${row.candidate_model}`}
            className="border-b border-gray-100 align-top"
          >
            <td className="py-2 pr-4 font-mono text-xs">
              {row.prompt_version_id.slice(0, 8)}
            </td>
            <td className="py-2 pr-4">{row.candidate_model}</td>
            <td className="py-2 pr-4 max-w-xs truncate" title={row.run_a_result?.generated_text}>
              {row.run_a_result ? row.run_a_result.generated_text : "— not present in this run —"}
            </td>
            <td className="py-2 pr-4 max-w-xs truncate" title={row.run_b_result?.generated_text}>
              {row.run_b_result ? row.run_b_result.generated_text : "— not present in this run —"}
            </td>
            <td className="py-2 pr-4">
              {Object.entries(row.score_delta).map(([judgeName, delta]) => (
                <ScoreDelta key={judgeName} judgeName={judgeName} delta={delta} />
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Write `app/compare/page.tsx`**

```typescript
"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCompare, listSuiteRuns, listSuites } from "@/lib/api";
import { CompareTable } from "@/components/CompareTable";
import type { RunStatusResponse } from "@/lib/types";

interface RunOption {
  runId: string;
  suiteName: string;
  status: string;
}

export default function ComparePage() {
  const [runIdA, setRunIdA] = useState("");
  const [runIdB, setRunIdB] = useState("");

  const optionsQuery = useQuery({
    queryKey: ["compare-run-options"],
    queryFn: async (): Promise<RunOption[]> => {
      const suites = await listSuites();
      const perSuite = await Promise.all(
        suites.map(async (suite) => {
          const runs: RunStatusResponse[] = await listSuiteRuns(suite.id);
          return runs
            .filter((run) => run.status === "completed")
            .map((run) => ({ runId: run.id, suiteName: suite.name, status: run.status }));
        })
      );
      return perSuite.flat();
    },
  });

  const compareQuery = useQuery({
    queryKey: ["compare", runIdA, runIdB],
    queryFn: () => getCompare(runIdA, runIdB),
    // Same-run comparison (runIdA === runIdB) is deliberately allowed, not
    // blocked — it's the degenerate-but-valid case this phase's E2E smoke
    // test uses to verify the diff table without needing two real runs.
    enabled: !!runIdA && !!runIdB,
  });

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-semibold">Compare runs</h1>
      {optionsQuery.isError ? (
        <p className="text-red-600 text-sm">
          {optionsQuery.error instanceof Error ? optionsQuery.error.message : "Failed to load runs"}
        </p>
      ) : (
        <div className="flex gap-4 items-end max-w-2xl">
          <div className="flex-1">
            <label htmlFor="compare-run-a" className="block text-sm font-medium">
              Run A
            </label>
            <select
              id="compare-run-a"
              className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
              value={runIdA}
              onChange={(e) => setRunIdA(e.target.value)}
            >
              <option value="">Select a run</option>
              {optionsQuery.data?.map((opt) => (
                <option key={opt.runId} value={opt.runId}>
                  {opt.suiteName} — {opt.runId.slice(0, 8)}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label htmlFor="compare-run-b" className="block text-sm font-medium">
              Run B
            </label>
            <select
              id="compare-run-b"
              className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
              value={runIdB}
              onChange={(e) => setRunIdB(e.target.value)}
            >
              <option value="">Select a run</option>
              {optionsQuery.data?.map((opt) => (
                <option key={opt.runId} value={opt.runId}>
                  {opt.suiteName} — {opt.runId.slice(0, 8)}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
      {runIdA && runIdB && runIdA === runIdB && (
        <p className="text-gray-500 text-sm">Comparing a run against itself — every delta should be 0.</p>
      )}
      {compareQuery.isError ? (
        <p className="text-red-600 text-sm">
          {compareQuery.error instanceof Error ? compareQuery.error.message : "Failed to load comparison"}
        </p>
      ) : compareQuery.data ? (
        <CompareTable compare={compareQuery.data} />
      ) : runIdA && runIdB ? (
        <p className="text-gray-500">Loading comparison...</p>
      ) : null}
    </div>
  );
}
```

`getCompare` accepts a same-run comparison (`run_id=a&run_id=a`) without special-casing
it — the backend endpoint doesn't reject it, and it's this phase's E2E smoke-test plan
for verifying the diff table without needing two separately-run suites.

- [ ] **Step 3: Add a "Compare" link to the nav**

Read `platform/dashboard/app/layout.tsx`. Modify the `<nav>` block to add a second link
next to the existing "EvalForge" home link:

```typescript
          <nav className="border-b border-gray-200 px-6 py-4 flex items-center gap-6">
            <Link href="/suites" className="font-semibold text-lg">
              EvalForge
            </Link>
            <Link href="/compare" className="text-sm text-gray-600 hover:text-gray-900">
              Compare
            </Link>
          </nav>
```
(This assumes the existing nav currently only has the `<Link href="/suites">` element
from Phase 3b-1 — read the actual current file first and adapt if its structure
differs, but the two `<Link>` elements shown above must both be present afterward.)

- [ ] **Step 4: Verify build**

```bash
npm run build
```
Expected: succeeds, `/compare` appears as a route in the output (dynamic, client
component).

- [ ] **Step 5: Run lint and the full test suite**

```bash
npm run lint
npm run test
```
Expected: lint clean, all 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add components/CompareTable.tsx app/compare/page.tsx app/layout.tsx
git commit -m "feat: add compare view with run selection and diff table"
```

---

### Task 6: Real end-to-end smoke test

**Files:** none created; this is a manual verification task.

- [ ] **Step 1: Start both servers**

From `platform/api/`: `.venv\Scripts\uvicorn evalforge.main:app --port 8000`
From `platform/dashboard/` (separate terminal): `npm run dev`

- [ ] **Step 2: Set up a run with multiple candidates**

In the browser at `http://localhost:3000/suites`:
1. Create a suite `rating-smoke-test` with 2 prompts (e.g. "What is 2+2? Answer with the
   number only." and "What is the capital of France? One word.").
2. On the suite detail page, launch a run with candidates
   `ollama:llama3.2,ollama:qwen2.5:14b` (or any two locally available Ollama models —
   check `ollama list` first and substitute) and judges `exact_match`.
3. Wait for the run to reach `completed` on its `/runs/{id}` page (2 prompts × 2
   candidates = 4 results, so this may take under a minute on local hardware).

- [ ] **Step 3: Rating room walkthrough**

1. Navigate back to the suite detail page — confirm the completed run now appears under
   "Rate completed runs" with a link.
2. Click into the rating room. Confirm the first pair renders with both outputs visible
   but candidate names hidden.
3. Click "Choose this" on one card — confirm the candidate name reveals on both cards,
   the vote count increments, and clicking "Next pair" advances.
4. On the second pair, use the `→` arrow key instead of clicking — confirm the same
   reveal/advance behavior.
5. On the third pair (if there are enough unique candidate pairs — 2 prompts × 1 pair
   per prompt for 2 candidates = 2 total pairs, so add a third candidate to the run if
   you want to test the skip path on a genuinely separate pair, or just test skip on
   whichever pair remains), press `S` to skip — confirm it reveals without a
   `chosen_result_id`, then advances.
6. After the last pair, confirm the "Done rating" screen shows the correct total vote
   count and links back to the suite.

- [ ] **Step 4: Compare view walkthrough**

1. Navigate to `/compare` via the nav link.
2. Select the same completed run for both "Run A" and "Run B" (the degenerate
   same-run case from the design doc).
3. Confirm the diff table renders one row per (prompt, candidate) pair, both output
   columns showing identical text, and every score delta reading `0.00` (or blank if
   `exact_match` didn't score every result — check the actual rendered deltas match
   what you'd expect from comparing identical data).

- [ ] **Step 5: Confirm votes were actually persisted**

```bash
curl http://localhost:8000/api/v1/runs/<run_id>/results
```
(There's no `GET /ratings` list endpoint — this is an accepted v1 gap, not something
this phase adds. Instead, confirm persistence indirectly: re-open the rating room for
the same run in a NEW browser tab. Since `buildRatingPairs`'s pair order is
deterministic per `runId` but each tab tracks "already seen" pairs only in its own
component state — not fetched from the server — the new tab will show the SAME first
pair again, not skip it. This is expected per the design's explicit choice that
"already rated" tracking is ephemeral/client-side, not server-persisted-and-fetched-back
— confirm this matches your expectation reading the design doc's section 2.2, rather
than treating it as a bug.)

- [ ] **Step 6: Stop both servers**

Ctrl+C in each terminal.

## Definition of done (Phase 3b-2)

- Backend: `GET /suites/{suite_id}/runs` implemented, tested (72 total backend tests),
  ruff/mypy clean.
- Frontend: `npm run build`, `npm run test` (9 tests), `npm run lint` all succeed.
- Real E2E smoke test confirms: rating room reachable from a suite's completed runs,
  blind voting works via both click and keyboard, votes persist (confirmed indirectly),
  compare view renders a same-run diff with all-zero deltas.

## Explicitly deferred (do not build in this plan)

- A `GET /ratings` list/export endpoint (would be needed to consume the collected
  preference-pair dataset for reward-model training — that's Phase 2's own deferred
  "post-ship" item in the parent spec, not this phase's concern).
- A `GET /prompt-versions/{id}` endpoint for showing real prompt text in the compare
  table (truncated-id fallback is accepted per the design doc).
- Multi-rater coordination, SSE, pagination — all explicitly deferred in the design doc.
