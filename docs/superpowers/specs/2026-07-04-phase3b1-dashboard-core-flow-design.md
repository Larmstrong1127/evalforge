# EvalForge Phase 3b-1 — Dashboard Core Flow Design

Date: 2026-07-04
Status: Approved
Parent spec: `docs/superpowers/specs/2026-07-02-evalforge-design.md` (section 2, "Next.js dashboard")
Depends on: Phase 3a FastAPI backend (merged to master, `platform/api/`)

## 1. Scope

The Next.js dashboard's full feature set (parent spec) is three fairly
independent subsystems: core run-management flow, the A/B rating room, and
the run-vs-run compare view. This phase builds only the first — a working
end-to-end flow of create a suite, launch a run against it, watch it
complete, and see results and costs. The A/B rating room and compare view
are explicitly out of scope here (Phase 3b-2, a separate spec/plan, built
after this one is merged — both are read/write layers on top of data this
phase already fetches and renders, so they're a natural follow-on rather
than a parallel track).

## 2. Tech stack

- **Next.js 15, App Router, TypeScript.** Matches the parent spec and the
  "Full Stack SWE" role-mapping rationale already recorded there.
- **TanStack Query** for the one genuinely stateful piece of this phase: a
  run's status poll. Per the parent spec's ADR-006, V1 dashboards poll;
  SSE is a deliberate v1.1 upgrade, not built here.
- **Tailwind CSS** for styling. No component library dependency yet —
  YAGNI until there's enough shared UI to justify one.
- **No API client generator** (no `openapi-typescript-codegen` or similar).
  The API surface is small (6 endpoints touched by this phase) and still
  young; hand-written fetch wrappers with hand-written TypeScript types are
  cheaper to read and change than debugging a codegen pipeline at this
  scale. Revisit if/when Phase 3b-2 roughly doubles the endpoint count.
- **New package**: `platform/dashboard/` (sibling to `platform/api/`),
  scaffolded via `create-next-app` with TypeScript + Tailwind + App Router,
  ESLint's default Next.js config.

## 3. Pages and routes

```
platform/dashboard/
  app/
    layout.tsx              # root layout, nav shell
    page.tsx                # redirects to /suites
    suites/
      page.tsx               # list suites, create-suite form
      [suiteId]/
        page.tsx               # suite detail: prompts, launch-run form
    runs/
      [runId]/
        page.tsx               # run status (polls), results table, costs
  lib/
    api.ts                   # typed fetch wrappers for the 6 endpoints used
    types.ts                  # TypeScript types mirroring the API's Pydantic schemas
  components/
    SuiteForm.tsx             # create-suite form (name, description, prompts)
    LaunchRunForm.tsx         # candidate/judge/concurrency inputs
    RunStatusPoll.tsx         # TanStack Query poller + status badge
    ResultsTable.tsx          # per-candidate, per-judge results grid
    CostSummary.tsx           # total cost/tokens, by-candidate breakdown
  __tests__/
    RunStatusPoll.test.tsx    # the one component with real behavior to test
```

**`/suites`** — `GET /api/v1/suites` on load (server component); a form
posting to `POST /api/v1/suites` (client component, redirects to the new
suite's detail page on success).

**`/suites/[suiteId]`** — shows the suite's prompts and a form to launch a
run (`POST /api/v1/runs`), redirecting to `/runs/[runId]` on the `202`
response. There's no `GET /suites/{id}` endpoint yet in the API (see
section 6), so this page fetches the full `GET /suites` list and finds
the matching id — this must work on a direct page load/refresh, not only
right after creation, so client-side state carried over from the create
form is not an option here.

**`/runs/[runId]`** — polls `GET /api/v1/runs/{id}` every 2s via TanStack
Query (`refetchInterval`) until `status` is `completed` or `failed`; once
terminal, fetches `GET /api/v1/runs/{id}/results` and
`GET /api/v1/runs/{id}/costs` once (no further polling needed — a
terminal run's results don't change).

## 4. Data flow and error handling

- Server components handle the initial suites list fetch (no client-side
  loading spinner needed for that path).
- `RunStatusPoll` is the only client component with real state: it owns
  the TanStack Query hook, renders a status badge (queued/running/
  completed/failed), and stops polling (`refetchInterval: false`) once
  the run reaches a terminal state.
- Network/API errors surface as a simple inline error banner per page —
  no global error boundary infrastructure yet (YAGNI at this scale; one
  dashboard, one API, no multi-tenant error taxonomy to build for).
- Form validation is minimal: rely on the API's existing 400/422
  responses and surface their `detail` message directly rather than
  duplicating validation logic client-side (the backend is the source of
  truth for "what's a valid candidate spec," etc.).

## 5. Testing

- **`RunStatusPoll.test.tsx`**: the one component with actual logic worth
  unit-testing — verify it polls while status is non-terminal, stops
  polling once terminal, and renders the right badge per status. Uses
  React Testing Library + a mocked fetch (msw or a simple `vi.fn()` stub,
  whichever the scaffolded test setup defaults to).
- **Real E2E smoke test** (manual, mirroring Phase 3a's Task 7 discipline):
  start both `uvicorn` and `next dev`, walk through create suite → launch
  run against local Ollama → watch it complete → see results/costs
  rendered, once at the end of the implementation plan. This is the step
  most likely to catch a real integration bug (CORS misconfiguration,
  a type mismatch between the hand-written TS types and the actual JSON
  shape, etc.) that component tests alone would miss — consistent with
  this project's established lesson that mocked/isolated tests miss
  real-environment surprises.
- No broad component-test coverage for purely presentational components
  (`ResultsTable`, `CostSummary`, form components) — they render props,
  there's no logic to regress.

## 6. Explicitly deferred / known gaps

- **A/B rating room and compare view** — Phase 3b-2, separate spec.
- **SSE for live run progress** — v1.1 per the parent spec's ADR-006.
- **No `GET /suites/{id}` endpoint exists yet.** The suite-detail page
  works around this by fetching the full `GET /suites` list and finding
  the matching suite client-side (acceptable at demo scale — suite counts
  are small; this mirrors the same reasoning already applied to
  `list_suites`'s prompt-count aggregation). If Phase 3b-2 or later work
  needs suite detail at scale, add the dedicated endpoint then rather than
  now — YAGNI.
- **No pagination anywhere** — consistent with the backend's own
  documented v1 limitation (parent API spec, section 9).
- **No auth** — consistent with the backend having none (local-first tool).
- **No dark mode / polish pass** — this phase is functional correctness;
  visual polish is a natural fit for the `impeccable` skill as a follow-up
  once the core flow works, not bundled into this implementation.
