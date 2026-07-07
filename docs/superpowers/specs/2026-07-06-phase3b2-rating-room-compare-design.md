# EvalForge Phase 3b-2 — A/B Rating Room + Compare View Design

Date: 2026-07-06
Status: Approved
Parent spec: `docs/superpowers/specs/2026-07-02-evalforge-design.md` (sections 2, 5)
Depends on: Phase 3a FastAPI backend (`POST /api/v1/ratings`, `GET /api/v1/compare`) and
Phase 3b-1 dashboard core flow (both merged to master)

## 1. Scope

The two remaining pieces of the dashboard's originally-scoped feature set (deferred from
3b-1 because they're independent subsystems built on data 3b-1 already fetches):

1. **A/B rating room** — blind pairwise voting between two candidates' outputs for the
   same prompt, LMSYS Chatbot Arena style. Persists votes via the existing
   `POST /api/v1/ratings` endpoint (Phase 2's preference-pair dataset).
2. **Compare view** — run-vs-run regression diff via the existing
   `GET /api/v1/compare` endpoint.

Both voting/compare endpoints and their schemas already exist, tested, and merged — no
changes needed there. One small, additive backend endpoint IS needed to make the rating
room reachable at all (there's no way to list a suite's runs today); see section 2.4 for
why this surfaced during design rather than being assumed away. Otherwise this phase is
entirely new pages/components in `platform/dashboard/`.

## 2. Rating room

### 2.1 Entry point and run selection

New route `/suites/[suiteId]/rate`. Linked from the suite detail page
(`/suites/[suiteId]`) once at least one run against that suite has reached `completed`
status — the suite detail page needs a small addition: list the suite's completed runs
(via a new `listRunsForSuite` — see 2.4 for why this needs a small client-side
workaround) with links into the rating room for each.

The rating room page itself takes `runId` as a query param (`/suites/[suiteId]/rate?runId=...`),
not a route segment — it's scoped to one suite but the specific run is a selection made
from that suite's completed runs list, and a query param keeps the URL shareable/bookmarkable
without adding another dynamic route segment for what's fundamentally a filter.

### 2.2 Pairing logic (client-side, per the parent spec's explicit note that pair
selection is dashboard logic, not a backend concern)

On load, fetch `GET /api/v1/runs/{runId}/results`. Group results by `prompt_version_id`.
For each prompt_version group with 2+ results, generate all unique candidate pairs
`(result_i, result_j)` where `i < j` (avoids showing the same pair twice in reversed
order). Flatten into one ordered list of pairs across all prompt_versions, shuffle once
with a seeded PRNG (seed = `runId`, not `Math.random()`) so left/right positions and
pair order are randomized per run but **stable across reloads/back-navigation within the
same run** — a rater who reloads mid-session doesn't get a completely different shuffle,
which would be confusing if they were tracking progress mentally. Track "already rated
this session" pairs in local component state (a `Set` of
`` `${result_a_id}:${result_b_id}` `` keys, both orderings normalized so a skip-then-see-again
isn't possible within the session) — voted/skipped pairs are removed from the remaining
queue, not persisted as a "don't show again" preference across sessions (ratings ARE
persisted server-side via `POST /ratings`, but which pairs a given browser tab has
already seen this session is ephemeral, matching the LMSYS Arena UX where you can
absolutely encounter the same models on a future session).

Left/right side assignment: randomized per pair (again seeded, not per-render) so a
rater can't learn "candidate A is always shown left." Blind means the candidate model
string (`"provider:model"`) is never rendered until a vote is cast — after voting or
skipping, briefly reveal both, then advance.

### 2.3 Voting interaction

- Two side-by-side result cards showing `generated_text` (blind), each with a "Choose
  this" button.
- Keyboard shortcuts: `←` votes for the left card, `→` votes for the right card, `S`
  skips the pair, matching the parent spec's explicit "arrows to vote, S to skip"
  requirement. Shortcuts are disabled while the post-vote reveal is showing (prevents a
  rater's fast double-press from accidentally voting on the next pair before it's fully
  rendered).
- On vote: `POST /api/v1/ratings` with `prompt_version_id`, `result_a_id`,
  `result_b_id` (using the pair's actual result ids, not "left"/"right" — the schema's
  `chosen_result_id` disambiguates which one won regardless of display side),
  `chosen_result_id` set to whichever result was picked, `skipped: false`.
- On skip: same call shape but `chosen_result_id: null`, `skipped: true`.
- `rater_session`: a random UUID generated once per browser tab (`crypto.randomUUID()`,
  stored in a `useRef` so it's stable for the component's lifetime, not persisted to
  `localStorage` — there's no login system, and a fresh session per tab/reload is an
  accepted simplification consistent with this being a local-first demo tool, not a
  production crowdsourcing platform).
- After the last pair: show a completion message with a count of votes cast and a link
  back to the suite.

### 2.4 A real API gap surfaced during this design — resolved without backend changes

There is no `GET /suites/{id}/runs` endpoint to list a suite's runs, and unlike 3b-1's
`SuiteResponse` gap (where the missing data — prompt text — could be worked around
client-side by fetching the full suite list), there is no client-side workaround here:
`GET /api/v1/runs/{run_id}` requires already knowing a run id, and nothing in the
existing API surface can enumerate the run ids belonging to a given suite.

**Resolution: this phase DOES need one new backend endpoint.** Unlike 3b-1's
`SuiteResponse` gap (which was avoidable because the missing data — prompt text — wasn't
essential to that phase's flow), enumerating a suite's runs is genuinely required for the
rating room to be reachable at all, and there is no client-side workaround for "list
things filtered by a foreign key with no list endpoint." This is a one-line addition
mirroring the existing `GET /suites` pattern:

```
GET /api/v1/suites/{suite_id}/runs   →  list[RunStatusResponse]
```

Implemented as a new route on the existing `evalforge/api/runs.py` router (it already
owns `Run`-related endpoints), filtering `Run` by `suite_id`. This is a small, additive,
backward-compatible change to the already-shipped Phase 3a API — not a redesign — and
goes through the same TDD + spec/quality review discipline as every other backend
endpoint in this project. It's called out explicitly here rather than silently added
during implementation, since discovering a real API gap mid-frontend-phase is exactly
the kind of thing that should surface in the spec, not get papered over.

## 3. Compare view

### 3.1 Entry point

New route `/compare`, linked from the global nav (alongside "EvalForge" in the layout
header — this is the first cross-suite view, since a comparison can span any two
completed runs regardless of suite).

### 3.2 Run selection

Two `<select>` dropdowns ("Run A", "Run B"), populated from `GET /api/v1/suites` →
for each suite, `GET /api/v1/suites/{id}/runs` (the same new endpoint from 2.4) →
flatten into one list of `{run_id, suite_name, status, started_at}` options, filtered to
`status === "completed"` only (comparing an in-progress or failed run against another
produces a diff that's mostly `null` on one side, which is a low-value view no phase-3-b
user story asks for — the spec doesn't disallow it at the API layer, but the UI doesn't
surface incomplete runs as selectable, since the "Definition of done" for this phase is
about the regression-diff use case, not defending against every combination that's
technically not blocked at the boundary). A "Compare" button is disabled until both are
selected and calls `GET /api/v1/compare?run_id={a}&run_id={b}`.

### 3.3 Diff rendering

One row per `CompareRow`: prompt (looked up... **second real gap**, see 3.4),
candidate model, run A's output, run B's output, and a per-judge score delta rendered
with a colored up/down indicator (green for a positive delta meaning run B scored
higher, red for negative — "positive" here is literally `score_delta[judge] > 0`, no
judge-specific inversion logic, since all judges in this codebase already normalize to
the "1.0 = good" convention per `evalforge/judges/__init__.py`'s `Judgment` docstring).
Rows where a result exists on only one side (per `CompareRow.run_a_result`/
`run_b_result` being nullable) render that side as "— not present in this run —" rather
than blank, so a missing result reads as an explicit absence, not a rendering bug.

### 3.4 Second real gap: `CompareRow` has `prompt_version_id`, not the prompt's text

Same class of issue as 3b-1's `SuiteResponse` gap, but here it's a **display-only**
concern (unlike the rating room's "genuinely can't reach this feature" gap) — the
compare table is still fully functional and useful without showing prompt text, just
less readable (rows are identified by a UUID instead of the actual question). Per the
established precedent, this is an acceptable v1 simplification, not a blocker: render
the `prompt_version_id` truncated to its first 8 characters as a fallback label,
with a one-line code comment noting a `GET /prompt-versions/{id}` endpoint would be
the real fix if this becomes a recurring pain point. Unlike 2.4, this does NOT
justify a new backend endpoint in this phase — it's cosmetic, not blocking.

## 4. Shared infrastructure

- `lib/types.ts` gains `RatingCreate`, `RatingResponse`, `CompareRow`, `CompareResponse`
  (mirroring the backend schemas exactly, same pattern as every existing type in this
  file).
- `lib/api.ts` gains `createRating(body)`, `getCompare(runIdA, runIdB)`, and
  `listSuiteRuns(suiteId)` (for the new backend endpoint from 2.4).
- No new shared components beyond what 3b-1 already has (`ResultsTable` and
  `CostSummary` are not reused here — the rating room's blind-card layout and the
  compare view's side-by-side diff table are visually and structurally distinct enough
  to warrant their own components, not awkward variants of existing ones).

## 5. Testing

- **Backend**: the new `GET /suites/{suite_id}/runs` endpoint gets the same TDD +
  spec/quality review treatment as every other Phase 3a endpoint — a new test file
  `tests/test_api_suites.py` addition (or a focused new test in the existing file),
  covering the happy path and an empty-list case for a suite with no runs yet.
- **Frontend — pairing logic**: the pair-generation/shuffle function (grouping by
  prompt_version, generating unique candidate pairs, seeded shuffle) is pure logic
  extracted into its own testable function (`lib/pairing.ts`), unit-tested for
  determinism (same `runId` seed → same pair order across two calls) and correctness
  (no pair appears twice, no result is paired with itself).
- **Frontend — components**: no unit tests for `ResultsTable`-style presentational
  rendering (consistent with 3b-1's established testing philosophy — presentational
  components render props, no regression risk). The rating room's keyboard-shortcut
  wiring and the compare view's dropdown-to-fetch wiring are exercised by the real E2E
  smoke test below, not unit tests, since they're thin glue over already-tested pieces
  (the pairing logic itself IS unit tested; the component wiring it into JSX is not).
- **Real E2E smoke test** (manual, same discipline as every prior phase): start both
  servers, create a suite with 2+ prompts, launch a run with 2+ candidates against real
  Ollama, wait for completion, open the rating room, cast at least 2 votes (one via
  click, one via keyboard), confirm the vote count increments and pairs don't repeat;
  then open `/compare`, select that run against itself (a degenerate but valid
  same-run comparison — every row's score_delta should be exactly 0 since it's diffing
  a run against itself) to confirm the diff table renders without needing a second real
  run.

## 6. Explicitly deferred

- Any reward-model training on the collected ratings (Phase 2's own "post-ship" note in
  the parent spec — this phase only collects the data, doesn't consume it).
- SSE for live rating-room progress across multiple raters (single-rater, single-tab
  usage is the only supported mode; no multi-rater coordination).
- A dedicated `GET /prompt-versions/{id}` endpoint for the compare view's prompt-text
  display gap (3.4) — truncated-id fallback is accepted for this phase.
- Pagination on the new `GET /suites/{suite_id}/runs` endpoint or the compare view's
  run-selection dropdowns — consistent with every other list endpoint in this project's
  documented v1 scope.
