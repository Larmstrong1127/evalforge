# EvalForge Phase 3b-1: Dashboard Core Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Next.js dashboard that can create a suite, launch a run
against the FastAPI backend, poll it to completion, and show results/costs.

**Architecture:** Next.js 15 App Router + TypeScript. Server components fetch
on initial load; one client component (`RunStatusPoll`) owns a TanStack Query
poll that stops once a run reaches a terminal status. Hand-written fetch
wrappers and types in `lib/` mirror the FastAPI backend's Pydantic schemas —
no codegen.

**Tech Stack:** Next.js 15, TypeScript, Tailwind CSS, TanStack Query, Vitest +
React Testing Library for the one component with real logic.

**Conventions:** Run commands from `platform/dashboard/`. Conventional
commits, no AI attribution. TDD for `RunStatusPoll` (the only piece with
logic to test); other components are presentational and don't need unit
tests per the design doc's section 5.

**Correction to the design doc surfaced during planning:** `SuiteResponse`
(the only way to read a suite back, since there's no `GET /suites/{id}`)
returns `{id, name, description, prompt_count}` — it does NOT include the
actual prompt texts. The design doc's "shows the suite's prompts" wording
assumed data the API doesn't expose. This plan's suite-detail page shows the
suite's name/description/prompt **count** and the launch-run form — not
individual prompt text. You don't need to see prompt text to launch a run
against a suite, and adding a new backend endpoint to expose it is out of
scope for this phase (the already-shipped Phase 3a API is not being modified
here).

---

## File structure (locked in by this plan)

```
platform/dashboard/
  app/
    layout.tsx                # root layout, QueryClientProvider, nav shell
    page.tsx                  # redirects to /suites
    providers.tsx             # client component wrapping QueryClientProvider
    suites/
      page.tsx                 # list suites, create-suite form
      [suiteId]/
        page.tsx                 # suite detail: name/description/count, launch-run form
    runs/
      [runId]/
        page.tsx                 # run status (polls), results table, costs
  lib/
    types.ts                  # TypeScript types mirroring the API's Pydantic schemas
    api.ts                     # typed fetch wrappers for the 6 endpoints used
  components/
    SuiteForm.tsx              # create-suite form (name, description, prompts)
    LaunchRunForm.tsx           # candidate/judge/concurrency inputs
    RunStatusPoll.tsx            # TanStack Query poller + status badge
    ResultsTable.tsx              # per-candidate, per-judge results grid
    CostSummary.tsx                # total cost/tokens, by-candidate breakdown
  vitest.config.ts
  vitest.setup.ts
  __tests__/
    RunStatusPoll.test.tsx
```

---

### Task 1: Scaffold Next.js app, install deps, add types and API client

**Files:**
- Create: `platform/dashboard/` (entire scaffolded app)
- Create: `platform/dashboard/lib/types.ts`
- Create: `platform/dashboard/lib/api.ts`
- Create: `platform/dashboard/vitest.config.ts`
- Create: `platform/dashboard/vitest.setup.ts`
- Modify: `platform/dashboard/package.json` (add `test` script)

- [ ] **Step 1: Scaffold the app non-interactively**

Run from the repo root (`C:\Users\cobra\ClaudeProjects\evalforge`):

```bash
npx create-next-app@latest platform/dashboard --typescript --tailwind --app --no-src-dir --import-alias "@/*" --eslint --use-npm
```

Expected: a working Next.js 15 app at `platform/dashboard/` with the default
starter page. Confirm with `cd platform/dashboard && npm run build` — it
should build with zero errors before you touch anything else.

- [ ] **Step 2: Install TanStack Query and test tooling**

Run from `platform/dashboard/`:

```bash
npm install @tanstack/react-query
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom @vitejs/plugin-react
```

- [ ] **Step 3: Add `vitest.config.ts`**

```typescript
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
```

- [ ] **Step 4: Add `vitest.setup.ts`**

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 5: Add the `test` script to `package.json`**

Open `platform/dashboard/package.json` and add to the `"scripts"` object
(alongside the existing `dev`/`build`/`start`/`lint` scripts):

```json
    "test": "vitest run"
```

- [ ] **Step 6: Create `.env.local` for the API base URL**

Create `platform/dashboard/.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

This file is gitignored by `create-next-app`'s default `.gitignore` — do not
add it to git. It will not be committed in Step 9.

- [ ] **Step 7: Write `lib/types.ts`**

```typescript
export interface SuiteResponse {
  id: string;
  name: string;
  description: string | null;
  prompt_count: number;
}

export interface PromptCreate {
  input_text: string;
  expected_output: string | null;
}

export interface SuiteCreate {
  name: string;
  description: string | null;
  prompts: PromptCreate[];
}

export interface RunCreate {
  suite_id: string;
  candidates: string[];
  judges: string[];
  concurrency: number;
}

export interface RunAccepted {
  run_id: string;
}

export type RunStatus = "queued" | "running" | "completed" | "failed";

export interface RunStatusResponse {
  id: string;
  status: RunStatus;
  completed_steps: number;
  total_steps: number;
  started_at: string | null;
  finished_at: string | null;
}

export interface JudgeEvaluationResponse {
  judge_name: string;
  score: number;
  justification: string | null;
}

export interface ResultResponse {
  id: string;
  prompt_version_id: string;
  candidate_model: string;
  status: string;
  generated_text: string;
  error: string | null;
  latency_ms: number;
  cost_usd: number;
  judge_evaluations: JudgeEvaluationResponse[];
}

export interface CostResponse {
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  by_candidate: Record<string, number>;
}
```

- [ ] **Step 8: Write `lib/api.ts`**

```typescript
import type {
  CostResponse,
  ResultResponse,
  RunAccepted,
  RunCreate,
  RunStatusResponse,
  SuiteCreate,
  SuiteResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, body.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export function listSuites(): Promise<SuiteResponse[]> {
  return request<SuiteResponse[]>("/api/v1/suites");
}

export function createSuite(body: SuiteCreate): Promise<SuiteResponse> {
  return request<SuiteResponse>("/api/v1/suites", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function createRun(body: RunCreate): Promise<RunAccepted> {
  return request<RunAccepted>("/api/v1/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getRunStatus(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/api/v1/runs/${runId}`);
}

export function getRunResults(runId: string): Promise<ResultResponse[]> {
  return request<ResultResponse[]>(`/api/v1/runs/${runId}/results`);
}

export function getRunCosts(runId: string): Promise<CostResponse> {
  return request<CostResponse>(`/api/v1/runs/${runId}/costs`);
}

export { ApiError };
```

- [ ] **Step 9: Commit**

```bash
cd platform/dashboard
git add -A
git commit -m "chore: scaffold Next.js dashboard, add types and API client"
```

(`create-next-app`'s generated `.gitignore` already excludes `node_modules/`,
`.next/`, and `.env.local` — confirm with `git status` before committing that
none of those are staged.)

---

### Task 2: Suites list + create-suite page

**Files:**
- Create: `platform/dashboard/app/providers.tsx`
- Modify: `platform/dashboard/app/layout.tsx`
- Modify: `platform/dashboard/app/page.tsx`
- Create: `platform/dashboard/app/suites/page.tsx`
- Create: `platform/dashboard/components/SuiteForm.tsx`

- [ ] **Step 1: Write `app/providers.tsx`** (client component wrapping TanStack Query)

```typescript
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient());
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 2: Wire `Providers` into `app/layout.tsx`**

Open the generated `app/layout.tsx`. It will look roughly like this after
`create-next-app`:

```typescript
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

Replace it with:

```typescript
import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "EvalForge Dashboard",
  description: "LLM evaluation dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <nav className="border-b border-gray-200 px-6 py-4">
            <a href="/suites" className="font-semibold text-lg">
              EvalForge
            </a>
          </nav>
          <main className="p-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Replace `app/page.tsx` with a redirect to `/suites`**

```typescript
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/suites");
}
```

- [ ] **Step 4: Write `components/SuiteForm.tsx`**

```typescript
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createSuite } from "@/lib/api";
import type { PromptCreate } from "@/lib/types";

export function SuiteForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [promptsText, setPromptsText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    const prompts: PromptCreate[] = promptsText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => ({ input_text: line, expected_output: null }));
    try {
      const suite = await createSuite({
        name,
        description: description.trim() || null,
        prompts,
      });
      router.push(`/suites/${suite.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create suite");
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-lg">
      <div>
        <label className="block text-sm font-medium">Name</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Description</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Prompts (one per line)</label>
        <textarea
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 h-32"
          value={promptsText}
          onChange={(e) => setPromptsText(e.target.value)}
        />
      </div>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {submitting ? "Creating..." : "Create suite"}
      </button>
    </form>
  );
}
```

- [ ] **Step 5: Write `app/suites/page.tsx`**

```typescript
import Link from "next/link";
import { listSuites } from "@/lib/api";
import { SuiteForm } from "@/components/SuiteForm";

export default async function SuitesPage() {
  const suites = await listSuites();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-4">Suites</h1>
        {suites.length === 0 ? (
          <p className="text-gray-500">No suites yet.</p>
        ) : (
          <ul className="space-y-2">
            {suites.map((suite) => (
              <li key={suite.id}>
                <Link href={`/suites/${suite.id}`} className="text-blue-600 hover:underline">
                  {suite.name}
                </Link>
                <span className="text-gray-500 text-sm ml-2">
                  ({suite.prompt_count} prompt{suite.prompt_count === 1 ? "" : "s"})
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <h2 className="text-lg font-semibold mb-4">Create a suite</h2>
        <SuiteForm />
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Verify by running the dev server**

With the FastAPI backend NOT necessarily running yet, `npm run build` must
still succeed (build-time type checking doesn't require a live API). Run:

```bash
npm run build
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 7: Commit**

```bash
git add app/providers.tsx app/layout.tsx app/page.tsx app/suites/page.tsx components/SuiteForm.tsx
git commit -m "feat: add suites list and create-suite page"
```

---

### Task 3: Suite detail page with launch-run form

**Files:**
- Create: `platform/dashboard/app/suites/[suiteId]/page.tsx`
- Create: `platform/dashboard/components/LaunchRunForm.tsx`

- [ ] **Step 1: Write `components/LaunchRunForm.tsx`**

```typescript
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createRun } from "@/lib/api";

export function LaunchRunForm({ suiteId }: { suiteId: string }) {
  const router = useRouter();
  const [candidates, setCandidates] = useState("ollama:llama3.2");
  const [judges, setJudges] = useState("exact_match");
  const [concurrency, setConcurrency] = useState(3);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { run_id } = await createRun({
        suite_id: suiteId,
        candidates: candidates
          .split(",")
          .map((c) => c.trim())
          .filter((c) => c.length > 0),
        judges: judges
          .split(",")
          .map((j) => j.trim())
          .filter((j) => j.length > 0),
        concurrency,
      });
      router.push(`/runs/${run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch run");
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-lg">
      <div>
        <label className="block text-sm font-medium">Candidates (comma-separated, provider:model)</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={candidates}
          onChange={(e) => setCandidates(e.target.value)}
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Judges (comma-separated)</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={judges}
          onChange={(e) => setJudges(e.target.value)}
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Concurrency</label>
        <input
          type="number"
          min={1}
          max={20}
          className="mt-1 block w-24 rounded border border-gray-300 px-3 py-2"
          value={concurrency}
          onChange={(e) => setConcurrency(Number(e.target.value))}
        />
      </div>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {submitting ? "Launching..." : "Launch run"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Write `app/suites/[suiteId]/page.tsx`**

Per this plan's header note: `SuiteResponse` has no per-suite fetch and no
prompt text, only `prompt_count`. This page fetches the full suite list and
finds the matching id — a direct page load/refresh must work, so this cannot
rely on client-side state passed from the create form.

```typescript
import { notFound } from "next/navigation";
import { listSuites } from "@/lib/api";
import { LaunchRunForm } from "@/components/LaunchRunForm";

export default async function SuiteDetailPage({
  params,
}: {
  params: Promise<{ suiteId: string }>;
}) {
  const { suiteId } = await params;
  const suites = await listSuites();
  const suite = suites.find((s) => s.id === suiteId);

  if (!suite) {
    notFound();
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
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

```bash
npm run build
```

Expected: succeeds with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add app/suites/[suiteId]/page.tsx components/LaunchRunForm.tsx
git commit -m "feat: add suite detail page with launch-run form"
```

---

### Task 4: Run status polling (TDD) + results/costs page

**Files:**
- Create: `platform/dashboard/components/RunStatusPoll.tsx`
- Create: `platform/dashboard/__tests__/RunStatusPoll.test.tsx`
- Create: `platform/dashboard/components/ResultsTable.tsx`
- Create: `platform/dashboard/components/CostSummary.tsx`
- Create: `platform/dashboard/app/runs/[runId]/page.tsx`

- [ ] **Step 1: Write the failing test for `RunStatusPoll`**

```typescript
// __tests__/RunStatusPoll.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { RunStatusPoll } from "@/components/RunStatusPoll";
import * as api from "@/lib/api";

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("RunStatusPoll", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the queued badge and keeps polling while non-terminal", async () => {
    const spy = vi.spyOn(api, "getRunStatus").mockResolvedValue({
      id: "run-1",
      status: "queued",
      completed_steps: 0,
      total_steps: 3,
      started_at: null,
      finished_at: null,
    });

    renderWithClient(<RunStatusPoll runId="run-1" onTerminal={() => {}} />);

    await waitFor(() => expect(screen.getByText(/queued/i)).toBeInTheDocument());
    expect(spy).toHaveBeenCalledWith("run-1");
  });

  it("calls onTerminal and stops polling once status is completed", async () => {
    const spy = vi.spyOn(api, "getRunStatus").mockResolvedValue({
      id: "run-1",
      status: "completed",
      completed_steps: 3,
      total_steps: 3,
      started_at: "2026-07-04T00:00:00Z",
      finished_at: "2026-07-04T00:00:05Z",
    });
    const onTerminal = vi.fn();

    renderWithClient(<RunStatusPoll runId="run-1" onTerminal={onTerminal} />);

    await waitFor(() => expect(screen.getByText(/completed/i)).toBeInTheDocument());
    await waitFor(() => expect(onTerminal).toHaveBeenCalledTimes(1));
  });

  it("renders the failed badge for a failed run", async () => {
    vi.spyOn(api, "getRunStatus").mockResolvedValue({
      id: "run-1",
      status: "failed",
      completed_steps: 1,
      total_steps: 3,
      started_at: "2026-07-04T00:00:00Z",
      finished_at: "2026-07-04T00:00:05Z",
    });

    renderWithClient(<RunStatusPoll runId="run-1" onTerminal={() => {}} />);

    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
npm run test
```

Expected: FAIL — `Cannot find module '@/components/RunStatusPoll'` (the
component doesn't exist yet).

- [ ] **Step 3: Write `components/RunStatusPoll.tsx`**

```typescript
"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getRunStatus } from "@/lib/api";
import type { RunStatus } from "@/lib/types";

const TERMINAL_STATUSES: RunStatus[] = ["completed", "failed"];

const BADGE_STYLES: Record<RunStatus, string> = {
  queued: "bg-gray-100 text-gray-800",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

export function RunStatusPoll({
  runId,
  onTerminal,
}: {
  runId: string;
  onTerminal: () => void;
}) {
  const { data } = useQuery({
    queryKey: ["run-status", runId],
    queryFn: () => getRunStatus(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && TERMINAL_STATUSES.includes(status) ? false : 2000;
    },
  });

  useEffect(() => {
    if (data && TERMINAL_STATUSES.includes(data.status)) {
      onTerminal();
    }
  }, [data, onTerminal]);

  if (!data) {
    return <p className="text-gray-500">Loading run status...</p>;
  }

  return (
    <div className="flex items-center gap-3">
      <span className={`rounded-full px-3 py-1 text-sm font-medium ${BADGE_STYLES[data.status]}`}>
        {data.status}
      </span>
      <span className="text-gray-600 text-sm">
        {data.completed_steps} / {data.total_steps} steps
      </span>
    </div>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
npm run test
```

Expected: 3 passed.

- [ ] **Step 5: Write `components/ResultsTable.tsx`**

```typescript
import type { ResultResponse } from "@/lib/types";

export function ResultsTable({ results }: { results: ResultResponse[] }) {
  if (results.length === 0) {
    return <p className="text-gray-500">No results yet.</p>;
  }

  return (
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="border-b border-gray-200 text-left">
          <th className="py-2 pr-4">Candidate</th>
          <th className="py-2 pr-4">Status</th>
          <th className="py-2 pr-4">Output</th>
          <th className="py-2 pr-4">Judges</th>
          <th className="py-2 pr-4">Latency</th>
          <th className="py-2 pr-4">Cost</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r) => (
          <tr key={r.id} className="border-b border-gray-100 align-top">
            <td className="py-2 pr-4">{r.candidate_model}</td>
            <td className="py-2 pr-4">{r.status}</td>
            <td className="py-2 pr-4 max-w-xs truncate" title={r.generated_text}>
              {r.error ?? r.generated_text}
            </td>
            <td className="py-2 pr-4">
              {r.judge_evaluations.map((e) => (
                <div key={e.judge_name}>
                  {e.judge_name}: {e.score.toFixed(2)}
                </div>
              ))}
            </td>
            <td className="py-2 pr-4">{r.latency_ms}ms</td>
            <td className="py-2 pr-4">${r.cost_usd.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 6: Write `components/CostSummary.tsx`**

```typescript
import type { CostResponse } from "@/lib/types";

export function CostSummary({ costs }: { costs: CostResponse }) {
  return (
    <div className="space-y-2">
      <p>
        Total: <strong>${costs.total_cost_usd.toFixed(4)}</strong> ·{" "}
        {costs.total_tokens_in} in / {costs.total_tokens_out} out tokens
      </p>
      <ul className="text-sm text-gray-600">
        {Object.entries(costs.by_candidate).map(([candidate, cost]) => (
          <li key={candidate}>
            {candidate}: ${cost.toFixed(4)}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 7: Write `app/runs/[runId]/page.tsx`**

```typescript
"use client";

import { use, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getRunCosts, getRunResults } from "@/lib/api";
import { RunStatusPoll } from "@/components/RunStatusPoll";
import { ResultsTable } from "@/components/ResultsTable";
import { CostSummary } from "@/components/CostSummary";

export default function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);
  const [isTerminal, setIsTerminal] = useState(false);

  const resultsQuery = useQuery({
    queryKey: ["run-results", runId],
    queryFn: () => getRunResults(runId),
    enabled: isTerminal,
  });
  const costsQuery = useQuery({
    queryKey: ["run-costs", runId],
    queryFn: () => getRunCosts(runId),
    enabled: isTerminal,
  });

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-semibold">Run {runId}</h1>
      <RunStatusPoll runId={runId} onTerminal={() => setIsTerminal(true)} />
      {isTerminal && (
        <>
          <div>
            <h2 className="text-lg font-semibold mb-4">Results</h2>
            {resultsQuery.data ? (
              <ResultsTable results={resultsQuery.data} />
            ) : (
              <p className="text-gray-500">Loading results...</p>
            )}
          </div>
          <div>
            <h2 className="text-lg font-semibold mb-4">Costs</h2>
            {costsQuery.data ? (
              <CostSummary costs={costsQuery.data} />
            ) : (
              <p className="text-gray-500">Loading costs...</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
```

`use(params)` unwraps the Promise-based `params` in a client component (this
page needs `useState`/`useQuery`, so it can't be a server component like the
suites pages — Next.js 15's `params` is a Promise in both server and client
components, and `use()` is the client-side way to read it).

- [ ] **Step 8: Verify build and full test suite**

```bash
npm run build
npm run test
```

Expected: build succeeds, all tests pass (3 total, all from
`RunStatusPoll.test.tsx`).

- [ ] **Step 9: Commit**

```bash
git add components/RunStatusPoll.tsx __tests__/RunStatusPoll.test.tsx components/ResultsTable.tsx components/CostSummary.tsx app/runs/[runId]/page.tsx
git commit -m "feat: add run status polling, results table, and cost summary"
```

---

### Task 5: Real end-to-end smoke test — actually run both servers

**Files:** none created; this is a manual verification task.

- [ ] **Step 1: Start the FastAPI backend**

From `platform/api/`:

```bash
.venv\Scripts\uvicorn evalforge.main:app --port 8000
```

Expected: `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **Step 2: Start the dashboard dev server**

From `platform/dashboard/` (separate terminal):

```bash
npm run dev
```

Expected: `Local: http://localhost:3000`.

- [ ] **Step 3: Walk through the full flow in a browser**

1. Open `http://localhost:3000` — should redirect to `/suites`.
2. Create a suite named `smoke-test` with one prompt line: `What is 2+2? Answer with the number only.`
3. Confirm redirect to the new suite's detail page, showing `1 prompt`.
4. Launch a run with candidates `ollama:llama3.2` and judges `exact_match`
   (requires `ollama list` to show `llama3.2` locally — if it's not
   installed, substitute any locally available Ollama model and adjust the
   prompt's `expected_output` expectation accordingly when reading results).
5. Confirm redirect to `/runs/<id>`, showing a `queued` or `running` badge
   that updates every ~2s.
6. Wait for the badge to reach `completed` — confirm the Results table and
   Cost summary render with real data (a generated answer, a judge score,
   nonzero token counts if the provider reports them).

This is the step most likely to catch a real integration bug (CORS
misconfiguration, a type mismatch between hand-written TS types and the
actual JSON shape) that component tests alone would miss — consistent with
this project's established lesson that mocked/isolated tests miss real-
environment surprises (see Phase 3a's own E2E smoke test, which caught a
genuine BackgroundTasks/session-commit ordering bug that no mocked test
exercised).

- [ ] **Step 4: Stop both servers**

Ctrl+C in each terminal.

## Definition of done (Phase 3b-1)

- `npm run build` and `npm run test` both succeed in `platform/dashboard/`.
- Real end-to-end smoke test confirms create suite → launch run → poll to
  completion → view results/costs works against the real FastAPI backend
  and a real local Ollama model.
- Three pages live: `/suites`, `/suites/[suiteId]`, `/runs/[runId]`.

## Explicitly deferred (do not build in this plan)

- The A/B rating room and compare view (Phase 3b-2, separate spec/plan).
- SSE for live run progress (v1.1 per the parent spec's ADR-006).
- A `GET /suites/{id}` backend endpoint (not needed — see this plan's header
  note; revisit only if a future phase genuinely needs individual prompt
  text on the suite detail page).
- Pagination, auth, dark mode / visual polish pass.
