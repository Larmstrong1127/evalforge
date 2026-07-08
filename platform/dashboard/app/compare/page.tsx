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
        <p className="text-red-600 text-sm" role="alert">
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
        <p className="text-red-600 text-sm" role="alert">
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
