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
            {resultsQuery.isError ? (
              <p className="text-red-600 text-sm">
                {resultsQuery.error instanceof Error ? resultsQuery.error.message : "Failed to load results"}
              </p>
            ) : resultsQuery.data ? (
              <ResultsTable results={resultsQuery.data} />
            ) : (
              <p className="text-gray-500">Loading results...</p>
            )}
          </div>
          <div>
            <h2 className="text-lg font-semibold mb-4">Costs</h2>
            {costsQuery.isError ? (
              <p className="text-red-600 text-sm">
                {costsQuery.error instanceof Error ? costsQuery.error.message : "Failed to load costs"}
              </p>
            ) : costsQuery.data ? (
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
