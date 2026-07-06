"use client";

import { useEffect, useRef } from "react";
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
  const { data, isError, error } = useQuery({
    queryKey: ["run-status", runId],
    queryFn: () => getRunStatus(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && TERMINAL_STATUSES.includes(status) ? false : 2000;
    },
  });

  // Guards against onTerminal firing more than once: refetchInterval only
  // stops the interval trigger, not other TanStack Query refetch triggers
  // (e.g. refetchOnWindowFocus, on by default). A refetch after the run is
  // already terminal returns a new `data` object even with the same status,
  // which would otherwise re-run this effect and call onTerminal again.
  const hasFiredTerminal = useRef(false);
  useEffect(() => {
    if (data && TERMINAL_STATUSES.includes(data.status) && !hasFiredTerminal.current) {
      hasFiredTerminal.current = true;
      onTerminal();
    }
  }, [data, onTerminal]);

  if (isError) {
    return (
      <p className="text-red-600 text-sm">
        {error instanceof Error ? error.message : "Failed to load run status"}
      </p>
    );
  }

  if (!data) {
    return <p className="text-gray-500">Loading run status...</p>;
  }

  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <span className={`rounded-full px-3 py-1 text-sm font-medium ${BADGE_STYLES[data.status]}`}>
        {data.status}
      </span>
      <span className="text-gray-600 text-sm">
        {data.completed_steps} / {data.total_steps} steps
      </span>
    </div>
  );
}
