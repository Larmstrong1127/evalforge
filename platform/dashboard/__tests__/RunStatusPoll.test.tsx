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
    vi.spyOn(api, "getRunStatus").mockResolvedValue({
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

  it("renders an inline error message when the status fetch fails", async () => {
    vi.spyOn(api, "getRunStatus").mockRejectedValue(new Error("run not found"));

    renderWithClient(<RunStatusPoll runId="run-1" onTerminal={() => {}} />);

    await waitFor(() => expect(screen.getByText(/run not found/i)).toBeInTheDocument());
  });
});
