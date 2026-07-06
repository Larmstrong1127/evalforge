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
    if (submitting) return;
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
        <label htmlFor="run-candidates" className="block text-sm font-medium">
          Candidates (comma-separated, provider:model)
        </label>
        <input
          id="run-candidates"
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={candidates}
          onChange={(e) => setCandidates(e.target.value)}
          required
        />
      </div>
      <div>
        <label htmlFor="run-judges" className="block text-sm font-medium">
          Judges (comma-separated)
        </label>
        <input
          id="run-judges"
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={judges}
          onChange={(e) => setJudges(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="run-concurrency" className="block text-sm font-medium">
          Concurrency
        </label>
        <input
          id="run-concurrency"
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
