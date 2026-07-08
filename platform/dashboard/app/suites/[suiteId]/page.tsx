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
