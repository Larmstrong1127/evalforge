import { notFound } from "next/navigation";
import { listSuites } from "@/lib/api";
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
