import Link from "next/link";
import { listSuites } from "@/lib/api";
import { SuiteForm } from "@/components/SuiteForm";

export const dynamic = "force-dynamic";

export default async function SuitesPage() {
  let suites: Awaited<ReturnType<typeof listSuites>> = [];
  let loadError: string | null = null;
  try {
    suites = await listSuites();
  } catch (err) {
    loadError = err instanceof Error ? err.message : "Failed to load suites";
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold mb-4">Suites</h1>
        {loadError ? (
          <p className="text-red-600 text-sm">{loadError}</p>
        ) : suites.length === 0 ? (
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
