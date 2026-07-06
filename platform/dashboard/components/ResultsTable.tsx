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
