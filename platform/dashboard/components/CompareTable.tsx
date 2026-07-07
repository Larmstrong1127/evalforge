import type { CompareResponse } from "@/lib/types";

function ScoreDelta({ judgeName, delta }: { judgeName: string; delta: number }) {
  const color = delta > 0 ? "text-green-600" : delta < 0 ? "text-red-600" : "text-gray-500";
  const sign = delta > 0 ? "+" : "";
  return (
    <div className={color}>
      {judgeName}: {sign}
      {delta.toFixed(2)}
    </div>
  );
}

export function CompareTable({ compare }: { compare: CompareResponse }) {
  if (compare.rows.length === 0) {
    return <p className="text-gray-500">No overlapping results between these two runs.</p>;
  }

  return (
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="border-b border-gray-200 text-left">
          <th className="py-2 pr-4">Prompt</th>
          <th className="py-2 pr-4">Candidate</th>
          <th className="py-2 pr-4">Run A output</th>
          <th className="py-2 pr-4">Run B output</th>
          <th className="py-2 pr-4">Score delta</th>
        </tr>
      </thead>
      <tbody>
        {compare.rows.map((row) => (
          <tr
            key={`${row.prompt_version_id}:${row.candidate_model}`}
            className="border-b border-gray-100 align-top"
          >
            <td className="py-2 pr-4 font-mono text-xs">
              {row.prompt_version_id.slice(0, 8)}
            </td>
            <td className="py-2 pr-4">{row.candidate_model}</td>
            <td className="py-2 pr-4 max-w-xs truncate" title={row.run_a_result?.generated_text}>
              {row.run_a_result ? row.run_a_result.generated_text : "— not present in this run —"}
            </td>
            <td className="py-2 pr-4 max-w-xs truncate" title={row.run_b_result?.generated_text}>
              {row.run_b_result ? row.run_b_result.generated_text : "— not present in this run —"}
            </td>
            <td className="py-2 pr-4">
              {Object.entries(row.score_delta).map(([judgeName, delta]) => (
                <ScoreDelta key={judgeName} judgeName={judgeName} delta={delta} />
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
