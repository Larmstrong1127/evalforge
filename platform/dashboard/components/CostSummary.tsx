import type { CostResponse } from "@/lib/types";

export function CostSummary({ costs }: { costs: CostResponse }) {
  return (
    <div className="space-y-2">
      <p>
        Total: <strong>${costs.total_cost_usd.toFixed(4)}</strong> ·{" "}
        {costs.total_tokens_in} in / {costs.total_tokens_out} out tokens
      </p>
      <ul className="text-sm text-gray-600">
        {Object.entries(costs.by_candidate).map(([candidate, cost]) => (
          <li key={candidate}>
            {candidate}: ${cost.toFixed(4)}
          </li>
        ))}
      </ul>
    </div>
  );
}
