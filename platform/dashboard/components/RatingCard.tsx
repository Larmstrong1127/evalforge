import type { ResultResponse } from "@/lib/types";

export function RatingCard({
  result,
  revealed,
  onChoose,
}: {
  result: ResultResponse;
  revealed: boolean;
  onChoose: () => void;
}) {
  return (
    <div className="flex-1 rounded border border-gray-300 p-4 space-y-3">
      <p className="whitespace-pre-wrap text-sm">
        {result.error ?? result.generated_text}
      </p>
      {revealed ? (
        <p className="text-xs text-gray-500 font-mono">{result.candidate_model}</p>
      ) : (
        <button
          type="button"
          onClick={onChoose}
          className="rounded bg-blue-600 px-4 py-2 text-white text-sm"
        >
          Choose this
        </button>
      )}
    </div>
  );
}
