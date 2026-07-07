"use client";

import { use, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { createRating, getRunResults } from "@/lib/api";
import { buildRatingPairs } from "@/lib/pairing";
import { RatingCard } from "@/components/RatingCard";

export default function RatingRoomPage({
  params,
}: {
  params: Promise<{ suiteId: string }>;
}) {
  const { suiteId } = use(params);
  const searchParams = useSearchParams();
  const runId = searchParams.get("runId");

  const raterSessionRef = useRef<string>(crypto.randomUUID());
  const [currentIndex, setCurrentIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [voteCount, setVoteCount] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const resultsQuery = useQuery({
    queryKey: ["run-results-for-rating", runId],
    queryFn: () => getRunResults(runId!),
    enabled: !!runId,
  });

  // Derived, not stateful: recomputing from resultsQuery.data is cheap and
  // keeps the pair order stable across re-renders because buildRatingPairs
  // is seeded deterministically by runId.
  const pairs = useMemo(
    () => (resultsQuery.data ? buildRatingPairs(resultsQuery.data, runId ?? "") : null),
    [resultsQuery.data, runId]
  );

  const currentPair = useMemo(
    () => (pairs && currentIndex < pairs.length ? pairs[currentIndex] : null),
    [pairs, currentIndex]
  );

  async function submitVote(chosenResultId: string | null, skipped: boolean) {
    if (submitting || !currentPair) return;
    setSubmitting(true);
    try {
      await createRating({
        prompt_version_id: currentPair.a.prompt_version_id,
        result_a_id: currentPair.a.id,
        result_b_id: currentPair.b.id,
        chosen_result_id: chosenResultId,
        skipped,
        rater_session: raterSessionRef.current,
      });
      setVoteCount((count) => count + 1);
      setRevealed(true);
    } finally {
      setSubmitting(false);
    }
  }

  function advance() {
    setRevealed(false);
    setCurrentIndex((i) => i + 1);
  }

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (revealed || submitting || !currentPair) return;
      if (e.key === "ArrowLeft") {
        submitVote(currentPair.a.id, false);
      } else if (e.key === "ArrowRight") {
        submitVote(currentPair.b.id, false);
      } else if (e.key === "s" || e.key === "S") {
        submitVote(null, true);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  if (!runId) {
    return <p className="text-red-600 text-sm">Missing runId query parameter.</p>;
  }

  if (resultsQuery.isError) {
    return (
      <p className="text-red-600 text-sm">
        {resultsQuery.error instanceof Error ? resultsQuery.error.message : "Failed to load results"}
      </p>
    );
  }

  if (!pairs) {
    return <p className="text-gray-500">Loading results to rate...</p>;
  }

  if (!currentPair) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Done rating</h1>
        <p>You cast {voteCount} vote{voteCount === 1 ? "" : "s"}.</p>
        <Link href={`/suites/${suiteId}`} className="text-blue-600 hover:underline">
          Back to suite
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4" role="region" aria-label="Rating room">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Rate this pair</h1>
        <p className="text-gray-500 text-sm" role="status" aria-live="polite">
          {currentIndex + 1} / {pairs.length} · {voteCount} vote{voteCount === 1 ? "" : "s"} cast
        </p>
      </div>
      <div className="flex gap-4">
        <RatingCard
          result={currentPair.a}
          revealed={revealed}
          onChoose={() => submitVote(currentPair.a.id, false)}
        />
        <RatingCard
          result={currentPair.b}
          revealed={revealed}
          onChoose={() => submitVote(currentPair.b.id, false)}
        />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => submitVote(null, true)}
          disabled={submitting || revealed}
          className="rounded border border-gray-300 px-4 py-2 text-sm disabled:opacity-50"
        >
          Skip (S)
        </button>
        {revealed && (
          <button
            type="button"
            onClick={advance}
            className="rounded bg-blue-600 px-4 py-2 text-white text-sm"
          >
            Next pair
          </button>
        )}
        <p className="text-gray-500 text-xs">
          Keyboard: ← left, → right, S to skip
        </p>
      </div>
    </div>
  );
}
