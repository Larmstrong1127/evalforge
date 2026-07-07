import { describe, expect, it } from "vitest";
import { buildRatingPairs } from "@/lib/pairing";
import type { ResultResponse } from "@/lib/types";

function makeResult(id: string, promptVersionId: string, candidate: string): ResultResponse {
  return {
    id,
    prompt_version_id: promptVersionId,
    candidate_model: candidate,
    status: "ok",
    generated_text: `answer from ${candidate}`,
    error: null,
    latency_ms: 10,
    cost_usd: 0,
    judge_evaluations: [],
  };
}

describe("buildRatingPairs", () => {
  it("pairs results within the same prompt_version and never across prompt_versions", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-2", "a"),
      makeResult("r4", "pv-2", "b"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    expect(pairs).toHaveLength(2);
    for (const pair of pairs) {
      expect(pair.a.prompt_version_id).toBe(pair.b.prompt_version_id);
    }
  });

  it("generates each unique candidate pair exactly once, never a result with itself", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    // C(3,2) = 3 unique pairs for one prompt_version with 3 candidates
    expect(pairs).toHaveLength(3);
    const seenKeys = new Set<string>();
    for (const pair of pairs) {
      expect(pair.a.id).not.toBe(pair.b.id);
      const key = [pair.a.id, pair.b.id].sort().join(":");
      expect(seenKeys.has(key)).toBe(false);
      seenKeys.add(key);
    }
  });

  it("skips prompt_versions with fewer than 2 results", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-2", "a"),
      makeResult("r3", "pv-2", "b"),
    ];
    const pairs = buildRatingPairs(results, "seed-1");
    expect(pairs).toHaveLength(1);
    expect(pairs[0].a.prompt_version_id).toBe("pv-2");
  });

  it("is deterministic for the same seed", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
      makeResult("r4", "pv-1", "d"),
    ];
    const first = buildRatingPairs(results, "same-seed");
    const second = buildRatingPairs(results, "same-seed");
    expect(first.map((p) => `${p.a.id}:${p.b.id}`)).toEqual(
      second.map((p) => `${p.a.id}:${p.b.id}`)
    );
  });

  it("produces a different order for a different seed (probabilistically, with a fixed fixture)", () => {
    const results = [
      makeResult("r1", "pv-1", "a"),
      makeResult("r2", "pv-1", "b"),
      makeResult("r3", "pv-1", "c"),
      makeResult("r4", "pv-1", "d"),
      makeResult("r5", "pv-1", "e"),
    ];
    const orderA = buildRatingPairs(results, "seed-a").map((p) => `${p.a.id}:${p.b.id}`);
    const orderB = buildRatingPairs(results, "seed-b").map((p) => `${p.a.id}:${p.b.id}`);
    expect(orderA).not.toEqual(orderB);
  });
});
