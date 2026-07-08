import type { ResultResponse } from "./types";

export interface RatingPair {
  a: ResultResponse;
  b: ResultResponse;
}

/**
 * Deterministic string hash → 32-bit int, used to seed the PRNG. Not
 * cryptographic — this only needs to produce a stable, well-distributed
 * seed from a runId string, not resist adversarial input.
 */
function hashSeed(seed: string): number {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (Math.imul(31, hash) + seed.charCodeAt(i)) | 0;
  }
  return hash >>> 0;
}

/** Mulberry32 PRNG — small, seedable, sufficient for shuffle ordering (not
 * used for anything security-sensitive). */
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(items: T[], rand: () => number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Groups results by prompt_version_id, generates every unique unordered pair
 * of results within each group (skipping groups with fewer than 2 results),
 * then shuffles both the pair order and each pair's internal (a, b) side
 * assignment using a PRNG seeded from `seed` — stable across repeated calls
 * with the same seed (e.g. reloading the rating room mid-session), different
 * across different seeds (e.g. different runs).
 */
export function buildRatingPairs(results: ResultResponse[], seed: string): RatingPair[] {
  const byPromptVersion = new Map<string, ResultResponse[]>();
  for (const result of results) {
    const group = byPromptVersion.get(result.prompt_version_id) ?? [];
    group.push(result);
    byPromptVersion.set(result.prompt_version_id, group);
  }

  const rand = mulberry32(hashSeed(seed));
  const pairs: RatingPair[] = [];
  // Map iteration order for a JS Map is insertion order, which depends on
  // the input array's order — sort group keys for determinism independent
  // of the caller's array ordering.
  const sortedPromptVersionIds = [...byPromptVersion.keys()].sort();
  for (const promptVersionId of sortedPromptVersionIds) {
    // Non-null: promptVersionId comes directly from byPromptVersion.keys()
    // one line above, with no mutation of the map in between.
    const group = byPromptVersion.get(promptVersionId)!;
    if (group.length < 2) continue;
    const sortedGroup = [...group].sort((x, y) => x.id.localeCompare(y.id));
    for (let i = 0; i < sortedGroup.length; i++) {
      for (let j = i + 1; j < sortedGroup.length; j++) {
        const [a, b] = rand() < 0.5 ? [sortedGroup[i], sortedGroup[j]] : [sortedGroup[j], sortedGroup[i]];
        pairs.push({ a, b });
      }
    }
  }

  return seededShuffle(pairs, rand);
}
