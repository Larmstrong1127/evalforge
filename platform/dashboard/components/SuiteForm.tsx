"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createSuite } from "@/lib/api";
import type { PromptCreate } from "@/lib/types";

export function SuiteForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [promptsText, setPromptsText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    const prompts: PromptCreate[] = promptsText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => ({ input_text: line, expected_output: null }));
    try {
      const suite = await createSuite({
        name,
        description: description.trim() || null,
        prompts,
      });
      router.push(`/suites/${suite.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create suite");
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 max-w-lg">
      <div>
        <label className="block text-sm font-medium">Name</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Description</label>
        <input
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>
      <div>
        <label className="block text-sm font-medium">Prompts (one per line)</label>
        <textarea
          className="mt-1 block w-full rounded border border-gray-300 px-3 py-2 h-32"
          value={promptsText}
          onChange={(e) => setPromptsText(e.target.value)}
        />
      </div>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {submitting ? "Creating..." : "Create suite"}
      </button>
    </form>
  );
}
