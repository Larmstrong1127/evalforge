export interface SuiteResponse {
  id: string;
  name: string;
  description: string | null;
  prompt_count: number;
}

export interface PromptCreate {
  input_text: string;
  expected_output: string | null;
}

export interface SuiteCreate {
  name: string;
  description: string | null;
  prompts: PromptCreate[];
}

export interface RunCreate {
  suite_id: string;
  candidates: string[];
  judges: string[];
  concurrency: number;
}

export interface RunAccepted {
  run_id: string;
}

export type RunStatus = "queued" | "running" | "completed" | "failed";

export interface RunStatusResponse {
  id: string;
  status: RunStatus;
  completed_steps: number;
  total_steps: number;
  started_at: string | null;
  finished_at: string | null;
}

export interface JudgeEvaluationResponse {
  judge_name: string;
  score: number;
  justification: string | null;
}

export interface ResultResponse {
  id: string;
  prompt_version_id: string;
  candidate_model: string;
  status: string;
  generated_text: string;
  error: string | null;
  latency_ms: number;
  cost_usd: number;
  judge_evaluations: JudgeEvaluationResponse[];
}

export interface CostResponse {
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  by_candidate: Record<string, number>;
}
