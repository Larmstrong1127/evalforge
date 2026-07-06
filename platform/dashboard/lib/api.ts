import type {
  CostResponse,
  ResultResponse,
  RunAccepted,
  RunCreate,
  RunStatusResponse,
  SuiteCreate,
  SuiteResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    // FastAPI's 422 validation errors return `detail` as an array of error
    // objects, not a string — normalize so ApiError.message is always readable.
    const detail =
      typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? response.statusText);
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export function listSuites(): Promise<SuiteResponse[]> {
  return request<SuiteResponse[]>("/api/v1/suites");
}

export function createSuite(body: SuiteCreate): Promise<SuiteResponse> {
  return request<SuiteResponse>("/api/v1/suites", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function createRun(body: RunCreate): Promise<RunAccepted> {
  return request<RunAccepted>("/api/v1/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getRunStatus(runId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/api/v1/runs/${runId}`);
}

export function getRunResults(runId: string): Promise<ResultResponse[]> {
  return request<ResultResponse[]>(`/api/v1/runs/${runId}/results`);
}

export function getRunCosts(runId: string): Promise<CostResponse> {
  return request<CostResponse>(`/api/v1/runs/${runId}/costs`);
}

export { ApiError };
