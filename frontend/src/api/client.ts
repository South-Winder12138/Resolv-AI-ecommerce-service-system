export const API_BASE = (import.meta.env?.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

export class ApiError extends Error {
  code: string
  runId: string | null
  status: number
  retryable: boolean
  submissionId: string | null
  constructor(status: number, code: string, runId: string | null = null, retryable = false, submissionId: string | null = null) {
    super(code)
    this.code = code
    this.runId = runId
    this.status = status
    this.retryable = retryable
    this.submissionId = submissionId
  }
}

export async function responseError(response: Response): Promise<ApiError> {
  try {
    const value = await response.json()
    const detail = value.detail
    return new ApiError(response.status, typeof detail === 'object' && detail ? String(detail.code || 'invalid_request') : String(detail || 'http_error'), typeof detail?.run_id === 'string' ? detail.run_id : null, detail?.retryable === true, typeof detail?.submission_id === 'string' ? detail.submission_id : null)
  } catch {
    return new ApiError(response.status, 'http_error')
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) throw await responseError(response)
  return response.json() as Promise<T>
}
