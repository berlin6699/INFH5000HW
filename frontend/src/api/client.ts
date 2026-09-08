/**
 * Backend API client.
 *
 * All calls go through the Vite dev proxy (`/api` -> 127.0.0.1:8000), so no
 * absolute URL and no CORS handling is needed here.
 */

export class ApiError extends Error {
  readonly status: number
  readonly detail?: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })

  if (!res.ok) {
    let detail: unknown
    try {
      detail = await res.json()
    } catch {
      detail = await res.text().catch(() => undefined)
    }
    throw new ApiError(res.status, `${init?.method ?? 'GET'} ${path} failed`, detail)
  }

  // 204 and empty bodies must not be JSON-parsed.
  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

export const get = <T>(path: string) => request<T>(path)
export const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

// --- System ---------------------------------------------------------------

export interface SystemInfo {
  llm: {
    provider: 'null' | 'openai' | 'deepseek'
    enabled: boolean
    model: string | null
    note: string
  }
  imaging: {
    mode: 'mock_preset' | 'uploaded_report'
    is_real_model: boolean
    badge: string | null
    supported_modalities: string[]
  }
  rag: { retriever: 'bm25' | 'embedding'; top_k: number }
  safety: {
    disclaimer: string
    guard_enforced: boolean
    produces_diagnosis: boolean
  }
}

export const fetchSystemInfo = () => get<SystemInfo>('/api/system/info')
export const fetchHealth = () => get<{ status: string }>('/api/health')
