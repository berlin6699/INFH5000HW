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
    mode: 'mock_preset' | 'uploaded_report' | 'real_model'
    is_real_model: boolean
    badge: string | null
    supported_modalities: string[]
    local_model?: ImagingModelStatus
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

export interface Symptom {
  name: string
  label?: string | null
  present?: boolean
  value?: number | null
  unit?: string | null
  severity: 'mild' | 'moderate' | 'severe'
  onset_days?: number | null
  is_progressive?: boolean
  notes?: string | null
}

export interface MonitoringSample {
  recorded_at: string
  spo2: number | null
  heart_rate: number | null
  temperature_c: number | null
  respiratory_rate: number | null
  sleep_hours: number | null
  activity_steps: number | null
}

export interface DemoData {
  patient: {
    patient_id: string
    full_name: string
    age: number
    sex: string
    chronic_conditions: Array<{ name: string }>
    active_medications: Array<{ name: string; dose?: string | null; frequency?: string | null }>
    allergies: Array<{ allergen: string; reaction?: string | null; severity: string }>
    risk_factors: Array<{ factor: string; category: string }>
    timeline: Array<{ occurred_on: string; label: string; detail?: string | null; is_current: boolean }>
  }
  symptoms: { reported_at: string; free_text: string; symptoms: Symptom[] }
  monitoring: MonitoringSample[]
  imaging: ImagingResult | null
}

export interface ImagingResult {
    source_mode: string
    badge?: string | null
    provenance: string
    findings: string[]
    abnormalities: Array<{ label: string; description?: string | null; location?: string | null; confidence: number }>
    confidence: Record<string, number>
    summary: string
}

export interface ImagingModelStatus {
  available: boolean
  dependencies_installed: boolean
  weights_downloaded: boolean
  weights_bytes: number
  model: string
  device: string
  loaded: boolean
}

export interface AnalysisResult {
  run_id: string
  duration_ms: number
  status: string
  assessment: {
    risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
    risk_score: number
    recommended_department: string
    urgency: string
    care_advice: string
    key_findings: string[]
    longitudinal_summary: string
    reasoning_summary: string
    historical_changes: Array<{ dimension: string; change_type: string; significance: string; prior_state?: string | null; current_state: string; basis: string }>
    score_breakdown: Array<{ rule_id: string; category: string; description: string; contribution: number; evidence: string }>
    evidence: Array<{ chunk_id: string; text: string; source: string; relevance_score: number; retrieved_for: string }>
    limitations: string[]
    disclaimer: string
  }
  history: DemoData['patient']
  monitoring: { current_abnormalities: string[]; rapid_deterioration: boolean; summary: string }
  imaging: NonNullable<DemoData['imaging']>
  triage: { summary: string; warning_signs: Array<{ sign: string; description: string }> }
  knowledge: { summary: string }
  traces: Array<{ agent_name: string; status: string; duration_ms: number; notes: string[] }>
}

export interface AnalysisProgressEvent {
  type: 'progress'
  agent: string
  status: 'running' | 'completed'
  message: string
  completed: number
  total: number
}

export const fetchDemo = () => get<DemoData>('/api/demo')
export const analyseImage = async (patientId: string, file: File) => {
  const body = new FormData()
  body.append('patient_id', patientId)
  body.append('file', file)
  const response = await fetch('/api/imaging/analyze', { method: 'POST', body })
  if (!response.ok) {
    let detail: unknown
    try { detail = await response.json() } catch { detail = await response.text().catch(() => undefined) }
    throw new ApiError(response.status, '胸片分析失败', detail)
  }
  return response.json() as Promise<ImagingResult>
}
export const runAnalysis = (patientId: string, symptoms: Symptom[], freeText: string) =>
  post<AnalysisResult>('/api/analysis/run', {
    patient_id: patientId,
    symptoms,
    free_text: freeText,
    use_llm: false,
    rag_enabled: false,
    longitudinal_enabled: true,
  })

export async function runAnalysisStream(
  patientId: string,
  symptoms: Symptom[],
  freeText: string,
  onProgress: (event: AnalysisProgressEvent) => void,
): Promise<AnalysisResult> {
  const response = await fetch('/api/analysis/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patient_id: patientId,
      symptoms,
      free_text: freeText,
      use_llm: false,
      rag_enabled: false,
      longitudinal_enabled: true,
    }),
  })
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, '无法启动联合分析', await response.text().catch(() => undefined))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: AnalysisResult | null = null

  const handleLine = (line: string) => {
    if (!line.trim()) return
    const event = JSON.parse(line) as AnalysisProgressEvent | { type: 'result'; data: AnalysisResult } | { type: 'error'; message: string }
    if (event.type === 'progress') onProgress(event)
    else if (event.type === 'result') result = event.data
    else throw new Error(event.message)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    lines.forEach(handleLine)
    if (done) break
  }
  handleLine(buffer)
  if (!result) throw new Error('联合分析结束，但没有收到最终结果。')
  return result
}
