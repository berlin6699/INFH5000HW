import { useEffect, useState } from 'react'
import { ApiError, fetchSystemInfo, type SystemInfo } from '@/api/client'
import { DisclaimerBanner } from '@/components/DisclaimerBanner/DisclaimerBanner'

type ConnState =
  | { kind: 'loading' }
  | { kind: 'ok'; info: SystemInfo }
  | { kind: 'error'; message: string }

/**
 * Dashboard shell.
 *
 * Phase 0 renders the layout skeleton and proves frontend -> backend
 * connectivity via /api/system/info. Each named panel is filled in by a later
 * phase; the placeholders are kept so the grid is reviewable early.
 */
function App() {
  const [conn, setConn] = useState<ConnState>({ kind: 'loading' })

  useEffect(() => {
    let cancelled = false
    fetchSystemInfo()
      .then((info) => !cancelled && setConn({ kind: 'ok', info }))
      .catch((err: unknown) => {
        if (cancelled) return
        const message =
          err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : err instanceof Error
              ? err.message
              : String(err)
        setConn({ kind: 'error', message })
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex min-h-full flex-col">
      {conn.kind === 'ok' && <DisclaimerBanner info={conn.info} />}

      <header className="border-b border-clinical-border bg-clinical-surface">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <h1 className="text-lg font-semibold tracking-tight">
            Multimodal Multi-Agent Healthcare Assistant
          </h1>
          <p className="mt-1 text-sm text-clinical-muted">
            Intelligent Triage &amp; Continuous Health Monitoring — longitudinal,
            multimodal decision-support prototype
          </p>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">
        {conn.kind === 'loading' && (
          <p className="text-sm text-clinical-muted">Connecting to backend…</p>
        )}

        {conn.kind === 'error' && (
          <div className="rounded-lg border border-risk-high/30 bg-risk-high/5 p-4">
            <p className="text-sm font-semibold text-risk-high">
              Backend unreachable
            </p>
            <p className="mt-1 text-sm text-clinical-muted">{conn.message}</p>
            <p className="mt-2 text-xs text-clinical-muted">
              Start it with <code className="font-mono">make backend</code> from
              the project root.
            </p>
          </div>
        )}

        {conn.kind === 'ok' && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Panel title="Patient Overview" phase="Phase 7" span={1} />
            <Panel title="Patient Timeline" phase="Phase 7" span={1} />
            <Panel title="Current Assessment" phase="Phase 7" span={1} />
            <Panel title="Monitoring Trends" phase="Phase 7" span={2} />
            <Panel title="Medical Imaging" phase="Phase 8" span={1} />
            <Panel title="Agent Reasoning & Evidence" phase="Phase 7" span={3} />
          </div>
        )}
      </main>

      <footer className="border-t border-clinical-border bg-clinical-surface">
        <div className="mx-auto max-w-7xl px-6 py-3 text-[11px] text-clinical-muted">
          Synthetic patient data only. No real protected health information is
          stored or processed by this prototype.
        </div>
      </footer>
    </div>
  )
}

function Panel({
  title,
  phase,
  span,
}: {
  title: string
  phase: string
  span: 1 | 2 | 3
}) {
  const spans = { 1: 'lg:col-span-1', 2: 'lg:col-span-2', 3: 'lg:col-span-3' }
  return (
    <section
      className={`rounded-lg border border-clinical-border bg-clinical-surface p-4 ${spans[span]}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold">{title}</h2>
        <span className="text-[11px] text-clinical-muted">{phase}</span>
      </div>
      <p className="mt-3 text-xs text-clinical-muted">Not yet implemented.</p>
    </section>
  )
}

export default App
