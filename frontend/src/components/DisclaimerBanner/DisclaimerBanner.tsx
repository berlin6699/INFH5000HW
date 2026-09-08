import type { SystemInfo } from '@/api/client'

/**
 * Persistent compliance banner.
 *
 * Two jobs, both non-negotiable for this prototype:
 *   1. State that the output is not a diagnosis.
 *   2. Surface whether the LLM layer is live and whether imaging output is
 *      mocked, so a viewer can never mistake demo data for a real model run.
 *
 * The badges are driven by /api/system/info rather than hardcoded, which means
 * they cannot drift out of sync with what the backend actually did.
 */
export function DisclaimerBanner({ info }: { info: SystemInfo }) {
  return (
    <div className="border-b border-clinical-border bg-clinical-surface">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-3 gap-y-2 px-6 py-3">
        <span className="rounded bg-risk-high/10 px-2 py-0.5 text-[11px] font-semibold tracking-wide text-risk-high uppercase">
          Research Prototype
        </span>

        <p className="min-w-0 flex-1 text-xs leading-relaxed text-clinical-muted">
          {info.safety.disclaimer}
        </p>

        <div className="flex items-center gap-2">
          <ModeBadge
            tone={info.llm.enabled ? 'live' : 'neutral'}
            label={info.llm.enabled ? `LLM: ${info.llm.model}` : 'LLM: offline / deterministic'}
          />
          <ModeBadge
            tone={info.imaging.is_real_model ? 'live' : 'warn'}
            label={
              info.imaging.is_real_model
                ? 'Imaging: real model'
                : `Imaging: ${info.imaging.mode}`
            }
          />
          <ModeBadge tone="neutral" label={`RAG: ${info.rag.retriever}`} />
        </div>
      </div>
    </div>
  )
}

function ModeBadge({
  tone,
  label,
}: {
  tone: 'live' | 'warn' | 'neutral'
  label: string
}) {
  const tones = {
    live: 'bg-risk-low/10 text-risk-low',
    warn: 'bg-risk-medium/10 text-risk-medium',
    neutral: 'bg-slate-100 text-clinical-muted',
  } as const

  return (
    <span
      className={`whitespace-nowrap rounded px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {label}
    </span>
  )
}
