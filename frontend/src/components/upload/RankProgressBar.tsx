import type { RankProgress } from '../../types'

const STAGES: Record<RankProgress['stage'], { label: string; start: number; span: number }> = {
  // Rough share of total time each stage takes, so the bar moves steadily.
  extracting: { label: 'Reading resumes', start: 0, span: 5 },
  ranking: { label: 'Scoring candidates', start: 5, span: 5 },
  enriching: { label: 'Checking GitHub and portfolio links', start: 10, span: 10 },
  assessing: { label: 'Writing assessments', start: 20, span: 80 },
}

export function RankProgressBar({ progress }: { progress: RankProgress | null }) {
  const stage = progress ? STAGES[progress.stage] : null
  const fraction = progress && progress.total > 0 ? progress.done / progress.total : 0
  const pct = stage ? Math.round(stage.start + stage.span * fraction) : 0
  const counter =
    progress && progress.total > 1 && progress.stage !== 'ranking'
      ? ` · ${progress.done} of ${progress.total}`
      : ''

  return (
    <div role="status" aria-live="polite" className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{stage ? `${stage.label}…${counter}` : 'Starting…'}</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: '#1e1e2e' }}>
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: 'linear-gradient(90deg, #6366f1, #8b5cf6)' }}
        />
      </div>
    </div>
  )
}
