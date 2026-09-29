import { motion } from 'framer-motion'
import type { PipelineStage, RankProgress } from '../../types'

// Rough share of total time each stage takes, so the bar moves steadily.
const STAGES: { key: PipelineStage; label: string; start: number; span: number }[] = [
  { key: 'extracting', label: 'Reading resumes', start: 0, span: 5 },
  { key: 'ranking', label: 'Scoring against the role', start: 5, span: 5 },
  { key: 'enriching', label: 'Checking GitHub & portfolio links', start: 10, span: 10 },
  { key: 'assessing', label: 'Writing assessments', start: 20, span: 80 },
]

/** Pipeline progress as a checklist of stages, with an overall bar. */
export function RankProgressBar({ progress }: { progress: RankProgress | null }) {
  const activeIndex = progress ? STAGES.findIndex(s => s.key === progress.stage) : -1
  const active = STAGES[activeIndex]
  const fraction = progress && progress.total > 0 ? progress.done / progress.total : 0
  const pct = active ? Math.round(active.start + active.span * fraction) : 0

  return (
    <div role="status" aria-live="polite" className="border border-ink bg-card p-4">
      <div className="flex items-baseline justify-between">
        <span className="label text-ink-2">In progress</span>
        <span className="font-display text-3xl leading-none tabular-nums">
          {pct}<span className="text-base text-ink-3">%</span>
        </span>
      </div>

      <div className="mt-3 h-1 bg-paper-sunk">
        <motion.div
          className="h-full bg-accent"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>

      <ol className="mt-4 space-y-1.5">
        {STAGES.map((stage, i) => {
          const done = i < activeIndex || (i === activeIndex && fraction >= 1)
          const current = i === activeIndex && !done
          const counter =
            current && progress && progress.total > 1 && stage.key !== 'ranking'
              ? `${progress.done} / ${progress.total}`
              : ''
          return (
            <li
              key={stage.key}
              className={`flex items-center gap-3 text-sm transition-colors ${
                done ? 'text-ink-3' : current ? 'text-ink' : 'text-ink-3 opacity-60'
              }`}
            >
              <span className="w-4 text-center font-mono text-xs" aria-hidden>
                {done ? '✓' : current ? <span className="inline-block animate-pulse text-accent">●</span> : '○'}
              </span>
              <span className={done ? 'line-through decoration-ink-3' : ''}>{stage.label}</span>
              <span className="ml-auto font-mono text-[11px] tabular-nums text-ink-2">{counter}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
