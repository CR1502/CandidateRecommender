/**
 * The three composite-score components as a ledger with bars. A null
 * component didn't apply to this job (e.g. the job lists no recognisable
 * skills) and is shown as "n/a".
 */
interface Props {
  semantic: number
  skillCoverage: number | null
  experience: number | null
  color: string
}

const ROWS = [
  { key: 'semantic', label: 'Semantic match', weight: '60%' },
  { key: 'skillCoverage', label: 'Skill coverage', weight: '30%' },
  { key: 'experience', label: 'Experience', weight: '10%' },
] as const

export function ScoreBreakdown({ color, ...values }: Props) {
  return (
    <dl className="space-y-3">
      {ROWS.map(({ key, label, weight }) => {
        const value = values[key]
        return (
          <div key={key}>
            <div className="flex items-baseline text-sm">
              <dt>
                {label} <span className="font-mono text-[10px] text-ink-3">×{weight}</span>
              </dt>
              <span className="leader" aria-hidden />
              <dd className="font-mono text-xs tabular-nums" style={{ color: value === null ? 'var(--ink-3)' : color }}>
                {value === null ? 'n/a' : `${Math.round(value * 100)}%`}
              </dd>
            </div>
            <div className="mt-1 h-[3px] bg-paper-sunk">
              {value !== null && (
                <div
                  className="h-full"
                  style={{ width: `${value * 100}%`, background: color, transition: 'width 0.6s ease-out' }}
                />
              )}
            </div>
          </div>
        )
      })}
    </dl>
  )
}
