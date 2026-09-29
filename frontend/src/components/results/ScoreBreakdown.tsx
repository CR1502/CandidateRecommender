/**
 * The three composite-score components as labelled bars. Replaces the WebGL
 * radar chart. A null component didn't apply to this job (e.g. the job lists
 * no recognisable skills) and is shown as "n/a".
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
    <dl className="space-y-2.5">
      {ROWS.map(({ key, label, weight }) => {
        const value = values[key]
        return (
          <div key={key}>
            <div className="flex justify-between text-xs mb-1">
              <dt className="text-slate-400">
                {label} <span className="text-slate-600">· weight {weight}</span>
              </dt>
              <dd className="tabular-nums" style={{ color: value === null ? '#64748b' : color }}>
                {value === null ? 'n/a' : `${Math.round(value * 100)}%`}
              </dd>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: '#1e1e2e' }}>
              {value !== null && (
                <div
                  className="h-full rounded-full"
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
