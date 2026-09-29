import type { Candidate } from '../../types'
import { CATEGORIES } from '../../theme'

interface Props {
  candidates: Candidate[]
  active: string | null
  onSelect: (category: string | null) => void
}

/**
 * How the shortlist splits across categories. Each segment of the bar and
 * each legend entry doubles as a filter.
 */
export function CategoryStrip({ candidates, active, onSelect }: Props) {
  const counts = CATEGORIES.map(cat => ({
    ...cat,
    count: candidates.filter(c => c.category === cat.key).length,
  }))
  const total = candidates.length || 1
  const toggle = (key: string) => onSelect(active === key ? null : key)

  return (
    <div>
      <div className="flex h-3 gap-[3px]" aria-hidden>
        {counts.map(cat =>
          cat.count > 0 ? (
            <button
              key={cat.key}
              tabIndex={-1}
              onClick={() => toggle(cat.key)}
              className="h-full transition-opacity"
              style={{
                width: `${(cat.count / total) * 100}%`,
                background: cat.color,
                opacity: active && active !== cat.key ? 0.25 : 1,
              }}
            />
          ) : null,
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2" role="group" aria-label="Filter by category">
        <FilterButton label="All" count={candidates.length} pressed={active === null} onClick={() => onSelect(null)} />
        {counts.map(cat => (
          <FilterButton
            key={cat.key}
            label={cat.short}
            count={cat.count}
            color={cat.color}
            pressed={active === cat.key}
            disabled={cat.count === 0}
            onClick={() => toggle(cat.key)}
          />
        ))}
      </div>
    </div>
  )
}

function FilterButton({ label, count, color, pressed, disabled, onClick }: {
  label: string
  count: number
  color?: string
  pressed: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-pressed={pressed}
      className={`flex items-center gap-1.5 border-b-2 pb-0.5 text-sm transition-colors disabled:opacity-35 ${
        pressed ? 'border-ink text-ink' : 'border-transparent text-ink-2 hover:text-ink'
      }`}
    >
      {color && <span className="inline-block h-2 w-2" style={{ background: color }} />}
      {label}
      <span className="font-mono text-[11px] tabular-nums text-ink-3">{count}</span>
    </button>
  )
}
