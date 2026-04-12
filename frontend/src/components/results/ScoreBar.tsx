import type { Candidate } from '../../types'

interface Props {
  candidates: Candidate[]
}

const CATEGORIES = [
  { label: 'Perfect',  color: '#00D26A', key: 'Perfect Match' },
  { label: 'Ideal',    color: '#4CAF50', key: 'Ideal Candidate' },
  { label: 'Good',     color: '#FFA726', key: 'Good Candidate' },
  { label: 'Okay',     color: '#FF9800', key: 'Okay Candidate' },
  { label: 'Not Rec.', color: '#F44336', key: 'Not Recommended' },
]

export function ScoreBar({ candidates }: Props) {
  const counts = CATEGORIES.map(cat => ({
    ...cat,
    count: candidates.filter(c => c.category === cat.key).length,
  }))

  const total = candidates.length || 1

  return (
    <div className="rounded-xl p-4" style={{ background: '#12121a', border: '1px solid #1e1e2e' }}>
      <div className="flex gap-1 h-3 rounded-full overflow-hidden mb-3">
        {counts.map(cat =>
          cat.count > 0 ? (
            <div
              key={cat.key}
              style={{ width: `${(cat.count / total) * 100}%`, background: cat.color }}
              className="transition-all duration-500"
            />
          ) : null
        )}
      </div>
      <div className="flex flex-wrap gap-4">
        {counts.map(cat => (
          <div key={cat.key} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: cat.color }}
            />
            <span className="text-xs text-slate-400">{cat.label}</span>
            <span className="text-xs font-bold text-slate-200">{cat.count}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
