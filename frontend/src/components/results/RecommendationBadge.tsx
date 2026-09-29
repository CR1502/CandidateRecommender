import type { Recommendation } from '../../types'

const STYLES: Record<Recommendation, { bg: string; fg: string }> = {
  'Strong Yes': { bg: 'rgba(0,210,106,0.15)', fg: '#34d399' },
  Yes: { bg: 'rgba(76,175,80,0.15)', fg: '#86efac' },
  Maybe: { bg: 'rgba(255,167,38,0.15)', fg: '#fbbf24' },
  No: { bg: 'rgba(244,67,54,0.15)', fg: '#f87171' },
}

/** The LLM's hiring recommendation. Absent for template (non-LLM) summaries. */
export function RecommendationBadge({ value }: { value: Recommendation }) {
  const { bg, fg } = STYLES[value]
  return (
    <span
      className="text-xs font-semibold px-2 py-0.5 rounded-full whitespace-nowrap"
      style={{ background: bg, color: fg }}
      title="AI recommendation — a starting point for a human reviewer, not a decision"
    >
      AI: {value}
    </span>
  )
}
