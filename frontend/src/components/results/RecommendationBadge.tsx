import { motion } from 'framer-motion'
import type { Recommendation } from '../../types'
import { RECOMMENDATION_COLOR } from '../../theme'

// Each verdict lands at a slightly different angle, like a real stamp.
const TILT: Record<Recommendation, number> = { 'Strong Yes': -7, Yes: -4, Maybe: 3, No: -2 }

/**
 * The LLM's hiring recommendation, drawn as a rubber stamp. Absent for
 * template (non-LLM) summaries.
 */
export function RecommendationBadge({ value, delay = 0 }: { value: Recommendation; delay?: number }) {
  const color = RECOMMENDATION_COLOR[value]
  return (
    <motion.span
      initial={{ scale: 2.2, opacity: 0, rotate: TILT[value] - 14 }}
      animate={{ scale: 1, opacity: 1, rotate: TILT[value] }}
      transition={{ delay, type: 'spring', stiffness: 520, damping: 20, mass: 0.7 }}
      className="stamp-ink inline-flex flex-col items-center whitespace-nowrap border-[2.5px] px-2 py-0.5 leading-none"
      style={{ color, borderColor: color, boxShadow: `inset 0 0 0 1.5px var(--card), inset 0 0 0 2.5px ${color}` }}
      title="AI recommendation: a starting point for a human reviewer, not a decision"
    >
      <span className="font-mono text-[7px] uppercase tracking-label opacity-80">AI verdict</span>
      <span className="font-mono text-[12px] font-medium uppercase tracking-[0.08em]">{value}</span>
    </motion.span>
  )
}
