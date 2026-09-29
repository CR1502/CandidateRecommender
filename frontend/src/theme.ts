import type { Recommendation } from './types'

// Presentation for the backend's category names. The API also sends an emoji
// and a hex colour per category; those are tuned for a dark neon UI, so the
// frontend uses its own ink tones (CSS variables in index.css) instead.
export const CATEGORIES = [
  { key: 'Perfect Match', short: 'Perfect', color: 'var(--cat-perfect)' },
  { key: 'Ideal Candidate', short: 'Ideal', color: 'var(--cat-ideal)' },
  { key: 'Good Candidate', short: 'Good', color: 'var(--cat-good)' },
  { key: 'Okay Candidate', short: 'Okay', color: 'var(--cat-okay)' },
  { key: 'Not Recommended', short: 'Not recommended', color: 'var(--cat-not)' },
] as const

export function categoryStyle(category: string) {
  return CATEGORIES.find(c => c.key === category) ?? { key: category, short: category, color: 'var(--ink-2)' }
}

export const RECOMMENDATION_COLOR: Record<Recommendation, string> = {
  'Strong Yes': 'var(--cat-perfect)',
  Yes: 'var(--cat-ideal)',
  Maybe: 'var(--cat-good)',
  No: 'var(--cat-not)',
}

export function pad2(n: number): string {
  return String(n).padStart(2, '0')
}
