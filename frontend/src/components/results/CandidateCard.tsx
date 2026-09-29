import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, ChevronUp, Mail, Phone, ExternalLink, Code2, MapPin, Globe, Check, CircleHelp } from 'lucide-react'
import type { Candidate } from '../../types'
import { RecommendationBadge } from './RecommendationBadge'
import { ScoreBreakdown } from './ScoreBreakdown'
import { ScoreRing } from './ScoreRing'

interface Props {
  candidate: Candidate
}

export function CandidateCard({ candidate: c }: Props) {
  const [expanded, setExpanded] = useState(false)

  const rankBg =
    c.rank === 1 ? '#f59e0b' :
    c.rank === 2 ? '#94a3b8' :
    c.rank === 3 ? '#b45309' : '#2d2d44'

  return (
    <motion.div
      layout
      className="rounded-xl overflow-hidden"
      style={{ background: '#12121a', border: '1px solid #1e1e2e' }}
    >
      {/* Header row */}
      <div className="flex items-center gap-4 p-4">
        {/* Rank badge */}
        <div
          className="shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold text-white"
          style={{ background: rankBg }}
        >
          #{c.rank}
        </div>

        {/* Name + category */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-100 truncate">{c.candidate_name}</span>
            <span className="text-base">{c.category_emoji}</span>
            {c.recommendation && <RecommendationBadge value={c.recommendation} />}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs" style={{ color: c.category_color }}>{c.category}</span>
            {c.contact.location && (
              <span className="text-xs text-slate-600 truncate">· {c.contact.location}</span>
            )}
          </div>
        </div>

        {/* Score */}
        <div className="shrink-0 text-right">
          <div className="text-xl font-bold" style={{ color: c.category_color }}>
            {c.percentage_score.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-600">match</div>
        </div>

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded(v => !v)}
          className="shrink-0 text-slate-500 hover:text-slate-300 transition-colors p-1"
          aria-label={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {/* Progress bar */}
      <div className="h-1 mx-4 rounded-full overflow-hidden" style={{ background: '#1e1e2e' }}>
        <motion.div
          className="h-full rounded-full"
          style={{ background: c.category_color }}
          initial={{ width: 0 }}
          animate={{ width: `${c.percentage_score}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>

      {/* Skills row */}
      {c.matching_skills.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-4 pt-3 pb-2">
          {c.matching_skills.slice(0, 8).map(skill => (
            <span
              key={skill}
              className="text-xs px-2 py-0.5 rounded-full font-medium"
              style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.25)' }}
            >
              {skill}
            </span>
          ))}
          {c.matching_skills.length > 8 && (
            <span className="text-xs text-slate-600 self-center">
              +{c.matching_skills.length - 8} more
            </span>
          )}
        </div>
      )}

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            key="detail"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4 border-t" style={{ borderColor: '#1e1e2e' }}>
              {/* Fit summary */}
              <div className="pt-4">
                <p className="text-sm text-slate-300 leading-relaxed">{c.fit_summary}</p>
                <p className="text-[11px] text-slate-600 mt-1.5">
                  {c.summary_source === 'llm'
                    ? 'Written by a local AI model from the resume (personal details hidden from it). Verify before acting on it.'
                    : 'Template summary — start Ollama for an AI-written assessment.'}
                </p>
              </div>

              {/* Strengths and gaps (LLM assessments only) */}
              {(c.strengths.length > 0 || c.gaps.length > 0) && (
                <div className="grid sm:grid-cols-2 gap-4">
                  <PointList title="Strengths" items={c.strengths} icon={<Check size={13} className="text-emerald-400" />} />
                  <PointList title="Gaps to probe" items={c.gaps} icon={<CircleHelp size={13} className="text-amber-400" />} />
                </div>
              )}

              {/* Score breakdown */}
              <div className="flex items-center gap-5 rounded-lg p-4" style={{ background: '#0d0d14' }}>
                <ScoreRing score={c.percentage_score} color={c.category_color} />
                <div className="flex-1 min-w-0">
                  <ScoreBreakdown
                    semantic={c.semantic_score}
                    skillCoverage={c.skill_coverage_score}
                    experience={c.experience_score}
                    color={c.category_color}
                  />
                </div>
              </div>

              {/* Contact info */}
              <ContactGrid contact={c.contact} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

/** Contact values may or may not already carry a scheme (e.g. "github.com/x" vs "https://site.dev"). */
function toHref(value: string | null | undefined): string | null {
  if (!value) return null
  return /^https?:\/\//i.test(value) ? value : `https://${value}`
}

function PointList({ title, items, icon }: { title: string; items: string[]; icon: React.ReactNode }) {
  if (items.length === 0) return null
  return (
    <div>
      <h4 className="text-xs font-semibold text-slate-400 mb-1.5">{title}</h4>
      <ul className="space-y-1">
        {items.map(item => (
          <li key={item} className="flex gap-2 text-sm text-slate-300">
            <span className="mt-1 shrink-0">{icon}</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ContactGrid({ contact }: { contact: Candidate['contact'] }) {
  const items = [
    { icon: <Mail size={13} />, label: 'Email', value: contact.email, href: contact.email ? `mailto:${contact.email}` : null },
    { icon: <Phone size={13} />, label: 'Phone', value: contact.phone, href: contact.phone ? `tel:${contact.phone}` : null },
    { icon: <ExternalLink size={13} />, label: 'LinkedIn', value: contact.linkedin, href: toHref(contact.linkedin) },
    { icon: <Code2 size={13} />, label: 'GitHub', value: contact.github, href: toHref(contact.github) },
    { icon: <MapPin size={13} />, label: 'Location', value: contact.location, href: null },
    { icon: <Globe size={13} />, label: 'Website', value: contact.website, href: toHref(contact.website) },
  ].filter(i => i.value)

  if (items.length === 0) return null

  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map(item => (
        <div
          key={item.label}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
          style={{ background: '#1a1a2e' }}
        >
          <span className="text-indigo-400 shrink-0">{item.icon}</span>
          {item.href ? (
            <a
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-300 hover:text-indigo-400 truncate transition-colors"
            >
              {item.value}
            </a>
          ) : (
            <span className="text-slate-300 truncate">{item.value}</span>
          )}
        </div>
      ))}
    </div>
  )
}
