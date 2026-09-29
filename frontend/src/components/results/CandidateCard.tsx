import { useId, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import type { Candidate } from '../../types'
import { categoryStyle, pad2 } from '../../theme'
import { RecommendationBadge } from './RecommendationBadge'
import { ScoreBreakdown } from './ScoreBreakdown'

interface Props {
  candidate: Candidate
  /** Delay before the stamp lands, so stamps fall in rank order. */
  stampDelay?: number
}

const VISIBLE_SKILLS = 6

export function CandidateCard({ candidate: c, stampDelay = 0 }: Props) {
  const [expanded, setExpanded] = useState(false)
  const detailId = useId()
  const cat = categoryStyle(c.category)
  const [whole, fraction] = c.percentage_score.toFixed(1).split('.')

  return (
    <article className={`relative border bg-card transition-shadow ${expanded ? 'border-ink shadow-[6px_6px_0_var(--ink)]' : 'border-rule hover:border-ink-3'}`}>
      {/* Category edge */}
      <span className="absolute inset-y-0 left-0 w-1" style={{ background: cat.color }} aria-hidden />

      <button
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
        aria-controls={detailId}
        className="grid w-full grid-cols-[auto_1fr_auto] items-start gap-x-4 gap-y-3 py-4 pl-5 pr-4 text-left sm:grid-cols-[4.5rem_1fr_auto_auto] sm:items-center sm:gap-x-6 sm:pl-6"
      >
        {/* Rank */}
        <span
          className={`font-display text-4xl leading-none tabular-nums sm:text-5xl ${c.rank === 1 ? 'text-accent' : c.rank <= 3 ? 'text-ink' : 'text-ink-3'}`}
          aria-label={`Rank ${c.rank}`}
        >
          {pad2(c.rank)}
        </span>

        {/* Name, category, skills */}
        <span className="min-w-0">
          <span className="block break-words font-display text-2xl leading-tight">{c.candidate_name}</span>
          <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="label" style={{ color: cat.color }}>{cat.short}</span>
            {c.contact.location && <span className="label">· {c.contact.location}</span>}
          </span>
          {c.matching_skills.length > 0 && (
            <span className="mt-2.5 flex flex-wrap gap-x-1.5 gap-y-1 text-[13px] text-ink-2">
              {c.matching_skills.slice(0, VISIBLE_SKILLS).map((skill, i) => (
                <span key={skill}>
                  {skill}{i < Math.min(c.matching_skills.length, VISIBLE_SKILLS) - 1 && <span className="ml-1.5 text-ink-3">/</span>}
                </span>
              ))}
              {c.matching_skills.length > VISIBLE_SKILLS && (
                <span className="font-mono text-[11px] text-ink-3 self-center">
                  +{c.matching_skills.length - VISIBLE_SKILLS}
                </span>
              )}
            </span>
          )}
        </span>

        {/* Stamp: its own column on wide screens, under the name on narrow ones */}
        <span className="col-start-2 row-start-2 justify-self-start sm:col-start-3 sm:row-start-1 sm:justify-self-center">
          {c.recommendation && <RecommendationBadge value={c.recommendation} delay={stampDelay} />}
        </span>

        {/* Score */}
        <span className="col-start-3 row-start-1 flex items-start gap-2 sm:col-start-4">
          <span className="text-right">
            <span className="block font-display text-4xl leading-none tabular-nums">
              {whole}<span className="text-lg text-ink-3">.{fraction}</span>
            </span>
            <span className="mt-1.5 block h-[3px] w-16 bg-paper-sunk">
              <motion.span
                className="block h-full"
                style={{ background: cat.color }}
                initial={{ width: 0 }}
                animate={{ width: `${c.percentage_score}%` }}
                transition={{ duration: 0.8, delay: stampDelay * 0.6, ease: 'easeOut' }}
              />
            </span>
            <span className="label mt-1 block">match</span>
          </span>
          <ChevronDown
            size={18}
            className={`mt-2 text-ink-3 transition-transform duration-300 ${expanded ? 'rotate-180 text-ink' : ''}`}
            aria-hidden
          />
        </span>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            id={detailId}
            key="detail"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="mx-5 border-t border-dashed border-ink-3 pb-6 pt-5 sm:mx-6 sm:ml-[7.5rem]">
              {/* Assessment */}
              <p className="max-w-3xl text-[17px] leading-relaxed">{c.fit_summary}</p>
              <p className="label mt-2 normal-case tracking-normal">
                {c.summary_source === 'llm'
                  ? 'Written by a local AI model from the resume, with personal details hidden from it. Check it before acting on it.'
                  : 'Template summary. Start Ollama for an AI-written assessment.'}
              </p>

              {(c.strengths.length > 0 || c.gaps.length > 0) && (
                <div className="mt-6 grid gap-6 sm:grid-cols-2">
                  <PointList title="Strengths" mark="+" color="var(--cat-perfect)" items={c.strengths} />
                  <PointList title="Gaps to probe" mark="?" color="var(--cat-good)" items={c.gaps} />
                </div>
              )}

              <div className="mt-6 grid gap-6 sm:grid-cols-2">
                <div>
                  <h4 className="label mb-3 text-ink-2">Score breakdown</h4>
                  <ScoreBreakdown
                    semantic={c.semantic_score}
                    skillCoverage={c.skill_coverage_score}
                    experience={c.experience_score}
                    color={cat.color}
                  />
                </div>
                <ContactList contact={c.contact} filename={c.filename} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </article>
  )
}

/** Contact values may or may not already carry a scheme (e.g. "github.com/x" vs "https://site.dev"). */
function toHref(value: string | null | undefined): string | null {
  if (!value) return null
  return /^https?:\/\//i.test(value) ? value : `https://${value}`
}

function PointList({ title, mark, color, items }: { title: string; mark: string; color: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <div>
      <h4 className="label mb-2 text-ink-2">{title}</h4>
      <ul className="space-y-2">
        {items.map(item => (
          <li key={item} className="flex gap-3 text-[15px] leading-snug">
            <span className="w-3 shrink-0 font-mono text-sm font-medium" style={{ color }} aria-hidden>{mark}</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ContactList({ contact, filename }: { contact: Candidate['contact']; filename: string }) {
  const items = [
    { label: 'Email', value: contact.email, href: contact.email ? `mailto:${contact.email}` : null },
    { label: 'Phone', value: contact.phone, href: contact.phone ? `tel:${contact.phone}` : null },
    { label: 'LinkedIn', value: contact.linkedin, href: toHref(contact.linkedin) },
    { label: 'GitHub', value: contact.github, href: toHref(contact.github) },
    { label: 'Website', value: contact.website, href: toHref(contact.website) },
    { label: 'Location', value: contact.location, href: null },
    { label: 'File', value: filename, href: null },
  ].filter(i => i.value)

  return (
    <div>
      <h4 className="label mb-3 text-ink-2">On file</h4>
      <dl className="border-t border-rule">
        {items.map(item => (
          <div key={item.label} className="flex items-baseline gap-4 border-b border-rule py-1.5">
            <dt className="label w-16 shrink-0">{item.label}</dt>
            <dd className="min-w-0 truncate text-sm">
              {item.href ? (
                <a
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline decoration-rule decoration-1 underline-offset-4 hover:text-accent hover:decoration-accent"
                >
                  {item.value}
                </a>
              ) : (
                item.value
              )}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
