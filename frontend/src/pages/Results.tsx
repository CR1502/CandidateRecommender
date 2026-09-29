import { useMemo } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { motion, type Variants } from 'framer-motion'
import { ArrowLeft, Download, Search } from 'lucide-react'
import { useAppStore } from '../store/useAppStore'
import { CandidateCard } from '../components/results/CandidateCard'
import { CategoryStrip } from '../components/results/CategoryStrip'

const STANDOUT = new Set(['Perfect Match', 'Ideal Candidate'])

function csvCell(value: string | number): string {
  const s = String(value)
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function buildCSV(candidates: ReturnType<typeof useAppStore.getState>['candidates']): string {
  const header = ['Rank', 'Name', 'Score (%)', 'Category', 'AI Recommendation', 'Matching Skills', 'Strengths', 'Gaps', 'Email', 'Phone', 'LinkedIn', 'Summary']
  const rows = candidates.map(c => [
    c.rank,
    c.candidate_name,
    c.percentage_score.toFixed(1),
    c.category,
    c.recommendation ?? '',
    c.matching_skills.join('; '),
    c.strengths.join('; '),
    c.gaps.join('; '),
    c.contact.email ?? '',
    c.contact.phone ?? '',
    c.contact.linkedin ?? '',
    c.fit_summary ?? '',
  ])
  return [header, ...rows].map(r => r.map(csvCell).join(',')).join('\n')
}

/** First meaningful line of the job description, for the report's "Re:" line. */
function roleLine(jd: string): string {
  const line = jd.split('\n').map(l => l.trim()).find(l => l.length > 0) ?? ''
  return line.length > 110 ? `${line.slice(0, 107)}…` : line
}

const page: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.08 } } }
const rise: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
}

export default function Results() {
  const {
    candidates, totalProcessed, totalDurationMs, jobDescriptionSnapshot,
    showNotRecommended, setShowNotRecommended,
    categoryFilter, setCategoryFilter,
    searchQuery, setSearchQuery,
  } = useAppStore()

  const filtered = useMemo(() => {
    let list = candidates
    // Picking a category shows it even if it's the hidden "Not Recommended" one
    if (categoryFilter) list = list.filter(c => c.category === categoryFilter)
    else if (!showNotRecommended) list = list.filter(c => c.category !== 'Not Recommended')
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      list = list.filter(c =>
        c.candidate_name.toLowerCase().includes(q) ||
        c.matching_skills.some(s => s.toLowerCase().includes(q)) ||
        c.fit_summary.toLowerCase().includes(q)
      )
    }
    return list
  }, [candidates, showNotRecommended, categoryFilter, searchQuery])

  const downloadCSV = () => {
    const csv = buildCSV(candidates)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'candidates.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (candidates.length === 0) {
    return <Navigate to="/" replace />
  }

  const standouts = candidates.filter(c => STANDOUT.has(c.category)).length
  const aiWritten = candidates.some(c => c.summary_source === 'llm')
  const hiddenCount = categoryFilter || showNotRecommended
    ? 0
    : candidates.filter(c => c.category === 'Not Recommended').length
  const role = roleLine(jobDescriptionSnapshot)

  return (
    <motion.main variants={page} initial="hidden" animate="show" className="mx-auto max-w-6xl px-4 sm:px-8 pb-24">
      {/* Report header */}
      <section className="grid gap-6 border-b border-ink py-10 md:grid-cols-[1fr_auto] md:items-end">
        <div>
          <motion.div variants={rise} className="flex items-center gap-4">
            <Link to="/" className="label flex items-center gap-1.5 text-ink-2 hover:text-accent transition-colors">
              <ArrowLeft size={12} /> New search
            </Link>
            <span className="label text-accent">Shortlist report</span>
          </motion.div>
          <motion.h1 variants={rise} className="mt-4 font-display text-[clamp(2.25rem,6vw,4.5rem)] leading-[0.95] tracking-[-0.02em]">
            {totalProcessed} {totalProcessed === 1 ? 'resume' : 'resumes'} read.{' '}
            <span className={standouts > 0 ? 'text-accent' : 'text-ink-3'}>
              {standouts === 0 ? 'None clearly stand out.' : `${standouts} stand${standouts === 1 ? 's' : ''} out.`}
            </span>
          </motion.h1>
          {role && (
            <motion.p variants={rise} className="mt-4 max-w-2xl text-ink-2">
              <span className="label mr-2 text-ink">Re:</span>
              {role}
            </motion.p>
          )}
        </div>

        <motion.dl variants={rise} className="grid grid-cols-3 gap-6 md:grid-cols-1 md:gap-2 md:text-right">
          <Meta label="Shown" value={`Top ${candidates.length}`} />
          <Meta label="Time" value={`${(totalDurationMs / 1000).toFixed(1)}s`} />
          <Meta label="Notes by" value={aiWritten ? 'Local AI' : 'Template'} />
        </motion.dl>
      </section>

      {/* Distribution, filters, export */}
      <motion.section variants={rise} className="border-b border-rule py-6">
        <CategoryStrip candidates={candidates} active={categoryFilter} onSelect={setCategoryFilter} />
        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3">
          <label className="flex min-w-[14rem] flex-1 items-center gap-2 border-b border-ink pb-1 sm:flex-none">
            <Search size={14} className="text-ink-3" aria-hidden />
            <span className="sr-only">Search candidates</span>
            <input
              type="search"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search name, skill, summary"
              className="w-full bg-transparent text-sm placeholder:text-ink-3 focus:outline-none"
            />
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={showNotRecommended}
              onChange={e => setShowNotRecommended(e.target.checked)}
              className="h-3.5 w-3.5 accent-[var(--accent)]"
            />
            Include not recommended
          </label>
          <button
            onClick={downloadCSV}
            className="ml-auto flex items-center gap-2 border border-ink px-3 py-1.5 text-sm transition-colors hover:bg-ink hover:text-paper"
          >
            <Download size={14} /> Export CSV
          </button>
        </div>
      </motion.section>

      {/* Candidates */}
      <section className="mt-6 space-y-3">
        {filtered.length === 0 ? (
          <motion.p variants={rise} className="py-16 text-center font-display text-2xl text-ink-3">
            No candidates match these filters.
          </motion.p>
        ) : (
          filtered.map((c, i) => (
            <motion.div
              key={`${c.filename}-${c.rank}`}
              initial={{ opacity: 0, y: 24, rotate: i % 2 ? 0.4 : -0.4 }}
              animate={{ opacity: 1, y: 0, rotate: 0 }}
              transition={{ delay: 0.25 + i * 0.07, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            >
              <CandidateCard candidate={c} stampDelay={0.55 + i * 0.07} />
            </motion.div>
          ))
        )}
        {hiddenCount > 0 && (
          <p className="pt-2 text-center text-sm text-ink-3">
            {hiddenCount} not-recommended {hiddenCount === 1 ? 'candidate is' : 'candidates are'} hidden.{' '}
            <button onClick={() => setShowNotRecommended(true)} className="underline underline-offset-4 hover:text-accent">
              Show
            </button>
          </p>
        )}
      </section>
    </motion.main>
  )
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="md:flex md:items-baseline md:justify-end md:gap-3">
      <dt className="label">{label}</dt>
      <dd className="font-display text-xl">{value}</dd>
    </div>
  )
}
