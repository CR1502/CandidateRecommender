import { useMemo } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Download, Clock, Users } from 'lucide-react'
import { useAppStore } from '../store/useAppStore'
import { CandidateCard } from '../components/results/CandidateCard'
import { ScoreBar } from '../components/results/ScoreBar'

const CATEGORY_FILTERS = [
  'Perfect Match',
  'Ideal Candidate',
  'Good Candidate',
  'Okay Candidate',
  'Not Recommended',
]

function csvCell(value: string | number): string {
  const s = String(value)
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function buildCSV(candidates: ReturnType<typeof useAppStore.getState>['candidates']): string {
  const header = ['Rank', 'Name', 'Score (%)', 'Category', 'Matching Skills', 'Email', 'Phone', 'LinkedIn', 'Summary']
  const rows = candidates.map(c => [
    c.rank,
    c.candidate_name,
    c.percentage_score.toFixed(1),
    c.category,
    c.matching_skills.join('; '),
    c.contact.email ?? '',
    c.contact.phone ?? '',
    c.contact.linkedin ?? '',
    c.fit_summary ?? '',
  ])
  return [header, ...rows].map(r => r.map(csvCell).join(',')).join('\n')
}

export default function Results() {
  const navigate = useNavigate()
  const {
    candidates, totalProcessed, totalDurationMs,
    showNotRecommended, setShowNotRecommended,
    categoryFilter, setCategoryFilter,
    searchQuery, setSearchQuery,
  } = useAppStore()

  const filtered = useMemo(() => {
    let list = candidates
    if (!showNotRecommended) list = list.filter(c => c.category !== 'Not Recommended')
    if (categoryFilter) list = list.filter(c => c.category === categoryFilter)
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

  return (
    <div className="min-h-screen px-4 md:px-8 py-6 max-w-4xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-6 gap-4 flex-wrap"
      >
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ArrowLeft size={15} /> Back
          </button>
          <h2 className="text-xl font-bold text-slate-100">Results</h2>
          <span className="flex items-center gap-1.5 text-xs text-slate-500">
            <Users size={12} /> {totalProcessed} candidates
          </span>
          <span className="flex items-center gap-1.5 text-xs text-slate-500">
            <Clock size={12} /> {(totalDurationMs / 1000).toFixed(1)}s
          </span>
        </div>

        <button
          onClick={downloadCSV}
          className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg font-medium text-indigo-300 hover:text-indigo-200 transition-colors"
          style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)' }}
        >
          <Download size={14} /> Export CSV
        </button>
      </motion.div>

      {/* Score distribution bar */}
      <div className="mb-5">
        <ScoreBar candidates={candidates} />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-5">
        <input
          type="text"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          placeholder="Search name, skill, summary…"
          className="rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          style={{ background: '#12121a', border: '1px solid #1e1e2e', minWidth: 200 }}
        />
        <button
          onClick={() => setCategoryFilter(null)}
          className="text-xs px-3 py-1.5 rounded-lg transition-colors"
          style={{
            background: categoryFilter === null ? '#6366f1' : '#12121a',
            color: categoryFilter === null ? '#fff' : '#94a3b8',
            border: '1px solid #1e1e2e',
          }}
        >
          All
        </button>
        {CATEGORY_FILTERS.map(cat => (
          <button
            key={cat}
            onClick={() => setCategoryFilter(cat === categoryFilter ? null : cat)}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors"
            style={{
              background: categoryFilter === cat ? '#6366f1' : '#12121a',
              color: categoryFilter === cat ? '#fff' : '#94a3b8',
              border: '1px solid #1e1e2e',
            }}
          >
            {cat.split(' ')[0]}
          </button>
        ))}
        <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={showNotRecommended}
            onChange={e => setShowNotRecommended(e.target.checked)}
            className="accent-indigo-500"
          />
          Show Not Recommended
        </label>
      </div>

      {/* Candidate list */}
      <div className="space-y-3">
        {filtered.length === 0 ? (
          <p className="text-center text-slate-500 py-12">No candidates match the current filters.</p>
        ) : (
          filtered.map(c => (
            <motion.div
              key={`${c.filename}-${c.rank}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: c.rank * 0.04 }}
            >
              <CandidateCard candidate={c} />
            </motion.div>
          ))
        )}
      </div>
    </div>
  )
}
