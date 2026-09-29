import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence, type Variants } from 'framer-motion'
import { ArrowRight, Loader2 } from 'lucide-react'
import { useAppStore } from '../store/useAppStore'
import { getErrorMessage, rankCandidates } from '../api/client'
import { DropZone } from '../components/upload/DropZone'
import { FileList } from '../components/upload/FileList'
import { RankProgressBar } from '../components/upload/RankProgressBar'
import type { RankProgress } from '../types'

const MIN_JD_CHARS = 50

// One orchestrated entrance: headline lines, then the two columns, then the action bar.
const page: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
}
const rise: Variants = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
}

const WEIGHTS = [
  ['Semantic match', '60'],
  ['Skill coverage', '30'],
  ['Experience', '10'],
] as const

export default function Home() {
  const navigate = useNavigate()
  const { jobDescription, setJobDescription, files, setResults } = useAppStore()
  const [validationError, setValidationError] = useState<string | null>(null)
  const [progress, setProgress] = useState<RankProgress | null>(null)
  const abort = useRef<AbortController | null>(null)

  const rank = useMutation({
    mutationFn: () => {
      abort.current = new AbortController()
      return rankCandidates(jobDescription, files, setProgress, 10, abort.current.signal)
    },
    onMutate: () => setProgress(null),
    onSuccess: (result) => {
      setResults(result)
      window.scrollTo(0, 0)
      navigate('/results')
    },
  })

  const isLoading = rank.isPending
  const cancelled = rank.error instanceof Error && rank.error.name === 'AbortError'
  const error = validationError ?? (rank.isError && !cancelled ? getErrorMessage(rank.error) : null)
  const jdLength = jobDescription.trim().length

  const handleSubmit = () => {
    if (jdLength < MIN_JD_CHARS) {
      setValidationError(`The job description needs at least ${MIN_JD_CHARS} characters.`)
      return
    }
    if (files.length === 0) {
      setValidationError('Add at least one resume to the tray.')
      return
    }
    setValidationError(null)
    rank.mutate()
  }

  return (
    <motion.main
      variants={page}
      initial="hidden"
      animate="show"
      className="mx-auto max-w-6xl px-4 sm:px-8 pb-20"
    >
      {/* Headline */}
      <section className="grid gap-8 border-b border-ink py-10 md:grid-cols-[1fr_auto] md:py-14">
        <div>
          <motion.p variants={rise} className="label text-accent">Hiring desk · Shortlist</motion.p>
          <h1 className="mt-4 font-display text-[clamp(2.75rem,8vw,6.25rem)] leading-[0.92] tracking-[-0.02em]">
            <motion.span variants={rise} className="block">Who should you</motion.span>
            <motion.span variants={rise} className="block">
              interview <span className="relative whitespace-nowrap">
                first?
                <motion.svg
                  viewBox="0 0 300 20"
                  preserveAspectRatio="none"
                  className="absolute -bottom-2 left-0 h-3 w-full text-accent"
                  aria-hidden
                >
                  <motion.path
                    d="M3 14 C 60 4, 140 4, 297 10"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="5"
                    strokeLinecap="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ delay: 0.65, duration: 0.7, ease: 'easeInOut' }}
                  />
                </motion.svg>
              </span>
            </motion.span>
          </h1>
          <motion.p variants={rise} className="mt-7 max-w-xl text-lg leading-relaxed text-ink-2">
            Paste the role and drop in the resumes. Every candidate is scored against the job
            and, when a local model is running, given a written assessment.
            <span className="text-ink"> You make the call.</span>
          </motion.p>
        </div>

        {/* How the score is built */}
        <motion.aside variants={rise} className="self-end md:w-64">
          <p className="label mb-2 text-ink-2">How the score is weighed</p>
          <dl className="border-t border-ink">
            {WEIGHTS.map(([label, weight]) => (
              <div key={label} className="flex items-baseline border-b border-rule py-2 text-sm">
                <dt>{label}</dt>
                <span className="leader" aria-hidden />
                <dd className="font-mono text-xs tabular-nums">{weight}%</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-xs leading-relaxed text-ink-3">
            Parts that don't apply to a role are left out, not scored as zero.
          </p>
        </motion.aside>
      </section>

      {/* Form */}
      <div className="grid md:grid-cols-2 md:divide-x md:divide-ink">
        <motion.section variants={rise} className="py-8 md:pr-10">
          <SectionHeading number="01" title="The role" />
          <div className="border border-ink bg-card">
            <label htmlFor="jd" className="sr-only">Job description</label>
            <textarea
              id="jd"
              value={jobDescription}
              onChange={e => setJobDescription(e.target.value)}
              placeholder="Paste the full job description: responsibilities, requirements, nice-to-haves…"
              rows={13}
              disabled={isLoading}
              className="ruled block w-full resize-y bg-transparent px-5 pt-1 text-[15px] text-ink placeholder:text-ink-3 focus:outline-none disabled:opacity-60"
            />
            <div className="flex items-center justify-between border-t border-rule px-5 py-2">
              <span className="label">
                {jdLength < MIN_JD_CHARS ? `${MIN_JD_CHARS - jdLength} more characters needed` : 'Ready'}
              </span>
              <span className="font-mono text-[11px] tabular-nums text-ink-3">{jobDescription.length} chars</span>
            </div>
          </div>
        </motion.section>

        <motion.section variants={rise} className="border-t border-ink py-8 md:border-t-0 md:pl-10">
          <SectionHeading number="02" title="The applicants" />
          <DropZone disabled={isLoading} />
          <FileList disabled={isLoading} />
        </motion.section>
      </div>

      {/* Action bar */}
      <motion.section variants={rise} className="border-t-[3px] border-ink pt-6">
        <div className="grid items-start gap-6 md:grid-cols-[1fr_auto]">
          <div className="min-h-[3rem]">
            <SectionHeading number="03" title="Rank them" className="mb-2" />
            <AnimatePresence mode="wait">
              {error ? (
                <motion.p
                  key="error"
                  role="alert"
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="border-l-[3px] border-accent pl-3 text-sm text-accent"
                >
                  {error}
                </motion.p>
              ) : (
                <motion.p key="hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-ink-2">
                  {files.length > 0
                    ? `${files.length} resume${files.length === 1 ? '' : 's'} ready. The top 10 are returned with scores and notes.`
                    : 'Resumes are processed on this machine. The only outside requests go to GitHub and portfolio links found in them.'}
                </motion.p>
              )}
            </AnimatePresence>
          </div>

          <div className="flex flex-col items-stretch gap-2 md:w-80">
            <motion.button
              onClick={handleSubmit}
              disabled={isLoading}
              whileTap={{ scale: isLoading ? 1 : 0.98 }}
              className="group relative flex items-center justify-between gap-3 overflow-hidden bg-ink px-6 py-4 text-left text-paper transition-colors hover:bg-accent hover:text-accent-ink disabled:cursor-wait disabled:hover:bg-ink disabled:hover:text-paper"
            >
              <span className="font-display text-2xl leading-none">
                {isLoading ? 'Reading…' : 'Rank candidates'}
              </span>
              {isLoading ? (
                <Loader2 size={22} className="animate-spin" />
              ) : (
                <ArrowRight size={22} className="transition-transform group-hover:translate-x-1" />
              )}
            </motion.button>
            {isLoading && (
              <button
                onClick={() => abort.current?.abort()}
                className="label self-end py-1 hover:text-accent transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </div>

        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-6 md:ml-auto md:w-[28rem]"
            >
              <RankProgressBar progress={progress} />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>
    </motion.main>
  )
}

function SectionHeading({ number, title, className = 'mb-4' }: { number: string; title: string; className?: string }) {
  return (
    <h2 className={`flex items-baseline gap-3 ${className}`}>
      <span className="font-mono text-xs text-accent">{number}</span>
      <span className="font-display text-2xl">{title}</span>
    </h2>
  )
}
