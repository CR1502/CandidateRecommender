import { useNavigate } from 'react-router-dom'
import { Canvas } from '@react-three/fiber'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, AlertCircle, Search } from 'lucide-react'
import { useAppStore } from '../store/useAppStore'
import { rankCandidates } from '../api/client'
import { DropZone } from '../components/upload/DropZone'
import { FileList } from '../components/upload/FileList'
import { ParticleField } from '../components/three/ParticleField'

export default function Home() {
  const navigate = useNavigate()
  const {
    jobDescription, setJobDescription,
    files, status, error,
    setStatus, setError, setResults,
  } = useAppStore()

  const isLoading = status === 'loading'

  const handleSubmit = async () => {
    if (!jobDescription.trim() || jobDescription.trim().length < 50) {
      setError('Job description must be at least 50 characters.')
      return
    }
    if (files.length === 0) {
      setError('Please upload at least one resume file.')
      return
    }

    setError(null)
    setStatus('loading')

    try {
      const result = await rankCandidates(jobDescription, files)
      setResults(result)
      navigate('/results')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Something went wrong. Is the backend running?'
      setError(msg)
      setStatus('error')
    }
  }

  return (
    <div className="relative min-h-screen">
      {/* Background 3D canvas */}
      <div className="fixed inset-0" style={{ pointerEvents: 'none', zIndex: 0 }}>
        <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
          <ParticleField fileCount={files.length} />
        </Canvas>
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="text-center pt-14 pb-6 px-6">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1
              className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2"
              style={{
                background: 'linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #8b5cf6 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Candidate Recommender
            </h1>
            <p className="text-slate-400 text-base max-w-md mx-auto">
              Upload resumes and a job description — AI ranks candidates by composite fit score.
            </p>
          </motion.div>
        </header>

        {/* Main form */}
        <main className="flex-1 px-6 pb-16 max-w-5xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="grid md:grid-cols-2 gap-6"
          >
            {/* Left: Job description */}
            <div className="flex flex-col gap-3">
              <label className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Search size={15} className="text-indigo-400" />
                Job Description
              </label>
              <textarea
                value={jobDescription}
                onChange={e => setJobDescription(e.target.value)}
                placeholder="Paste the full job description here… (min 50 characters)"
                rows={14}
                className="flex-1 w-full rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
                style={{ background: '#12121a', border: '1px solid #1e1e2e' }}
                disabled={isLoading}
              />
              <span className="text-xs text-slate-600 text-right">
                {jobDescription.length} chars
              </span>
            </div>

            {/* Right: Upload + action */}
            <div className="flex flex-col gap-4">
              <div>
                <label className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
                  Resumes
                  <span className="text-xs font-normal text-slate-500">(PDF, DOCX, TXT)</span>
                </label>
                <DropZone />
                <FileList />
              </div>

              <div className="mt-auto space-y-3">
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="flex items-start gap-2 text-sm text-red-400 rounded-lg px-3 py-2.5"
                      style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
                    >
                      <AlertCircle size={15} className="mt-0.5 shrink-0" />
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                <motion.button
                  onClick={handleSubmit}
                  disabled={isLoading}
                  whileHover={{ scale: isLoading ? 1 : 1.02 }}
                  whileTap={{ scale: isLoading ? 1 : 0.98 }}
                  className="w-full py-3.5 rounded-xl font-semibold text-white flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
                  style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
                >
                  {isLoading ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      Analysing candidates…
                    </>
                  ) : (
                    'Find Best Candidates'
                  )}
                </motion.button>
              </div>
            </div>
          </motion.div>
        </main>
      </div>
    </div>
  )
}
