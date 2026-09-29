import { useCallback, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useAppStore } from '../../store/useAppStore'

const ACCEPTED = ['.pdf', '.docx', '.txt']
const ACCEPT_MIME = 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain'

function filterFiles(list: FileList | null): File[] {
  if (!list) return []
  return Array.from(list).filter(f =>
    ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext))
  )
}

/** The "in-tray": drag resumes in, or click to browse. */
export function DropZone({ disabled = false }: { disabled?: boolean }) {
  const { files, setFiles } = useAppStore()
  const [dragging, setDragging] = useState(false)

  const addFiles = useCallback((incoming: File[]) => {
    const existingNames = new Set(files.map(f => f.name))
    const fresh = incoming.filter(f => !existingNames.has(f.name))
    setFiles([...files, ...fresh])
  }, [files, setFiles])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    if (!disabled) addFiles(filterFiles(e.dataTransfer.files))
  }

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(filterFiles(e.target.files))
    e.target.value = ''
  }

  return (
    <label
      htmlFor="resume-upload"
      className={`group relative block cursor-pointer border border-dashed px-6 py-9 text-center transition-colors duration-200
        focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent
        ${dragging ? 'border-accent bg-accent-wash' : 'border-ink-3 hover:border-ink'}
        ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        id="resume-upload"
        type="file"
        multiple
        accept={ACCEPT_MIME}
        className="sr-only"
        onChange={onInput}
        disabled={disabled}
      />

      {/* Stacked-sheets glyph: the top sheet lifts while dragging */}
      <div className="relative mx-auto mb-4 h-14 w-11" aria-hidden>
        <span className="absolute inset-0 translate-x-1.5 translate-y-1.5 border border-ink-3 bg-card" />
        <motion.span
          className="absolute inset-0 border border-ink bg-card"
          animate={dragging ? { y: -8, rotate: -6 } : { y: 0, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 320, damping: 18 }}
        >
          <span className="absolute left-2 right-2 top-3 border-t border-ink-3" />
          <span className="absolute left-2 right-4 top-5 border-t border-ink-3" />
          <span className="absolute left-2 right-3 top-7 border-t border-ink-3" />
        </motion.span>
      </div>

      <p className="font-display text-xl">
        {dragging ? 'Release to file them' : 'Drop resumes here'}
      </p>
      <p className="mt-1 text-sm text-ink-2">
        or <span className="underline decoration-accent decoration-2 underline-offset-4 group-hover:text-accent">browse your files</span>
      </p>
      <p className="label mt-4">PDF · DOCX · TXT — up to 10 MB each</p>

      <AnimatePresence>
        {files.length > 0 && (
          <motion.span
            key="count"
            initial={{ scale: 1.8, opacity: 0, rotate: -18 }}
            animate={{ scale: 1, opacity: 1, rotate: -8 }}
            exit={{ scale: 0.6, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 420, damping: 16 }}
            className="stamp-ink absolute -right-3 -top-4 flex items-baseline gap-1.5 border-[2.5px] border-accent bg-card px-2 py-1 text-accent"
          >
            <span className="font-display text-xl leading-none">{files.length}</span>
            <span className="font-mono text-[10px] uppercase leading-none tracking-[0.08em]">filed</span>
          </motion.span>
        )}
      </AnimatePresence>
    </label>
  )
}
