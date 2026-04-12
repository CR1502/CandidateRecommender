import { useCallback, useState } from 'react'
import { motion } from 'framer-motion'
import { UploadCloud } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'

const ACCEPTED = ['.pdf', '.docx', '.txt']
const ACCEPT_MIME = 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain'

function filterFiles(list: FileList | null): File[] {
  if (!list) return []
  return Array.from(list).filter(f =>
    ACCEPTED.some(ext => f.name.toLowerCase().endsWith(ext))
  )
}

export function DropZone() {
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
    addFiles(filterFiles(e.dataTransfer.files))
  }

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(filterFiles(e.target.files))
    e.target.value = ''
  }

  return (
    <motion.label
      htmlFor="resume-upload"
      className="relative flex flex-col items-center justify-center w-full cursor-pointer rounded-xl border-2 border-dashed transition-colors duration-200 p-8"
      animate={{
        borderColor: dragging ? '#6366f1' : files.length > 0 ? '#4f46e5' : '#2d2d44',
        backgroundColor: dragging ? 'rgba(99,102,241,0.08)' : 'rgba(255,255,255,0.02)',
      }}
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
      />

      <motion.div
        animate={{ scale: dragging ? 1.1 : 1 }}
        transition={{ type: 'spring', stiffness: 300 }}
      >
        <UploadCloud
          className="mb-3 mx-auto"
          size={36}
          color={dragging ? '#818cf8' : files.length > 0 ? '#6366f1' : '#475569'}
        />
      </motion.div>

      <p className="text-sm font-medium text-slate-300">
        {dragging ? 'Drop resumes here' : 'Drag & drop resumes'}
      </p>
      <p className="text-xs text-slate-500 mt-1">
        PDF, DOCX, TXT · up to 10 MB each
      </p>

      {files.length > 0 && (
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="absolute -top-2.5 -right-2.5 flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold text-white"
          style={{ background: '#6366f1' }}
        >
          {files.length}
        </motion.span>
      )}
    </motion.label>
  )
}
