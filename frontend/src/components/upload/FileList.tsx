import { AnimatePresence, motion } from 'framer-motion'
import { FileText, X } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function FileList() {
  const { files, setFiles } = useAppStore()

  const remove = (name: string) => {
    setFiles(files.filter(f => f.name !== name))
  }

  if (files.length === 0) return null

  return (
    <ul className="mt-3 space-y-2 max-h-52 overflow-y-auto pr-1">
      <AnimatePresence initial={false}>
        {files.map(file => (
          <motion.li
            key={file.name}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8, height: 0, marginTop: 0 }}
            transition={{ duration: 0.18 }}
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm"
            style={{ background: '#1a1a2e', border: '1px solid #1e1e2e' }}
          >
            <FileText size={14} className="text-indigo-400 shrink-0" />
            <span className="flex-1 truncate text-slate-300 text-xs">{file.name}</span>
            <span className="text-slate-500 text-xs shrink-0">{formatBytes(file.size)}</span>
            <button
              onClick={() => remove(file.name)}
              className="shrink-0 text-slate-600 hover:text-red-400 transition-colors"
              aria-label={`Remove ${file.name}`}
            >
              <X size={13} />
            </button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
