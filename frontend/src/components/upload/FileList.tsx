import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'
import { pad2 } from '../../theme'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Ledger of uploaded files. */
export function FileList({ disabled = false }: { disabled?: boolean }) {
  const { files, setFiles } = useAppStore()

  if (files.length === 0) return null

  return (
    <div className="mt-5">
      <div className="flex items-baseline justify-between border-b border-ink pb-1.5">
        <span className="label text-ink-2">Filed</span>
        {!disabled && (
          <button
            onClick={() => setFiles([])}
            className="label hover:text-accent transition-colors"
          >
            Clear all
          </button>
        )}
      </div>
      <ul className="max-h-56 overflow-y-auto">
        <AnimatePresence initial={false}>
          {files.map((file, i) => (
            <motion.li
              key={file.name}
              layout
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.18 }}
              className="flex items-center gap-3 border-b border-rule py-2 text-sm"
            >
              <span className="font-mono text-[11px] text-ink-3 tabular-nums">{pad2(i + 1)}</span>
              <span className="flex-1 truncate">{file.name}</span>
              <span className="font-mono text-[11px] text-ink-3 shrink-0">{formatBytes(file.size)}</span>
              <button
                onClick={() => setFiles(files.filter(f => f.name !== file.name))}
                disabled={disabled}
                className="shrink-0 p-0.5 text-ink-3 hover:text-accent transition-colors disabled:opacity-40"
                aria-label={`Remove ${file.name}`}
              >
                <X size={14} />
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  )
}
