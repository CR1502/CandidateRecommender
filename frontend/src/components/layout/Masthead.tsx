import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { checkHealth } from '../../api/client'

const TODAY = new Date().toLocaleDateString(undefined, {
  weekday: 'long',
  day: 'numeric',
  month: 'long',
  year: 'numeric',
})

/** Newspaper-style top strip: wordmark, date, and whether AI assessments are on. */
export function Masthead() {
  return (
    <header className="px-4 sm:px-8 pt-5">
      <div className="mx-auto max-w-6xl">
        <div className="flex items-center justify-between gap-4 pb-3">
          <Link to="/" className="font-display text-xl leading-none hover:text-accent transition-colors">
            Candidate Recommender
          </Link>
          <span className="label hidden md:block">{TODAY}</span>
          <ServiceStatus />
        </div>
        {/* Broadsheet double rule */}
        <div className="border-t-[3px] border-ink" />
        <div className="mt-[3px] border-t border-ink" />
      </div>
    </header>
  )
}

function ServiceStatus() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: checkHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  let dot = 'var(--ink-3)'
  let text = 'Checking services…'
  let title = ''
  if (health.isError) {
    dot = 'var(--cat-not)'
    text = 'API offline'
    title = 'Start the backend with `make api`.'
  } else if (health.data?.ollama_available) {
    dot = 'var(--cat-perfect)'
    text = `AI on · ${health.data.ollama_model}`
    title = 'Each candidate gets a written assessment from the local model.'
  } else if (health.data) {
    dot = 'var(--cat-good)'
    text = 'Template summaries'
    title = `Ollama or ${health.data.ollama_model} isn't available, so summaries come from a template.`
  }

  return (
    <span className="label flex items-center gap-2 text-ink-2 whitespace-nowrap" title={title} role="status">
      <span className="relative flex h-2 w-2">
        {health.data?.ollama_available && (
          <span className="absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping" style={{ background: dot }} />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: dot }} />
      </span>
      {text}
    </span>
  )
}
