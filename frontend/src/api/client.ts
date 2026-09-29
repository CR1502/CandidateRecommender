import type { HealthResponse, RankProgress, RankResponse } from '../types'

const BASE = '/api'

export class ApiError extends Error {}

/** FastAPI errors carry `detail` as a string or a list of validation errors. */
function detailMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map(d => (typeof d?.msg === 'string' ? d.msg : String(d))).join('; ')
  }
  return fallback
}

async function errorFrom(res: Response): Promise<ApiError> {
  const body = await res.json().catch(() => null)
  return new ApiError(detailMessage(body, `Request failed (${res.status})`))
}

/** Parse a text/event-stream body into (event, data) pairs as they arrive. */
async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<[string, unknown]> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })
    let end: number
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      if (data) yield [event, JSON.parse(data)]
    }
  }
}

/**
 * Rank resumes against a job description, reporting progress as the backend
 * works (LLM assessments can take several seconds per candidate).
 */
export async function rankCandidates(
  jobDescription: string,
  files: File[],
  onProgress: (progress: RankProgress) => void,
  topK = 10,
  signal?: AbortSignal,
): Promise<RankResponse> {
  const form = new FormData()
  form.append('job_description', jobDescription)
  form.append('top_k', String(topK))
  files.forEach(f => form.append('files', f))

  let res: Response
  try {
    res = await fetch(`${BASE}/rank/stream`, { method: 'POST', body: form, signal })
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err
    throw new ApiError('Could not reach the server. Is the backend running?')
  }
  if (!res.ok || !res.body) throw await errorFrom(res)

  for await (const [event, data] of readEvents(res.body)) {
    if (event === 'progress') onProgress(data as RankProgress)
    else if (event === 'result') return data as RankResponse
    else if (event === 'error') throw new ApiError(detailMessage(data, 'Processing failed.'))
  }
  throw new ApiError('The server closed the connection before sending results.')
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw await errorFrom(res)
  return res.json()
}

export function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : 'Something went wrong. Is the backend running?'
}
