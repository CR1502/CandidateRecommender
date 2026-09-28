import axios from 'axios'
import type { RankResponse } from '../types'

const http = axios.create({ baseURL: '/api' })

export async function rankCandidates(
  jobDescription: string,
  files: File[],
  topK = 10,
): Promise<RankResponse> {
  const form = new FormData()
  form.append('job_description', jobDescription)
  form.append('top_k', String(topK))
  files.forEach(f => form.append('files', f))
  const { data } = await http.post<RankResponse>('/rank', form)
  return data
}

/** Prefer the backend's `detail` (string, or FastAPI's validation-error list) over axios' generic message. */
export function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map(d => (typeof d?.msg === 'string' ? d.msg : String(d))).join('; ')
    }
    if (!err.response) return 'Could not reach the server. Is the backend running?'
  }
  return err instanceof Error ? err.message : 'Something went wrong. Is the backend running?'
}

export async function checkHealth() {
  const { data } = await http.get('/health')
  return data
}
