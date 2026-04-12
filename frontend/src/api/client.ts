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

export async function checkHealth() {
  const { data } = await http.get('/health')
  return data
}
