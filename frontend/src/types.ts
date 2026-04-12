export interface ContactInfo {
  email: string | null
  phone: string | null
  linkedin: string | null
  github: string | null
  location: string | null
  website: string | null
}

export interface Candidate {
  rank: number
  candidate_name: string
  filename: string
  percentage_score: number
  composite_score: number
  similarity_score: number
  skill_coverage_score: number
  experience_score: number
  category: string
  category_emoji: string
  category_color: string
  matching_skills: string[]
  fit_summary: string
  contact: ContactInfo
}

export interface RankResponse {
  total_processed: number
  total_duration_ms: number
  job_description: string
  candidates: Candidate[]
}

export type AppStatus = 'idle' | 'loading' | 'success' | 'error'
