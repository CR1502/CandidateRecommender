import { create } from 'zustand'
import type { Candidate, RankResponse, AppStatus } from '../types'

interface AppStore {
  jobDescription: string
  files: File[]
  status: AppStatus
  error: string | null
  candidates: Candidate[]
  totalProcessed: number
  totalDurationMs: number
  jobDescriptionSnapshot: string
  expandedCandidateId: string | null
  showNotRecommended: boolean
  categoryFilter: string | null
  searchQuery: string

  setJobDescription: (v: string) => void
  setFiles: (v: File[]) => void
  setStatus: (v: AppStatus) => void
  setError: (v: string | null) => void
  setResults: (data: RankResponse) => void
  setExpanded: (id: string | null) => void
  setShowNotRecommended: (v: boolean) => void
  setCategoryFilter: (v: string | null) => void
  setSearchQuery: (v: string) => void
  reset: () => void
}

export const useAppStore = create<AppStore>((set) => ({
  jobDescription: '',
  files: [],
  status: 'idle',
  error: null,
  candidates: [],
  totalProcessed: 0,
  totalDurationMs: 0,
  jobDescriptionSnapshot: '',
  expandedCandidateId: null,
  showNotRecommended: false,
  categoryFilter: null,
  searchQuery: '',

  setJobDescription: (v) => set({ jobDescription: v }),
  setFiles: (v) => set({ files: v }),
  setStatus: (v) => set({ status: v }),
  setError: (v) => set({ error: v }),
  setResults: (data) => set({
    candidates: data.candidates,
    totalProcessed: data.total_processed,
    totalDurationMs: data.total_duration_ms,
    jobDescriptionSnapshot: data.job_description,
    status: 'success',
    error: null,
  }),
  setExpanded: (id) => set({ expandedCandidateId: id }),
  setShowNotRecommended: (v) => set({ showNotRecommended: v }),
  setCategoryFilter: (v) => set({ categoryFilter: v }),
  setSearchQuery: (v) => set({ searchQuery: v }),
  reset: () => set({
    jobDescription: '',
    files: [],
    status: 'idle',
    error: null,
    candidates: [],
    totalProcessed: 0,
    totalDurationMs: 0,
    jobDescriptionSnapshot: '',
    expandedCandidateId: null,
    categoryFilter: null,
    searchQuery: '',
  }),
}))
