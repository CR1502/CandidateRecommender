import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import type { Candidate, RankResponse } from '../types'

// Request state (loading / error / progress) lives in React Query's
// useMutation on the Home page; this store holds the form and the results.
interface AppStore {
  jobDescription: string
  files: File[]
  candidates: Candidate[]
  totalProcessed: number
  totalDurationMs: number
  jobDescriptionSnapshot: string
  showNotRecommended: boolean
  categoryFilter: string | null
  searchQuery: string

  setJobDescription: (v: string) => void
  setFiles: (v: File[]) => void
  setResults: (data: RankResponse) => void
  setShowNotRecommended: (v: boolean) => void
  setCategoryFilter: (v: string | null) => void
  setSearchQuery: (v: string) => void
  reset: () => void
}

const initial = {
  jobDescription: '',
  files: [] as File[],
  candidates: [] as Candidate[],
  totalProcessed: 0,
  totalDurationMs: 0,
  jobDescriptionSnapshot: '',
  showNotRecommended: false,
  categoryFilter: null as string | null,
  searchQuery: '',
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      ...initial,
      setJobDescription: (v) => set({ jobDescription: v }),
      setFiles: (v) => set({ files: v }),
      setResults: (data) => set({
        candidates: data.candidates,
        totalProcessed: data.total_processed,
        totalDurationMs: data.total_duration_ms,
        jobDescriptionSnapshot: data.job_description,
      }),
      setShowNotRecommended: (v) => set({ showNotRecommended: v }),
      setCategoryFilter: (v) => set({ categoryFilter: v }),
      setSearchQuery: (v) => set({ searchQuery: v }),
      reset: () => set(initial),
    }),
    {
      // Results survive a page refresh but not closing the tab (they include
      // candidates' contact details). File objects can't be serialised.
      name: 'candidate-recommender',
      storage: createJSONStorage(() => sessionStorage),
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      partialize: ({ files, ...rest }) => rest,
    },
  ),
)
