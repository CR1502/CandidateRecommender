// API types are generated from the backend's OpenAPI schema — don't edit them
// by hand. Regenerate after changing a response model:
//   uv run python -m candidate_recommender.api.export_openapi frontend/src/api/openapi.json
//   cd frontend && npm run gen:api
import type { components } from './api/schema'

type Schemas = components['schemas']

export type Candidate = Schemas['CandidateResult']
export type ContactInfo = Schemas['ContactInfo']
export type RankResponse = Schemas['RankResponse']
export type HealthResponse = Schemas['HealthResponse']
export type Recommendation = NonNullable<Candidate['recommendation']>

// Server-Sent Events from POST /api/rank/stream (not described by OpenAPI).
export type PipelineStage = 'extracting' | 'ranking' | 'enriching' | 'assessing'

export interface RankProgress {
  stage: PipelineStage
  done: number
  total: number
}
