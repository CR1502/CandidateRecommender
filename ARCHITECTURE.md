# Candidate Recommender — Architecture & Flow

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend API | FastAPI | Async, typed, auto-generates OpenAPI docs |
| ML Core | sentence-transformers + Ollama | Already working, just needs HTTP wrappers |
| Frontend | React 18 + Vite | Fast dev loop, good ecosystem |
| 3D | React Three Fiber (R3F) + Drei | Declarative Three.js in React, minimal boilerplate |
| Styling | Tailwind CSS + shadcn/ui | Utility-first, consistent components |
| State | Zustand | Lightweight, no boilerplate vs Redux |
| HTTP client | TanStack Query + axios | Caching, loading states, error boundaries built in |

---

## Repository Layout

```
CandidateRecommender/
├── backend/                        ← FastAPI app
│   ├── main.py                     ← App entry point, CORS, router registration
│   ├── routers/
│   │   ├── rank.py                 ← POST /api/rank
│   │   ├── extract.py              ← POST /api/extract
│   │   └── health.py               ← GET  /api/health
│   ├── schemas/
│   │   ├── requests.py             ← Pydantic input models
│   │   └── responses.py            ← Pydantic output models
│   ├── services/
│   │   └── pipeline.py             ← Orchestrates file → embed → rank → summarise
│   └── dependencies.py             ← Model singleton injection (loaded once at startup)
│
├── src/                            ← Existing ML core (unchanged)
│   ├── config.py
│   └── core/
│       ├── embeddings.py
│       ├── summarizer.py
│       ├── text_cleaner.py
│       └── file_processor.py
│
└── frontend/                       ← React + Vite app
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── api/
    │   │   └── client.ts           ← axios instance + typed API calls
    │   ├── store/
    │   │   └── useAppStore.ts      ← Zustand store
    │   ├── components/
    │   │   ├── three/              ← All 3D components
    │   │   │   ├── Scene.tsx       ← R3F Canvas wrapper
    │   │   │   ├── ParticleField.tsx
    │   │   │   ├── ScoreOrb.tsx
    │   │   │   └── RadarChart3D.tsx
    │   │   ├── upload/
    │   │   │   ├── DropZone.tsx
    │   │   │   └── FileList.tsx
    │   │   ├── results/
    │   │   │   ├── CandidateCard.tsx
    │   │   │   ├── CandidateDetail.tsx
    │   │   │   └── RankingTable.tsx
    │   │   └── ui/                 ← shadcn/ui components
    │   └── pages/
    │       ├── Home.tsx            ← Upload + job description input
    │       └── Results.tsx         ← Ranked candidate display
    ├── index.html
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── package.json
```

---

## API Contract

### `POST /api/rank`

Accepts a multipart form with the job description and resume files.

**Request**
```
Content-Type: multipart/form-data

job_description: string         (required, min 50 chars)
files:           File[]          (required, 1–20 files, pdf/docx/txt, max 10MB each)
top_k:           int = 10        (optional, max results to return)
```

**Response `200`**
```json
{
  "total_processed": 8,
  "total_duration_ms": 4320,
  "job_description": "...",
  "candidates": [
    {
      "rank": 1,
      "candidate_name": "Alice Johnson",
      "filename": "alice_johnson.pdf",
      "percentage_score": 87.3,
      "composite_score": 0.873,
      "similarity_score": 0.86,
      "skill_coverage_score": 0.86,
      "experience_score": 1.0,
      "category": "Perfect Match",
      "category_emoji": "🌟",
      "category_color": "#00D26A",
      "matching_skills": ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"],
      "fit_summary": "This candidate is an excellent fit...",
      "contact": {
        "email": "alice@example.com",
        "phone": "+1 (555) 234-5678",
        "linkedin": "linkedin.com/in/alicejohnson",
        "github": "github.com/alice-j",
        "location": "San Francisco, CA",
        "website": null
      }
    }
  ]
}
```

**Error responses**
```json
// 422 — validation error (empty JD, no files, bad file type)
{ "detail": "Job description must be at least 50 characters." }

// 413 — file too large
{ "detail": "alice.pdf exceeds the 10MB limit." }

// 500 — model/processing error
{ "detail": "Embedding model failed: ..." }
```

---

### `POST /api/extract`

Extract contact info and skills from a single resume without running a full ranking.
Useful for the "quick scan" use case.

**Request**
```
Content-Type: multipart/form-data
file: File
```

**Response `200`**
```json
{
  "candidate_name": "Alice Johnson",
  "skills": ["Python", "FastAPI", "Docker"],
  "contact": { "email": "...", "phone": "...", ... }
}
```

---

### `GET /api/health`

**Response `200`**
```json
{
  "status": "ok",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "embedding_device": "cpu",
  "ollama_available": true,
  "ollama_model": "llama3.2",
  "summary_mode": "ollama"   // or "template"
}
```

---

## Data Flow

```
USER
 │
 │  1. Pastes job description
 │  2. Drops resume files (PDF/DOCX/TXT)
 │  3. Clicks "Find Candidates"
 ▼
FRONTEND (React)
 │
 │  Zustand store sets: { status: 'uploading', files: [...] }
 │  Multipart POST → /api/rank
 │  TanStack Query manages loading / error / success states
 ▼
BACKEND (FastAPI)
 │
 │  router/rank.py receives request
 │  ↓
 │  services/pipeline.py orchestrates:
 │
 │  ┌─────────────────────────────────────────┐
 │  │  For each file:                          │
 │  │    FileProcessor.process_file()          │
 │  │      → extracted text + candidate name   │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  ┌─────────────────────────────────────────┐
 │  │  TextCleaner.prepare_for_embedding()     │
 │  │  on job description + all resume texts   │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  ┌─────────────────────────────────────────┐
 │  │  EmbeddingEngine.rank_candidates()       │
 │  │    → deduplication                       │
 │  │    → bge-small embeddings (batched)      │
 │  │    → composite score per candidate       │
 │  │      (semantic 60% + skills 30% + exp 10%)│
 │  │    → sorted results                      │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  ┌─────────────────────────────────────────┐
 │  │  For each ranked candidate:              │
 │  │    EmbeddingEngine.find_matching_skills()│
 │  │    TextCleaner.extract_contact_details() │
 │  │    CandidateSummarizer.generate_fit_summary() │
 │  │      (Ollama if available, else template) │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  Serialise to RankResponse JSON
 ▼
FRONTEND (React)
 │
 │  TanStack Query updates cache with response
 │  Zustand store: { status: 'success', candidates: [...] }
 │
 │  Navigate to Results page
 ▼
RESULTS PAGE
 │
 ├── 3D Scene (background)
 │     ParticleField — animated particle cloud, density reacts to top score
 │
 ├── Score Distribution Bar (top)
 │     Horizontal breakdown: Perfect | Ideal | Good | Okay | Not Recommended
 │
 ├── Candidate Cards (main content)
 │     Sorted by rank, grouped by category
 │     Each card shows:
 │       - Name, score badge, category label
 │       - Matching skills as chips
 │       - Fit summary
 │       - Expand → contact info + ScoreOrb 3D widget + RadarChart3D
 │
 └── Export button → downloads CSV
```

---

## 3D Elements

Three distinct 3D moments — each purposeful, not decorative noise:

### 1. ParticleField (Home page background)
- Floating particle cloud built with `<Points>` from Drei
- Particles slowly drift and rotate
- On file upload, particles pulse outward (scale animation triggered by file count)
- Keeps the landing page from feeling static without distracting from the form

### 2. ScoreOrb (inside expanded CandidateCard)
- Glowing sphere whose colour maps to the candidate's score
  - ≥85% → green `#00D26A`
  - 70–85% → teal `#4CAF50`
  - 50–70% → amber `#FFA726`
  - <50% → red `#F44336`
- Outer shell has a subtle wireframe that spins
- Score number floats as HTML overlay (via `<Html>` from Drei) so it's selectable/readable
- Renders inside a small fixed-size canvas per card — not a full-screen scene

### 3. RadarChart3D (inside expanded CandidateCard)
- 3-axis radar showing the three scoring components:
  - Semantic match (0–1)
  - Skill coverage (0–1)
  - Experience alignment (0–1)
- Built with custom geometry — three axes drawn as lines, filled polygon as a mesh
- Hovering an axis shows a tooltip with the raw value
- Makes composite scoring transparent — the user can see *why* someone scored what they did

---

## Frontend State Shape (Zustand)

```ts
interface AppState {
  // Upload stage
  jobDescription: string
  files: File[]
  status: 'idle' | 'uploading' | 'success' | 'error'
  error: string | null

  // Results
  candidates: Candidate[]
  jobDescriptionSnapshot: string   // what was submitted (for display)
  expandedCandidateId: string | null

  // Filters / UI
  showNotRecommended: boolean
  categoryFilter: string | null     // null = all
  searchQuery: string

  // Actions
  setJobDescription: (jd: string) => void
  setFiles: (files: File[]) => void
  setResults: (data: RankResponse) => void
  setExpanded: (name: string | null) => void
  reset: () => void
}
```

---

## Key Design Decisions

**Why multipart POST instead of JSON?**
Files can't be base64-encoded at 10MB each without significant overhead. Multipart is the correct transport for binary file uploads alongside metadata.

**Why contact extraction happens on the backend, not frontend?**
The regex patterns that extract phone numbers, emails, and LinkedIn handles need to run on the raw extracted text from the PDF/DOCX, before any display-layer transformation. Keeping it server-side also means the frontend never needs to hold the full resume text.

**Why Zustand over Context or Redux?**
This app has one main data event (ranking completes) and a handful of UI filters. Redux is overkill. Context re-renders the whole tree on every update. Zustand gives per-slice subscriptions with no boilerplate.

**Why TanStack Query alongside Zustand?**
TanStack Query handles the async lifecycle (loading, error, retries, cache invalidation) for API calls. Zustand holds the derived UI state (filters, expanded card, etc). Mixing them is intentional — they solve different problems.

**Why R3F (React Three Fiber) over plain Three.js?**
Three.js is imperative — you manage the render loop, refs, and cleanup manually. R3F wraps it in React's component model, so 3D elements compose naturally with the rest of the UI. Drei adds ready-made helpers (`<OrbitControls>`, `<Html>`, `<Points>`) that would take hours to write from scratch.

**CORS**
FastAPI will be configured to allow `http://localhost:5173` (Vite dev server) in development. In production, the frontend is built and served as static files from FastAPI itself — no separate CORS needed.

---

## Local Dev Setup (once built)

```bash
# Terminal 1 — Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev          # Vite serves on http://localhost:5173

# Terminal 3 — LLM (optional, for real summaries)
ollama serve
ollama pull llama3.2
```

Production: `npm run build` outputs `frontend/dist/` which FastAPI serves as static files via `StaticFiles`.
