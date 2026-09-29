# Candidate Recommender — Architecture & Flow

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend API | FastAPI | Async, typed, auto-generates OpenAPI docs |
| ML Core | sentence-transformers (ranking) + Ollama (assessments) | Both run locally |
| Frontend | React 19 + Vite | Fast dev loop, good ecosystem |
| Charts | SVG/CSS components | Readable at a glance; no WebGL, small bundle |
| Styling | Tailwind CSS | Utility-first styling |
| State | Zustand | Lightweight, no boilerplate vs Redux |
| HTTP client | TanStack Query `useMutation` + `fetch` (streams Server-Sent Events) | Loading/error state built in; fetch can read a streamed POST response |
| API types | Generated from the OpenAPI schema (`openapi-typescript`) | Frontend types can't drift from the backend |

---

## Repository Layout

```
CandidateRecommender/
├── src/candidate_recommender/      ← single Python package (installed by uv)
│   ├── config.py                   ← pydantic-settings Settings + score tiers
│   ├── api/                        ← FastAPI app (export_openapi.py writes the schema)
│   │   ├── main.py                 ← App, lifespan model loading, CORS, SPA static serving
│   │   ├── dependencies.py         ← Hands the models on app.state to routes
│   │   ├── routers/
│   │   │   ├── rank.py             ← POST /api/rank
│   │   │   ├── extract.py          ← POST /api/extract
│   │   │   └── health.py           ← GET  /api/health
│   │   ├── schemas/
│   │   │   └── responses.py        ← Pydantic output models
│   │   └── services/
│   │       └── pipeline.py         ← Orchestrates file → embed → rank → enrich → summarise
│   └── core/                       ← ML + text processing
│       ├── embeddings.py
│       ├── experience.py
│       ├── enricher.py
│       ├── llm.py                  ← Ollama client (structured JSON, availability, cache)
│       ├── redact.py               ← strips personal details before the LLM
│       ├── summarizer.py
│       ├── text_cleaner.py
│       └── file_processor.py
│
├── tests/                          ← pytest suite (models mocked)
├── eval/                           ← labelled ranking dataset + run_eval.py
│
└── frontend/                       ← React + Vite app
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── types.ts                ← Aliases over the generated API types
    │   ├── theme.ts                ← Category / recommendation colours (CSS variables)
    │   ├── index.css               ← Design tokens (light + dark), paper grain
    │   ├── api/
    │   │   ├── client.ts           ← fetch calls; reads the SSE progress stream
    │   │   ├── openapi.json        ← exported from the backend (make gen-api)
    │   │   └── schema.d.ts         ← generated from openapi.json — don't edit
    │   ├── store/
    │   │   └── useAppStore.ts      ← Zustand store
    │   ├── components/
    │   │   ├── layout/             ← Masthead (wordmark + live API/Ollama status)
    │   │   ├── upload/             ← DropZone, FileList, RankProgressBar
    │   │   └── results/            ← CandidateCard, CategoryStrip, ScoreBreakdown, RecommendationBadge
    │   └── pages/
    │       ├── Home.tsx            ← Upload + job description input
    │       └── Results.tsx         ← Ranked candidate display
    ├── index.html
    ├── vite.config.ts
    ├── tailwind.config.js
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
top_k:           int             (optional, 1–50; defaults to TOP_CANDIDATES_COUNT = 10)
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
      "similarity_score": 0.81,          // raw cosine similarity
      "semantic_score": 0.9,             // calibrated 0–1 (used in the composite)
      "skill_coverage_score": 0.86,      // null if the job lists no known skills
      "experience_score": 1.0,           // null if the job states no years
      "category": "Perfect Match",
      "category_emoji": "🌟",
      "category_color": "#00D26A",
      "matching_skills": ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"],
      "fit_summary": "The candidate built payment APIs in FastAPI and PostgreSQL...",
      "strengths": ["8 years building payment APIs", "Production Kubernetes on AWS"],
      "gaps": ["Probe depth with Kafka"],
      "recommendation": "Strong Yes",    // null for template summaries
      "summary_source": "llm",           // "llm" | "template"
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

// 413 — more files than MAX_FILES_PER_UPLOAD (default 20)
{ "detail": "Too many files: 25. Maximum is 20." }

// 415 — unsupported file extension
{ "detail": "Unsupported file type: resume.exe. Allowed: PDF, DOCX, TXT." }

// 500 — internal error (details are logged server-side, not returned)
{ "detail": "Processing failed due to an internal error." }
```

---

### `POST /api/rank/stream`

Same form fields and validation as `/api/rank`, but the response is a stream of Server-Sent Events (`text/event-stream`), because LLM assessments take several seconds per candidate:

```
event: progress
data: {"stage": "assessing", "done": 3, "total": 10}     // stages: extracting, ranking, enriching, assessing

event: result
data: { ...RankResponse... }                               // last event on success

event: error
data: {"detail": "No valid resume text could be extracted..."}   // last event on failure
```

Validation errors (415, 413, 422) are returned as normal HTTP responses before the stream starts. The frontend reads the stream with `fetch`, since `EventSource` only supports GET.

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
 │  useMutation → multipart POST /api/rank/stream
 │  progress events drive the progress bar; Cancel aborts the fetch
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
 │  │    → bge-small embeddings of 250-word    │
 │  │      chunks (batched)                    │
 │  │    → composite score per candidate       │
 │  │      (semantic 60% + skills 30% + exp 10%)│
 │  │    → sorted results                      │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  ┌─────────────────────────────────────────┐
 │  │  For each ranked candidate:              │
 │  │    contact details + keyword skill match │
 │  │    GitHub/portfolio enrichment (parallel)│
 │  │    CandidateSummarizer.batch_assess()    │
 │  │      redact PII → one structured Ollama  │
 │  │      call → summary, strengths, gaps,    │
 │  │      recommendation (template fallback)  │
 │  └─────────────────────────────────────────┘
 │         ↓
 │  Serialise to RankResponse JSON
 ▼
FRONTEND (React)
 │
 │  Zustand store: setResults(result) — persisted to sessionStorage
 │
 │  Navigate to Results page
 ▼
RESULTS PAGE
 │
 ├── Category strip (top)
 │     Horizontal breakdown: Perfect | Ideal | Good | Okay | Not Recommended;
 │     each segment and legend entry is also a filter
 │
 ├── Candidate Cards (main content)
 │     Sorted by rank, grouped by category
 │     Each card shows:
 │       - Name, score, category label, AI recommendation badge
 │       - Matching skills
 │       - Expand → summary, strengths and gaps, ScoreBreakdown,
 │         contact details
 │
 └── Export button → downloads CSV
```

---

## Visualisations

The UI is styled as a recruiter's dossier: warm paper, ink, and one vermilion accent, with light and dark themes that follow the OS setting. Colours are CSS variables in `index.css`, which the Tailwind config maps to class names (`bg-paper`, `text-ink-2`, `text-accent`…). Fonts are Gloock (display), Schibsted Grotesk (text) and Martian Mono (labels and numbers), loaded from Google Fonts. There's no WebGL; three.js was removed.

- **CategoryStrip.** How the shortlist splits across categories. Segments and legend entries filter the list.
- **ScoreBreakdown.** The three composite components as a ledger with bars and weights, showing "n/a" when a component doesn't apply to the job.
- **RecommendationBadge.** The LLM's recommendation drawn as a rubber stamp labelled "AI verdict", and titled as a starting point for a human reviewer. Stamps land in rank order when results load.
- **RankProgressBar.** The pipeline's four stages as a checklist, driven by the SSE progress stream.
- **Masthead status.** Polls `/api/health` every 30s and shows whether the API is up and AI assessments are on.
- **Motion.** Framer Motion, wrapped in `MotionConfig reducedMotion="user"` so the OS "reduce motion" setting is respected.

---

## Frontend State

- **Request state** (pending, error, progress) comes from React Query's `useMutation` on the Home page.
- **Zustand** holds the form (job description, files) and the results and filters. Results persist to `sessionStorage`, so a refresh keeps them but closing the tab clears them, since they include contact details. Files can't be serialised and aren't persisted.

---

## Key Design Decisions

**Why multipart POST instead of JSON?**
Files can't be base64-encoded at 10MB each without significant overhead. Multipart is the correct transport for binary file uploads alongside metadata.

**Why contact extraction happens on the backend, not frontend?**
The regex patterns that extract phone numbers, emails, and LinkedIn handles need to run on the raw extracted text from the PDF/DOCX, before any display-layer transformation. Keeping it server-side also means the frontend never needs to hold the full resume text.

**Why Zustand over Context or Redux?**
This app has one main data event (ranking completes) and a handful of UI filters. Redux is overkill. Context re-renders the whole tree on every update. Zustand gives per-slice subscriptions with no boilerplate.

**Why TanStack Query alongside Zustand?**
TanStack Query handles the request lifecycle (pending, error) for the ranking call. Zustand holds the form and the results, which must outlive the request and survive a refresh. They solve different problems.

**Why redact before the LLM?**
The model doesn't need a name, contact details, or location to judge fit, and seeing them invites bias. The name is taken from the resume's first line, not the filename, so a file like `final_v2.pdf` can't cause ordinary words to be redacted.

**Why doesn't the LLM change the ranking?**
Ranking stays deterministic, fast, and measured by `eval/run_eval.py`. The LLM explains candidates and can disagree; letting it reorder results would make rankings slower, non-deterministic, and dependent on which model is installed.

**Why no 3D?**
This is a reading tool: people scan names, scores and notes. WebGL charts were slower to read than a number and a bar, cost about 870KB of JavaScript, and per-card canvases hit the browser's limit on WebGL contexts. Plain SVG and CSS do the job.

**CORS**
FastAPI will be configured to allow `http://localhost:5173` (Vite dev server) in development. In production, the frontend is built and served as static files from FastAPI itself — no separate CORS needed.

---

## Local Dev Setup

```bash
# Terminal 1 — Backend (models load at startup)
uv sync
uv run uvicorn candidate_recommender.api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm ci
npm run dev          # Vite serves on http://localhost:5173, proxying /api to :8000

# Terminal 3 — LLM (optional, for real summaries)
ollama serve
ollama pull llama3.2
```

Production: `npm run build` outputs `frontend/dist/`, which FastAPI serves as static files (with a fallback to `index.html` for client routes like `/results`). `docker compose up --build` does all of this in one container, with Ollama alongside.
