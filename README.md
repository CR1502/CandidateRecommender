# 🎯 AI-Powered Candidate Recommendation Engine

An intelligent resume screening system that ranks candidates against a job description using semantic embeddings, skill coverage, and experience signals, then writes an evidence-based fit assessment for each one. A FastAPI backend serves a React frontend, and the AI models run locally.

## 📖 Table of Contents
- [My Approach](#my-approach)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Limitations](#limitations)

## My Approach

I built this system with three principles in mind:

1. **Semantic understanding > keyword matching.** Traditional ATS systems rely on exact keyword matches, so a candidate who writes "built REST APIs" can be rejected for a job asking for "RESTful services". This system uses transformer embeddings to compare meaning, and combines that with explicit skill and experience checks so the score stays explainable.

2. **Local by default.** The embedding model and the LLM (via [Ollama](https://ollama.com)) run on your machine; resume text is not sent to a hosted AI service. The one exception is profile enrichment, which fetches the public GitHub profile and portfolio pages linked from a resume (see [Limitations](#limitations)).

3. **Actionable output.** Beyond a score, each candidate gets matching skills, extracted contact details, and a written assessment with a hiring recommendation.

## How It Works

```
Resumes (PDF/DOCX/TXT) ─┐
                        ├─► text extraction ─► cleaning ─► embeddings ─► composite score ─► rank
Job description ────────┘                                                        │
                                              contact details, matching skills ◄─┤
                                         GitHub / portfolio enrichment (public) ◄─┤
                                            fit summary (Ollama, or template) ◄──┘
```

**Composite score**, per candidate:

| Component | Weight | What it measures |
|---|---|---|
| Semantic similarity | 60% | Cosine similarity of [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings (the job description gets the BGE query prefix) |
| Skill coverage | 30% | Fraction of the job's skills found in the resume, from a curated skill registry |
| Experience | 10% | Stated years of experience vs. the job's requirement (ranges like "3–5 years" use the minimum) |

Duplicate resumes (identical text) are removed before ranking. Candidates are then placed in tiers:

| Tier | Composite score |
|---|---|
| 🌟 Perfect Match | ≥ 85% |
| ⭐ Ideal Candidate | 70–85% |
| ✅ Good Candidate | 50–70% |
| 👍 Okay Candidate | 25–50% |
| ❌ Not Recommended | < 25% |

**Fit summaries** come from a local Ollama model (default `llama3.2`) when it's running, and otherwise from a deterministic template built from the extracted signals.

## Quick Start

### Option A: Docker (app + Ollama)

```bash
docker compose up --build
```

Open http://localhost:8000. The first run downloads the Ollama model (about 2GB for `llama3.2`), and the app starts once that finishes. Use `OLLAMA_MODEL=mistral docker compose up` to choose a different model.

### Option B: Local development

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22+, and optionally [Ollama](https://ollama.com) for LLM summaries.

```bash
# Install dependencies (uv installs a matching Python automatically)
uv sync
cd frontend && npm ci && cd ..

# Terminal 1 — API on http://localhost:8000 (docs at /api/docs)
uv run uvicorn candidate_recommender.api.main:app --reload --port 8000

# Terminal 2 — frontend on http://localhost:5173 (proxies /api to :8000)
cd frontend && npm run dev

# Optional — local LLM for real summaries
ollama pull llama3.2
```

With `make` installed, these are `make install`, `make api`, and `make web`.

The embedding model (~130MB) downloads on the first start. To serve the built frontend from FastAPI on a single port instead, run `npm run build` in `frontend/`, then start the API.

Sample resumes for testing: `uv run python data/generate_sample_resumes.py` writes them to `data/sample_resumes/`.

## Configuration

All settings are environment variables (or a `.env` file; copy [`.env.example`](.env.example)). They're defined and validated in [`config.py`](src/candidate_recommender/config.py).

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model; `BAAI/bge-base-en-v1.5` is more accurate but larger |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Model used for summaries and LLM skill extraction |
| `OLLAMA_TIMEOUT` | `60` | Seconds per Ollama request |
| `SCORING_WEIGHTS` | `{"semantic": 0.6, "skill_coverage": 0.3, "experience": 0.1}` | JSON; must sum to 1.0 |
| `MAX_FILE_SIZE_MB` | `10` | Per-file upload limit |
| `MAX_FILES_PER_UPLOAD` | `20` | Files per ranking request |
| `TOP_CANDIDATES_COUNT` | `10` | Default number of results |
| `LOG_LEVEL` | `INFO` | Log verbosity |

## Project Structure

```
CandidateRecommender/
├── src/candidate_recommender/
│   ├── config.py              # Settings (pydantic-settings), score tiers
│   ├── api/                   # FastAPI app
│   │   ├── main.py            # App, lifespan (model loading), static frontend
│   │   ├── dependencies.py    # Injects the loaded models into routes
│   │   ├── routers/           # /api/rank, /api/extract, /api/health
│   │   ├── schemas/           # Pydantic response models
│   │   └── services/          # pipeline.py — orchestrates the ranking flow
│   └── core/                  # ML and text processing
│       ├── embeddings.py      # Embeddings, composite scoring, ranking
│       ├── text_cleaner.py    # Cleaning, skill registry, contact extraction
│       ├── file_processor.py  # PDF / DOCX / TXT extraction
│       ├── summarizer.py      # Ollama + template fit summaries
│       └── enricher.py        # GitHub + portfolio enrichment (SSRF-guarded)
├── frontend/                  # React + Vite + TypeScript UI
├── tests/                     # pytest suite
├── data/                      # Sample resume generator
├── pyproject.toml / uv.lock   # Python dependencies (uv)
├── Dockerfile                 # Builds frontend, then Python runtime
└── docker-compose.yml         # App + Ollama
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the API contract and data flow.

## Development

```bash
uv run pytest -q          # tests
uv run ruff check         # lint
uv run ruff format        # format
cd frontend && npm run lint && npm run build
uv run pre-commit install # optional: lint + format on every commit
```

CI (GitHub Actions) runs backend lint and tests, frontend lint and build, and a Docker image build on every push and pull request.

## Limitations

- **English, text-based resumes only.** Scanned PDFs have no extractable text (no OCR yet).
- **The embedding model reads roughly the first 512 tokens** (about one page) of each resume.
- **Skill scoring uses a curated registry**, so skills outside it don't affect the score. With Ollama running, the displayed matching skills also include LLM-extracted ones.
- **Enrichment makes outbound requests.** It fetches public GitHub profiles and portfolio pages linked in resumes. Only public addresses are allowed (private, loopback, and cloud-metadata IPs are blocked), and LinkedIn and other social sites are skipped.
- **Screening support, not a decision-maker.** Scores and summaries are aids for a human reviewer; automated hiring decisions carry legal and fairness obligations in many jurisdictions.
