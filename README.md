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
                        assessment: summary, strengths, gaps,  ◄──┘
                        recommendation (local LLM, or template)
```

**Composite score**, per candidate:

| Component | Weight | What it measures |
|---|---|---|
| Semantic similarity | 60% | Cosine similarity of [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings, rescaled from the model's working range (0.45–0.85) onto 0–1. Resumes are split into overlapping 250-word chunks so the whole resume is read, not just the first ~512 tokens |
| Skill coverage | 30% | Weighted share of the job's skills the resume demonstrates. Nice-to-have skills count half; skills that appear only in a skills list (never in the work history) earn half credit |
| Experience | 10% | Years of experience (stated, or computed from employment date ranges) vs. the job's requirement, scaled by relevance so that years in an unrelated field don't add points |

If a job lists no recognisable skills or states no years of experience, that component is dropped and the remaining weights renormalised, rather than scored as a "neutral" 0.5. Duplicate resumes (identical text) are removed before ranking. Candidates are then placed in tiers:

| Tier | Composite score |
|---|---|
| 🌟 Perfect Match | ≥ 85% |
| ⭐ Ideal Candidate | 70–85% |
| ✅ Good Candidate | 50–70% |
| 👍 Okay Candidate | 25–50% |
| ❌ Not Recommended | < 25% |

These choices were measured against a labelled evaluation set. See [eval/README.md](eval/README.md) for the results, including the models and rerankers that were tried and not adopted.

**Assessments** come from a local LLM through [Ollama](https://ollama.com), default `gemma4:12b`, when it's running. Each candidate gets one structured call that returns a 2–3 sentence summary, up to three strengths and three gaps to probe, and a recommendation (Strong Yes / Yes / Maybe / No). Before the resume reaches the model:

- The candidate's name, email, phone, links, and location are **redacted**, to reduce bias and keep personal data out of prompts.
- All supplied text is wrapped in tags the model is told to treat as data, so a resume saying "ignore your instructions" can't redirect it.
- Skills the model reports are kept only if they actually appear in the resume.

Without Ollama, a deterministic template summary is used instead. Ollama availability is re-checked every 30 seconds, so starting it later needs no restart. Assessments take roughly 10–30 seconds per candidate on a laptop, so the UI shows live progress (`POST /api/rank/stream`).

## Quick Start

### Option A: Docker (app + Ollama)

```bash
docker compose up --build
```

Open http://localhost:8000. The first run downloads the Ollama model (about 7.6GB for `gemma4:12b`), and the app starts once that finishes. Use `OLLAMA_MODEL=qwen3:8b docker compose up` to choose a different model.

On macOS, Docker can't use the Apple GPU, so an LLM in Docker is very slow. Run Ollama natively instead (see Option B) and start only the app: `OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up app`.

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

# Optional — local LLM for AI assessments (one-time ~7.6GB download)
brew install ollama && brew services start ollama   # macOS; see ollama.com for others
ollama pull gemma4:12b
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
| `OLLAMA_MODEL` | `gemma4:12b` | Any Ollama model, e.g. `qwen3:8b`, or a GGUF straight from Hugging Face: `hf.co/<org>/<repo>:<quant>` |
| `OLLAMA_TIMEOUT` | `180` | Seconds per Ollama request |
| `LLM_NUM_CTX` | `6144` | LLM context window in tokens (Ollama's default of 4096 can cut off long resumes) |
| `LLM_CONCURRENCY` | `1` | Candidates assessed at once. On one GPU, 2 was only ~16% faster and doubles context memory |
| `REDACT_PII` | `true` | Hide names, contact details, links, and location from the LLM |
| `SCORING_WEIGHTS` | `{"semantic": 0.6, "skill_coverage": 0.3, "experience": 0.1}` | JSON; must sum to 1.0 |
| `SEMANTIC_FLOOR` / `SEMANTIC_CEILING` | `0.45` / `0.85` | Cosine range rescaled onto 0–1; re-tune with `eval/` if you change models |
| `CHUNK_WORDS` | `250` | Resume chunk size for embedding (`0` disables chunking) |
| `CHUNK_AGGREGATION` | `max_mean` | How chunk scores combine: `max`, `mean`, or `max_mean` |
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
│       ├── embeddings.py      # Embeddings, chunking, composite scoring, ranking
│       ├── experience.py      # Years of experience from employment dates
│       ├── text_cleaner.py    # Cleaning, skill registry, contact extraction
│       ├── file_processor.py  # PDF / DOCX / TXT extraction
│       ├── summarizer.py      # LLM assessments (+ template fallback)
│       ├── llm.py             # Ollama client: structured JSON, availability, cache
│       ├── redact.py          # Removes personal details before text reaches the LLM
│       └── enricher.py        # GitHub + portfolio enrichment (SSRF-guarded)
├── frontend/                  # React + Vite + TypeScript UI
├── tests/                     # pytest suite
├── eval/                      # Labelled ranking dataset + run_eval.py
├── data/                      # Sample resume generator
├── pyproject.toml / uv.lock   # Python dependencies (uv)
├── Dockerfile                 # Builds frontend, then Python runtime
└── docker-compose.yml         # App + Ollama
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the API contract and data flow.

## Development

```bash
uv run python eval/run_eval.py   # ranking quality on the labelled eval set
uv run pytest -q          # tests
uv run ruff check         # lint
uv run ruff format        # format
cd frontend && npm run lint && npm run build
uv run pre-commit install # optional: lint + format on every commit
```

CI (GitHub Actions) runs backend lint and tests, frontend lint and build, the ranking evaluation (failing if mean NDCG@5 drops below 0.93), and a Docker image build on every push and pull request.

## Limitations

- **English, text-based resumes only.** Scanned PDFs have no extractable text (no OCR yet).
- **Skill scoring uses a curated registry**, so skills outside it don't affect the score. With Ollama running, the displayed matching skills also include LLM-reported ones that appear in the resume.
- **The LLM needs memory.** `gemma4:12b` uses about 8GB. On a 16–18GB laptop with other apps open, macOS starts swapping and generation slows from ~16 to ~1–3 tokens a second. If that happens, close other apps or use a smaller model (`OLLAMA_MODEL=qwen3:8b`, ~5GB).
- **The LLM doesn't affect the ranking.** Order and scores come from the embedding and skill scoring, and the assessment explains them. A candidate the LLM calls "No" can still rank highly, so read both.
- **Enrichment makes outbound requests.** It fetches public GitHub profiles and portfolio pages linked in resumes. Only public addresses are allowed (private, loopback, and cloud-metadata IPs are blocked), and LinkedIn and other social sites are skipped.
- **Screening support, not a decision-maker.** Scores and summaries are aids for a human reviewer; automated hiring decisions carry legal and fairness obligations in many jurisdictions.
