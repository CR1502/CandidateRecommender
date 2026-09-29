# Convenience targets. Each is a thin wrapper — see README for the raw commands
# (e.g. on Windows without make).
.PHONY: install api web test lint format eval eval-llm gen-api docker

install:          ## Install backend + frontend dependencies
	uv sync
	cd frontend && npm ci

api:              ## Run the FastAPI backend with auto-reload on :8000
	uv run uvicorn candidate_recommender.api.main:app --reload --port 8000

web:              ## Run the Vite dev server on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:             ## Run the backend test suite
	uv run pytest -q

lint:             ## Lint backend and frontend
	uv run ruff check
	uv run ruff format --check
	cd frontend && npm run lint

format:           ## Auto-fix lint issues and format backend code
	uv run ruff check --fix
	uv run ruff format

eval:             ## Ranking quality on the labelled eval set
	uv run python eval/run_eval.py

eval-llm:         ## LLM assessment quality (needs Ollama running; slow)
	uv run python eval/run_llm_eval.py

gen-api:          ## Regenerate the OpenAPI schema and the frontend's API types
	uv run python -m candidate_recommender.api.export_openapi frontend/src/api/openapi.json
	cd frontend && npm run gen:api

docker:           ## Build and run app + Ollama with Docker Compose
	docker compose up --build
