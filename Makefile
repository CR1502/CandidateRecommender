# Convenience targets. Each is a thin wrapper — see README for the raw commands
# (e.g. on Windows without make).
.PHONY: install api web test lint format docker

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

docker:           ## Build and run app + Ollama with Docker Compose
	docker compose up --build
