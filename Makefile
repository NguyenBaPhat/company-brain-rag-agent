# Company Brain — developer commands. `make help` lists them.
PY      := .venv/bin/python
SSL     := SSL_CERT_FILE=$${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}

.DEFAULT_GOAL := help
.PHONY: help setup qdrant ingest backend frontend test test-fast test-ui eval eval-agent lint up down logs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv, install backend + frontend deps, copy .env
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e "backend[dev]"
	cd frontend && npm install
	@test -f .env || cp .env.example .env && echo ".env ready: add your GOOGLE_API_KEY"

qdrant:  ## Start only the Qdrant vector database (dashboard: http://localhost:6333/dashboard)
	docker compose up -d qdrant

ingest:  ## Chunk + embed the knowledge base and upsert into Qdrant (idempotent)
	$(SSL) $(PY) -m company_brain.knowledge.ingest

backend:  ## Run the API + agent with auto-reload on :8000
	$(SSL) $(PY) -m uvicorn company_brain.api.app:create_app --factory --reload --port 8000 --reload-dir backend/src

frontend:  ## Run the React dev server on :5173 (proxies /api and /ws to :8000)
	cd frontend && npm run dev

test:  ## Backend tests (real local embeddings, mock LLM) + frontend unit tests
	$(SSL) $(PY) -m pytest backend/tests -q
	cd frontend && npm test

test-fast:  ## Backend tests that do not load embedding models
	$(PY) -m pytest backend/tests -q -m "not slow"

eval:  ## Offline retrieval eval: hit@k, MRR and relevance-gate calibration
	$(SSL) $(PY) -m company_brain.evaluation.retrieval -v

eval-agent:  ## Behavioural eval of the live agent (needs GOOGLE_API_KEY)
	$(SSL) $(PY) -m company_brain.evaluation.agent_eval

lint:  ## Ruff + TypeScript type-check
	$(PY) -m ruff check backend
	cd frontend && npm run typecheck

up:  ## Build and run the full stack in Docker (UI on :8080)
	docker compose up --build -d

down:  ## Stop the Docker stack
	docker compose down

logs:  ## Tail backend logs
	docker compose logs -f backend
