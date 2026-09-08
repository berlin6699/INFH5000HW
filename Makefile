# ============================================================================
# Multimodal Multi-Agent Healthcare Assistant — developer entry points
#
# Everything runs against a conda env because the system Python is 3.9.6,
# which is too old for this dependency set. Override with `make PY=/path/to/python`.
# ============================================================================

CONDA_ENV  ?= medai
CONDA_BASE ?= $(shell conda info --base 2>/dev/null)
PY         ?= $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python

HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

# --- Guard ------------------------------------------------------------------
# Fail with an actionable message instead of a confusing "command not found".
$(PY):
	@echo "ERROR: Python interpreter not found at $(PY)"
	@echo "Create it with:  make setup"
	@echo "Or point at one: make PY=/path/to/python3.11 <target>"
	@exit 1

.PHONY: help setup check backend frontend dev seed demo test types lint clean distclean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create conda env, install backend + frontend deps
	@command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
	conda create -y -n $(CONDA_ENV) python=3.11
	$(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python -m pip install --upgrade pip
	$(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python -m pip install -r backend/requirements.txt
	cd frontend && npm install
	@[ -f .env ] || cp .env.example .env
	@echo "\nSetup complete. Next:  make dev"

check: $(PY) ## Verify the toolchain and print resolved versions
	@$(PY) -c "import sys,importlib.metadata as md; \
print('python', sys.version.split()[0]); \
[print(f'{p:16}', md.version(p)) for p in ['langgraph','fastapi','pydantic','sqlalchemy','numpy']]"
	@cd frontend && node -e "const p=require('./package.json'); \
console.log('react',p.dependencies.react); console.log('vite',p.devDependencies.vite)"

backend: $(PY) ## Run FastAPI with autoreload
	cd backend && $(PY) -m uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

frontend: ## Run the Vite dev server
	cd frontend && npm run dev

dev: ## Run backend + frontend together (Ctrl-C stops both)
	@$(MAKE) -j2 backend frontend

seed: $(PY) ## Recreate the database and load the synthetic demo patient
	cd backend && $(PY) -m app.db.seed

test: $(PY) ## Run backend tests
	cd backend && $(PY) -m pytest -q

types: $(PY) ## Regenerate frontend/src/types/generated.ts from pydantic schemas
	cd backend && $(PY) -m app.tools.export_types
	cd frontend && npx --yes json-schema-to-typescript src/types/schema.json \
		-o src/types/generated.ts \
		--unreachableDefinitions \
		--bannerComment "/* AUTO-GENERATED from the backend pydantic schemas via \`make types\`. Do not edit. */"
	@echo "Types written to frontend/src/types/generated.ts"

lint: ## Lint frontend and byte-compile backend
	cd frontend && npm run lint
	$(PY) -m compileall -q backend/app

clean: ## Remove build artifacts and caches
	rm -rf frontend/dist backend/.pytest_cache
	find backend -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean ## Also delete the database (forces a fresh seed)
	rm -f backend/data/*.db backend/data/*.sqlite*

# Phase 9 target: one command from empty checkout to a completed, asserted
# multi-agent analysis. Wired up once the Coordinator exists.
demo: seed ## Recreate DB, seed, run the full multi-agent analysis end-to-end
	@echo "Phase 9 will extend this to run the analysis and assert risk_level=HIGH."
