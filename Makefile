# Xoe-NovAi Makefile (Full Version)
# Last Updated: 2025-10-28
# Purpose: Production utilities for setup, docker, testing, debugging

.PHONY: help wheelhouse deps download-models validate health benchmark curate ingest test build up down logs debug-rag debug-ui debug-crawler debug-redis restart cleanup

COMPOSE := sudo docker compose
PYTHON := python3
PYTEST := pytest
DOCKER_EXEC := sudo docker exec
WHEELHOUSE_DIR := wheelhouse
REQ_GLOB := "requirements-*.txt"
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

help: ## Show this help message
    @echo "Xoe-NovAi Makefile Targets:"
    @awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

wheelhouse: ## Download all Python dependencies to wheelhouse/ for offline install
    @echo "$(CYAN)Downloading Python packages to wheelhouse/...$(NC)"
    ./scripts/download_wheelhouse.sh $(WHEELHOUSE_DIR) $(REQ_GLOB)
    @echo "$(GREEN)✓ Wheelhouse created in $(WHEELHOUSE_DIR)/$(NC)"

deps: wheelhouse ## Install dependencies from wheelhouse (offline)
    @echo "$(CYAN)Installing dependencies from wheelhouse...$(NC)"
    $(PYTHON) -m pip install --no-index --find-links=$(WHEELHOUSE_DIR) -r requirements-api.txt
    $(PYTHON) -m pip install --no-index --find-links=$(WHEELHOUSE_DIR) -r requirements-chainlit.txt
    $(PYTHON) -m pip install --no-index --find-links=$(WHEELHOUSE_DIR) -r requirements-crawl.txt
    $(PYTHON) -m pip install --no-index --find-links=$(WHEELHOUSE_DIR) -r requirements-curation_worker.txt
    @echo "$(GREEN)✓ Dependencies installed from wheelhouse$(NC)"

download-models: ## Download models and embeddings
    @echo "Downloading models..."
    mkdir -p models embeddings
    wget -P models https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-UD-Q5_K_XL.gguf?download=true
    wget -P embeddings https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q8_0.gguf?download=true
#   wget -P embeddings https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.F16.gguf?download=true
#   wget -P embeddings https://huggingface.co/prithivida/all-MiniLM-L6-v2-gguf/resolve/main/all-MiniLM-L6-v2-q8_0.gguf?download=true

validate: ## Run configuration validation
    @echo "Validating configuration..."
    python3 scripts/validate_config.py

health: ## Run health checks
    @echo "Running health checks..."
    python3 app/XNAi_rag_app/healthcheck.py

benchmark: ## Run performance benchmark
    @echo "Running benchmark..."
    python3 scripts/query_test.py --benchmark

curate: ## Run curation (example: Gutenberg classics)
    @echo "Running curation..."
    sudo docker exec xnai_crawler python3 /app/XNAi_rag_app/crawl.py --curate gutenberg -c classics -q "Plato" --max-items=50

ingest: ## Run library ingestion
    @echo "Running ingestion..."
    sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/ingest_library.py --library-path /library

test: ## Run tests with coverage
    @echo "Running tests..."
    cp .env.example .env
    pytest --cov

build: ## Build Docker images
    @echo "Building images..."
    $(COMPOSE) build --no-cache

up: ## Start stack
    @echo "Starting stack..."
    cat redis_password.txt | docker secret create redis_password -
    $(COMPOSE) up -d

down: ## Stop stack
    @echo "Stopping stack..."
    $(COMPOSE) down

logs: ## Show stack logs
    @echo "Showing logs..."
    $(COMPOSE) logs -f

debug-rag: ## Debug shell for RAG
    @echo "Entering RAG shell..."
    $(DOCKER_EXEC) -it xnai_rag_api bash

debug-ui: ## Debug shell for UI
    @echo "Entering UI shell..."
    $(DOCKER_EXEC) -it xnai_chainlit_ui bash

debug-crawler: ## Debug shell for Crawler
    @echo "Entering Crawler shell..."
    $(DOCKER_EXEC) -it xnai_crawler bash

debug-redis: ## Debug shell for Redis
    @echo "Entering Redis shell..."
    $(DOCKER_EXEC) -it xnai_redis bash

restart: ## Restart stack
    @echo "Restarting stack..."
    $(COMPOSE) restart

cleanup: ## Clean volumes and images (warning: data loss)
    @echo "Cleaning up (data loss possible)..."
    $(COMPOSE) down -v --rmi all