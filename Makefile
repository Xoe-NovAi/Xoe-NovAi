
# Xoe-NovAi Phase 1 v0.1.2 Makefile
# Purpose: Production utilities for setup, docker management, testing, debugging
# Guide Reference: Section 5.3 (Health Checks), 2.4 (Validation)
# Last Updated: 2025-10-18

.PHONY: help download-models validate health benchmark curate ingest test build up down logs debug-rag debug-ui debug-crawler debug-redis restart cleanup

help: ## Show this help message
	@echo "Xoe-NovAi Makefile Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $1, $2}' $(MAKEFILE_LIST)

download-models: ## Download models and embeddings
	@echo "Downloading models..."
	mkdir -p models embeddings
	wget -P models https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-UD-Q5_K_XL.gguf?download=true
	wget -P embeddings https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/all-MiniLM-L12-v2.Q8_0.gguf

validate: ## Run configuration validation
	@echo "Validating configuration..."
	python3 scripts/validate_config.py

health: ## Run health checks
	@echo "Running health checks..."
	python3 app/XNAi_rag_app/healthcheck.py

benchmark: ## Run performance benchmark
	@echo "Running benchmark..."
	python3 scripts/query_test.py --benchmark  # Assumes --benchmark flag for token rate/memory check

curate: ## Run curation (example: Gutenberg classics)
	@echo "Running curation..."
	sudo docker exec xnai_crawler python3 /app/XNAi_rag_app/crawl.py --curate gutenberg -c classics -q "Plato" --max-items=50

ingest: ## Run library ingestion
	@echo "Running ingestion..."
	sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/ingest_library.py --library-path /library

test: ## Run tests with coverage
	@echo "Running tests..."
	pytest --cov

build: ## Build Docker images
	@echo "Building images..."
	sudo docker compose build --no-cache

up: ## Start stack
	@echo "Starting stack..."
	sudo docker compose up -d

down: ## Stop stack
	@echo "Stopping stack..."
	sudo docker compose down

logs: ## Show stack logs
	@echo "Showing logs..."
	sudo docker compose logs -f

debug-rag: ## Debug shell for RAG
	@echo "Entering RAG shell..."
	sudo docker exec -it xnai_rag_api bash

debug-ui: ## Debug shell for UI
	@echo "Entering UI shell..."
	sudo docker exec -it xnai_chainlit_ui bash

debug-crawler: ## Debug shell for Crawler
	@echo "Entering Crawler shell..."
	sudo docker exec -it xnai_crawler bash

debug-redis: ## Debug shell for Redis
	@echo "Entering Redis shell..."
	sudo docker exec -it xnai_redis bash

restart: ## Restart stack
	@echo "Restarting stack..."
	sudo docker compose restart

cleanup: ## Clean volumes and images (warning: data loss)
	@echo "Cleaning up (data loss possible)..."
	sudo docker compose down -v --rmi all