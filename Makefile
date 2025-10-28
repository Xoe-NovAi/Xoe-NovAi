# Xoe-NovAi Phase 1 v0.1.3 Makefile
# Purpose: Production utilities for setup, docker, testing, debugging
# Guide Reference: Section 5.3 (Health Checks), 2.4 (Validation)

# Last Updated: 2025-10-18

This document outlines best practices for managing secrets in the Xoe-NovAi stack. Follow these guidelines to ensure secure handling of sensitive information.

.PHONY: help download-models validate health benchmark curate ingest test build up down logs \
        debug-rag debug-ui debug-crawler debug-redis restart cleanup wheelhouse deps

# ============================================================================
# CONFIGURATION
# ============================================================================

COMPOSE := sudo docker compose
PYTHON := python3
PYTEST := pytest
DOCKER_EXEC := sudo docker exec
WHEELHOUSE_DIR := wheelhouse
REQ_GLOB := "requirements-*.txt"

# Color output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo "Xoe-NovAi Makefile Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ============================================================================
# SETUP & BUILD
# ============================================================================

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

	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $1, $2}' $(MAKEFILE_LIST)

- Never commit real secrets to version control

- Use placeholder values in `.env.example` filesdownload-models: ## Download models and embeddings

- Use strong, randomly generated passwords	@echo "Downloading models..."

- Keep production secrets in a secure secrets manager	mkdir -p models embeddings

	wget -P models https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-UD-Q5_K_XL.gguf?download=true

### 2. Docker Secrets#   wget -P embeddings https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.F16.gguf?download=true

    wget -P embeddings https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.Q8_0.gguf?download=true

```bash#	wget -P embeddings https://huggingface.co/prithivida/all-MiniLM-L6-v2-gguf/resolve/main/all-MiniLM-L6-v2-q8_0.gguf?download=true

# Create a Docker secret

echo "your_secure_password" | docker secret create redis_password -validate: ## Run configuration validation



# Use in docker-compose.ymlsecurity-scan: ## Scan Docker images for vulnerabilities

services:	@echo "Scanning Docker images for security issues..."

  redis:	@docker scan xnai_rag_api || true

    secrets:	@docker scan xnai_chainlit_ui || true

      - redis_password	@docker scan xnai_crawler || true

    environment:	@docker scan xnai_curation_worker || true

      - REDIS_PASSWORD_FILE=/run/secrets/redis_password	@echo "Security scan complete. Check above for any vulnerabilities."

```	@echo "Validating configuration..."

	python3 scripts/validate_config.py

### 3. CI/CD Security

health: ## Run health checks

- Use GitHub Actions secrets or similar for CI/CD	@echo "Running health checks..."

- Rotate secrets regularly	python3 app/XNAi_rag_app/healthcheck.py

- Use environment-specific secrets

- Never log or display secretsbenchmark: ## Run performance benchmark

	@echo "Running benchmark..."

### 4. HashiCorp Vault Integration (Phase 2)	python3 scripts/query_test.py --benchmark  # Assumes --benchmark flag for token rate/memory check



When PHASE2_QDRANT_ENABLED=true:curate: ## Run curation (example: Gutenberg classics)

- Use Vault for dynamic secrets	@echo "Running curation..."

- Implement secret rotation	sudo docker exec xnai_crawler python3 /app/XNAi_rag_app/crawl.py --curate gutenberg -c classics -q "Plato" --max-items=50

- Enable audit logging

ingest: ## Run library ingestion

## Implementation Guide	@echo "Running ingestion..."

	sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/ingest_library.py --library-path /library

1. Local Development:

   ```bashtest: ## Run tests with coverage

   # Copy example env and replace placeholders	@echo "Running tests..."

   cp .env.example .env	pytest --cov

   # Generate secure password

   openssl rand -base64 32 > redis_password.txtbuild: ## Build Docker images

   ```	@echo "Building images..."

	sudo docker compose build --no-cache

2. Production Deployment:

   ```bashup: ## Start stack

   # Create Docker secrets	@echo "Starting stack..."

   cat redis_password.txt | docker secret create redis_password -	sudo docker compose up -d

   

   # Update docker-compose.yml to use secretsdown: ## Stop stack

   # See Docker Secrets section above	@echo "Stopping stack..."

   ```	sudo docker compose down



3. CI/CD Setup:logs: ## Show stack logs

   ```yaml	@echo "Showing logs..."

   # GitHub Actions example	sudo docker compose logs -f

   steps:

     - uses: actions/checkout@v4debug-rag: ## Debug shell for RAG

     - name: Build and test	@echo "Entering RAG shell..."

       env:	sudo docker exec -it xnai_rag_api bash

         REDIS_PASSWORD: ${{ secrets.REDIS_PASSWORD }}

   ```debug-ui: ## Debug shell for UI

	@echo "Entering UI shell..."

## Security Checklist	sudo docker exec -it xnai_chainlit_ui bash



- [ ] No secrets in version controldebug-crawler: ## Debug shell for Crawler

- [ ] Strong password generation	@echo "Entering Crawler shell..."

- [ ] Secrets manager integration	sudo docker exec -it xnai_crawler bash

- [ ] Regular secret rotation

- [ ] Audit logging enableddebug-redis: ## Debug shell for Redis

- [ ] CI/CD secrets configured	@echo "Entering Redis shell..."

- [ ] Docker secrets implemented	sudo docker exec -it xnai_redis bash



## Referencerestart: ## Restart stack

	@echo "Restarting stack..."

- Docker Secrets: https://docs.docker.com/engine/swarm/secrets/	sudo docker compose restart

- GitHub Secrets: https://docs.github.com/en/actions/security-guides/encrypted-secrets

- HashiCorp Vault: https://www.vaultproject.io/cleanup: ## Clean volumes and images (warning: data loss)
	@echo "Cleaning up (data loss possible)..."
	sudo docker compose down -v --rmi all