# Xoe-NovAi Makefile (Full Version)
# Last Updated: 2026-01-08 (Voice-to-Voice + Build Tracking + Wheel Management v0.1.5)
# Purpose: Production utilities for setup, docker, testing, debugging
# Guide Reference: Section 6.3 (Build Orchestration)
# Features: Voice-to-Voice conversation, dependency tracking, wheel management
# Ryzen Opt: N_THREADS=6 implicit in env; Telemetry: 8 disables verified in Dockerfiles

.PHONY: help wheelhouse deps download-models validate health benchmark curate ingest test build up down logs debug-rag debug-ui debug-crawler debug-redis restart cleanup build-analyze build-report check-duplicates voice-test voice-build wheel-build wheel-analyze build-tracking stack-cat stack-cat-default stack-cat-api stack-cat-rag stack-cat-frontend stack-cat-crawler stack-cat-voice stack-cat-all stack-cat-separate stack-cat-deconcat stack-cat-clean stack-cat-archive

COMPOSE := sudo DOCKER_BUILDKIT=1 docker compose
PYTHON := python3
PYTEST := pytest
DOCKER_EXEC := sudo docker exec
WHEELHOUSE_DIR := wheelhouse
REQ_GLOB := "requirements-*.txt"
# BuildKit enabled for advanced caching and offline builds
export DOCKER_BUILDKIT := 1
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
#	wget -P embeddings https://huggingface.co/leliuga/all-MiniLM-L12-v2-GGUF/resolve/main/all-MiniLM-L12-v2.F16.gguf?download=true
#	wget -P embeddings https://huggingface.co/prithivida/all-MiniLM-L6-v2-gguf/resolve/main/all-MiniLM-L6-v2-q8_0.gguf?download=true

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

build: ## Build Docker images with BuildKit caching and offline optimization
	echo "$(CYAN)Starting enterprise-grade build process with BuildKit caching...$(NC)"
	@if [ ! -f versions/versions.toml ]; then \
		echo "$(YELLOW)Warning: versions/versions.toml not found - skipping version validation$(NC)"; \
	else \
		@echo "$(CYAN)Running pre-build validation...$(NC)"; \
		python3 versions/scripts/update_versions.py 2>/dev/null || { \
			echo "$(YELLOW)Warning: Version validation failed - continuing build$(NC)"; \
		}; \
	fi
	@echo "$(CYAN)Building Docker images with BuildKit cache mounts...$(NC)"
	@echo "$(YELLOW)Note: Wheelhouse is now built inside Docker with persistent caching$(NC)"
	@echo "$(YELLOW)No external downloads needed - all caching handled by BuildKit$(NC)"
	@if [ -f docker-compose.yml ]; then \
		$(COMPOSE) build --progress=plain || { \
			echo "$(RED)Error: Build failed. Check Docker build logs with:$(NC)"; \
			echo "$(YELLOW)  docker compose logs$(NC)"; \
			echo "$(YELLOW)  docker compose build --no-cache --progress=plain$(NC)"; \
			exit 1; \
		}; \
	else \
		echo "$(RED)Error: docker-compose.yml not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Build completed successfully with BuildKit caching$(NC)"
	@sudo docker buildx du --format 'table {{.Size}}' 2>/dev/null | tail -1 | sed 's/^/$(YELLOW)Cache utilization: /' || echo "$(YELLOW)Build cache info unavailable$(NC)"

up: ## Start stack
	@echo "Starting stack..."
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Creating from .env.example...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "$(RED)Error: .env.example not found$(NC)"; \
	fi
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

# ============================================================================
# VOICE-TO-VOICE CONVERSATION SYSTEM TARGETS
# ============================================================================

voice-test: ## Test voice interface functionality
	@echo "$(CYAN)Testing voice interface...$(NC)"
	@if [ ! -d "app/XNAi_rag_app" ]; then \
		echo "$(RED)Error: app/XNAi_rag_app directory not found$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON) -c "import sys; sys.path.insert(0, 'app/XNAi_rag_app'); \
	try: \
		from voice_interface import VoiceInterface, VoiceConfig; \
		print('$(GREEN)✓ Voice interface imports successful$(NC)'); \
		config = VoiceConfig(); \
		print(f'✓ Voice config: STT={config.stt_provider.value}, TTS={config.tts_provider.value}'); \
	except ImportError as e: \
		print(f'$(YELLOW)⚠ Voice interface not fully installed (run make deps first): {e}$(NC)'); \
		exit(0); \
	except Exception as e: \
		print(f'$(RED)✗ Voice interface test failed: {e}$(NC)'); \
		exit(1)"

voice-build: ## Build Docker image with voice-to-voice support
	@echo "$(CYAN)Building Docker image with voice-to-voice support...$(NC)"
	$(COMPOSE) build chainlit
	@echo "$(GREEN)✓ Voice-enabled Chainlit image built$(NC)"
	@echo "$(YELLOW)Run 'make voice-up' to start voice-enabled UI$(NC)"

voice-up: ## Start voice-enabled UI only
	@echo "$(CYAN)Starting voice-enabled UI...$(NC)"
	$(COMPOSE) up -d chainlit
	@echo "$(GREEN)✓ Voice-enabled UI started$(NC)"
	@echo "$(YELLOW)Access at: http://localhost:8001$(NC)"
	@echo "$(YELLOW)Voice features: Click '🎤 Start Voice Chat' to begin$(NC)"

# ============================================================================
# BUILD TRACKING & DEPENDENCY MANAGEMENT TARGETS
# ============================================================================

build-tracking: ## Run build dependency tracking analysis
	@echo "$(CYAN)Running build dependency tracking...$(NC)"
	@if [ ! -f scripts/build_tracking.py ]; then \
		echo "$(RED)Error: scripts/build_tracking.py not found$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/build_tracking.py parse-requirements
	$(PYTHON) scripts/build_tracking.py analyze-installation 2>/dev/null || echo "$(YELLOW)Note: No installation log found (run after pip install)$(NC)"
	$(PYTHON) scripts/build_tracking.py generate-report
	@echo "$(GREEN)✓ Build tracking analysis complete$(NC)"
	@echo "$(YELLOW)Reports saved in current directory$(NC)"

build-analyze: ## Analyze current build state and dependencies
	@echo "$(CYAN)Analyzing current build state...$(NC)"
	@if [ ! -f scripts/build_tracking.py ]; then \
		echo "$(RED)Error: scripts/build_tracking.py not found$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/build_tracking.py parse-requirements
	@echo "$(CYAN)Current dependency status:$(NC)"
	$(PYTHON) scripts/build_tracking.py analyze-installation 2>/dev/null || echo "$(YELLOW)No installation data available$(NC)"
	$(PYTHON) scripts/build_tracking.py check-duplicates
	@echo "$(GREEN)✓ Build analysis complete$(NC)"

build-report: ## Generate comprehensive build report
	@echo "$(CYAN)Generating comprehensive build report...$(NC)"
	@if [ ! -f scripts/build_tracking.py ]; then \
		echo "$(RED)Error: scripts/build_tracking.py not found$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/build_tracking.py generate-report
	@echo "$(GREEN)✓ Build report generated$(NC)"
	@if [ -f build-report.json ]; then \
		echo "$(CYAN)Report summary:$(NC)"; \
		$(PYTHON) -c "import json; print('  Build report saved to build-report.json')"; \
	fi

check-duplicates: ## Check for duplicate packages in current environment
	@echo "$(CYAN)Checking for duplicate packages...$(NC)"
	@if [ ! -f scripts/build_tracking.py ]; then \
		echo "$(RED)Error: scripts/build_tracking.py not found$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) scripts/build_tracking.py check-duplicates
	@echo "$(GREEN)✓ Duplicate check complete$(NC)"

# ============================================================================
# WHEEL MANAGEMENT TARGETS
# ============================================================================

wheel-build: ## Build wheels for all requirements (for offline caching)
	@echo "$(CYAN)Building wheels for offline caching...$(NC)"
	@if [ ! -d $(WHEELHOUSE_DIR) ]; then \
		mkdir -p $(WHEELHOUSE_DIR); \
	fi
	@echo "$(CYAN)Building wheels for API requirements...$(NC)"
	$(PYTHON) -m pip wheel --no-deps -r requirements-api.txt -w $(WHEELHOUSE_DIR) --progress-bar off
	@echo "$(CYAN)Building wheels for Chainlit requirements...$(NC)"
	$(PYTHON) -m pip wheel --no-deps -r requirements-chainlit.txt -w $(WHEELHOUSE_DIR) --progress-bar off
	@echo "$(CYAN)Building wheels for Crawl requirements...$(NC)"
	$(PYTHON) -m pip wheel --no-deps -r requirements-crawl.txt -w $(WHEELHOUSE_DIR) --progress-bar off
	@echo "$(CYAN)Building wheels for Curation Worker requirements...$(NC)"
	$(PYTHON) -m pip wheel --no-deps -r requirements-curation_worker.txt -w $(WHEELHOUSE_DIR) --progress-bar off
	@echo "$(CYAN)Compressing wheelhouse...$(NC)"
	@if [ "$$(ls -1 $(WHEELHOUSE_DIR)/*.whl 2>/dev/null | wc -l)" -gt 0 ]; then \
		tar -czf wheelhouse.tgz -C $(WHEELHOUSE_DIR) . && \
		echo "$(GREEN)✓ Wheelhouse compressed: $$(ls -lh wheelhouse.tgz | awk '{print $$5}')$(NC)"; \
	else \
		echo "$(YELLOW)Warning: No wheels built$(NC)"; \
	fi
	@echo "$(GREEN)✓ Wheel building complete$(NC)"
	@echo "$(YELLOW)Use 'make deps' to install from wheelhouse$(NC)"

wheel-analyze: ## Analyze wheelhouse contents and dependencies
	@echo "$(CYAN)Analyzing wheelhouse contents...$(NC)"
	@if [ ! -d $(WHEELHOUSE_DIR) ]; then \
		echo "$(RED)Error: Wheelhouse directory not found. Run 'make wheel-build' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Wheelhouse statistics:$(NC)"
	@ls -1 $(WHEELHOUSE_DIR)/*.whl 2>/dev/null | wc -l | xargs echo "  Total wheels:"
	@du -sh $(WHEELHOUSE_DIR) 2>/dev/null | awk '{print "  Total size: " $$1}' || echo "  Total size: Unknown"
	@if [ -f wheelhouse.tgz ]; then \
		ls -lh wheelhouse.tgz | awk '{print "  Compressed size: " $$5}'; \
	fi
	@echo "$(CYAN)Sample wheels:$(NC)"
	@ls -1 $(WHEELHOUSE_DIR)/*.whl 2>/dev/null | head -5 | sed 's/^/  /'
	@echo "$(GREEN)✓ Wheelhouse analysis complete$(NC)"
	@du -sh $(WHEELHOUSE_DIR) 2>/dev/null | awk '{print "  Total size: " $$1}' || echo "  Total size: Unknown"

# ============================================================================
# STACK-CAT DOCUMENTATION GENERATOR TARGETS
# ============================================================================

stack-cat: stack-cat-default ## Generate default stack documentation (alias)

stack-cat-default: ## Generate default stack documentation (all components)
	@echo "$(CYAN)Generating Xoe-NovAi v0.1.5 stack documentation...$(NC)"
	@if [ ! -f scripts/stack-cat/stack-cat.sh ]; then \
		echo "$(RED)Error: Stack-Cat script not found at scripts/stack-cat/stack-cat.sh$(NC)"; \
		exit 1; \
	fi
	@cd scripts/stack-cat && ./stack-cat.sh -g default -f all
	@echo "$(GREEN)✓ Stack documentation generated$(NC)"
	@echo "$(YELLOW)Output: scripts/stack-cat/stack-cat-output/$(NC)"
	@ls -la scripts/stack-cat/stack-cat-output/ | tail -3

stack-cat-api: ## Generate API backend documentation only
	@echo "$(CYAN)Generating API documentation...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g api -f all
	@echo "$(GREEN)✓ API documentation generated$(NC)"

stack-cat-rag: ## Generate RAG subsystem documentation only
	@echo "$(CYAN)Generating RAG documentation...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g rag -f all
	@echo "$(GREEN)✓ RAG documentation generated$(NC)"

stack-cat-frontend: ## Generate UI frontend documentation only
	@echo "$(CYAN)Generating UI frontend documentation...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g frontend -f all
	@echo "$(GREEN)✓ UI frontend documentation generated$(NC)"

stack-cat-crawler: ## Generate CrawlModule subsystem documentation only
	@echo "$(CYAN)Generating CrawlModule documentation...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g crawler -f all
	@echo "$(GREEN)✓ CrawlModule documentation generated$(NC)"

stack-cat-voice: ## Generate voice interface documentation only
	@echo "$(CYAN)Generating voice interface documentation...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g voice -f all
	@echo "$(GREEN)✓ Voice interface documentation generated$(NC)"

stack-cat-all: ## Generate documentation for all groups
	@echo "$(CYAN)Generating documentation for all groups...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g default -f all
	@cd scripts/stack-cat && ./stack-cat.sh -g api -f all
	@cd scripts/stack-cat && ./stack-cat.sh -g rag -f all
	@cd scripts/stack-cat && ./stack-cat.sh -g frontend -f all
	@cd scripts/stack-cat && ./stack-cat.sh -g crawler -f all
	@cd scripts/stack-cat && ./stack-cat.sh -g voice -f all
	@echo "$(GREEN)✓ All documentation generated$(NC)"

stack-cat-separate: ## Generate separate markdown files for each source file
	@echo "$(CYAN)Generating separate markdown files...$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -g default -s
	@echo "$(GREEN)✓ Separate markdown files generated$(NC)"
	@echo "$(YELLOW)Files: scripts/stack-cat/stack-cat-output/separate-md/$(NC)"

stack-cat-deconcat: ## De-concatenate markdown file into separate files
	@echo "$(CYAN)De-concatenating markdown file...$(NC)"
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: Specify FILE variable (e.g., make stack-cat-deconcat FILE=stack-cat-output/20251021_143022/stack-cat_20251021_143022.md)$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)De-concatenating: $(FILE)$(NC)"
	@cd scripts/stack-cat && ./stack-cat.sh -d "$(FILE)"
	@echo "$(GREEN)✓ De-concatenation complete$(NC)"

stack-cat-clean: ## Clean up stack-cat output directories (WARNING: PERMANENT DELETION)
	@echo "$(RED)⚠️  WARNING: This will permanently delete ALL historical Stack-Cat documentation snapshots!$(NC)"
	@echo "$(YELLOW)Output directory: scripts/stack-cat/stack-cat-output/$(NC)"
	@read -p "Are you sure you want to permanently delete all Stack-Cat output? (type 'yes' to confirm): " confirm && \
	if [ "$$confirm" = "yes" ]; then \
		if [ -d scripts/stack-cat/stack-cat-output ]; then \
			echo "$(CYAN)Cleaning up Stack-Cat output...$(NC)"; \
			rm -rf scripts/stack-cat/stack-cat-output && \
			echo "$(GREEN)✓ Stack-Cat output permanently deleted$(NC)"; \
		else \
			echo "$(YELLOW)No Stack-Cat output directory found$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW)Cancellation confirmed - no files deleted$(NC)"; \
	fi

stack-cat-archive: ## Move Stack-Cat outputs older than 1 week to archive folder
	@echo "$(CYAN)Archiving Stack-Cat outputs older than 1 week...$(NC)"
	@if [ ! -d scripts/stack-cat/stack-cat-output ]; then \
		echo "$(YELLOW)No stack-cat-output directory found$(NC)"; \
		exit 0; \
	fi
	@mkdir -p scripts/stack-cat/stack-cat-archive
	@echo "$(CYAN)Finding files older than 7 days...$(NC)"
	@cd scripts/stack-cat && find stack-cat-output -type f -mtime +7 | while read -r file; do \
		echo "$(YELLOW)Archiving: $$file$(NC)"; \
		dirpath=$$(dirname "stack-cat-archive/$$file"); \
		mkdir -p "$$dirpath"; \
		mv "$$file" "stack-cat-archive/$$file"; \
	done
	@echo "$(CYAN)Removing empty directories from output...$(NC)"
	@cd scripts/stack-cat && find stack-cat-output -type d -empty -delete 2>/dev/null || true
	@archived_count=$$(find scripts/stack-cat/stack-cat-archive -type f -mtime +7 2>/dev/null | wc -l); \
	if [ "$$archived_count" -gt 0 ]; then \
		echo "$(GREEN)✓ Archived $$archived_count files older than 1 week$(NC)"; \
		echo "$(YELLOW)Archive location: scripts/stack-cat/stack-cat-archive/$(NC)"; \
	else \
		echo "$(YELLOW)No files older than 1 week to archive$(NC)"; \
	fi
