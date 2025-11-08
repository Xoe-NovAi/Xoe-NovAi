# Xoe-NovAi v0.1.3-beta: Docker Orchestration & Deployment

**Sections**: 7 (Docker Configuration), 13.1-13.3 (Deployment Checklist)  
**Purpose**: Complete Docker build/deploy procedures with security hardening  
**Cross-References**: See Artifact 2 (Prerequisites), Artifact 3 (Config), Artifact 5 (Health Checks)

---

## Section 7: Docker Orchestration & Multi-Stage Builds

### 7.1 Multi-Stage Build Strategy

**Why Multi-Stage**: Reduces image size by 50-85% (industry standard since Docker 17.05), improves security by excluding build tools from runtime, enables separate builder and runtime environments.

**Pattern Applied Across All Dockerfiles**:
1. **Stage 1 (Builder)**: Install build dependencies, compile native extensions (llama-cpp-python)
2. **Stage 2 (Runtime)**: Copy compiled artifacts, use slim base image, run as non-root user

**Security Standards** (applied consistently):
- Non-root user (UID 1001, GID 1001) — 58% of production containers still run as root [Sysdig 2025]
- Minimal base image (python:3.12-slim ~140MB vs python:3.12 ~900MB)
- `cap_drop: [ALL]` — removes all Linux capabilities
- `no-new-privileges:true` — prevents privilege escalation
- Directory creation in builder stage (before USER switch)

---

### 7.2 Dockerfile.api (FastAPI RAG Service)

**Purpose**: RAG API (port 8000) + metrics (port 8002), optimized for Ryzen CPU inference

**File**: `Dockerfile.api` (multi-stage, non-root, verified)

```dockerfile
# Xoe-NovAi v0.1.3-beta: FastAPI RAG API
# Guide Ref: Section 7.2

# ==============================================================================
# Stage 1: Builder (build dependencies, compile native extensions)
# ==============================================================================
FROM python:3.12-slim AS builder

LABEL maintainer="Xoe-NovAi Team" \
      version="v0.1.3-beta" \
      description="FastAPI RAG API builder stage"

WORKDIR /build

# Install build dependencies (gcc, cmake for llama-cpp-python compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build
COPY requirements-api.txt .

# CRITICAL: Ryzen optimization flags for llama-cpp-python
# Guide Ref: Appendix C (Performance Tuning)
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON" \
    FORCE_CMAKE=1

# Build with user site-packages (non-root compatible)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements-api.txt

# Verify critical imports built successfully
RUN python3 -c "from llama_cpp import Llama; print('llama-cpp OK')" && \
    python3 -c "from langchain_community.vectorstores import FAISS; print('FAISS OK')"

# ==============================================================================
# Stage 2: Runtime (slim, non-root, health-checked)
# ==============================================================================
FROM python:3.12-slim

LABEL maintainer="Xoe-NovAi Team" \
      version="v0.1.3-beta" \
      description="FastAPI RAG API runtime"

# Install runtime dependencies only (curl for healthcheck, libopenblas for inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group (UID/GID 1001)
# Guide Ref: Section 7.1 (Security Standards)
RUN groupadd -g 1001 appgroup && \
    useradd -m -u 1001 -g appgroup appuser

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# CRITICAL: Directory creation + ownership BEFORE USER switch (v0.1.3 fix)
# Guide Ref: comprehensive-fix-pr.md (FIX #2)
RUN mkdir -p \
    /app/XNAi_rag_app/logs \
    /app/XNAi_rag_app/faiss_index \
    /backups \
    /prometheus_data \
    /tmp/llm_cache && \
    chown -R appuser:appgroup /app /backups /prometheus_data /tmp/llm_cache && \
    chmod -R 755 /app /backups /prometheus_data && \
    chmod 777 /app/XNAi_rag_app/logs /tmp/llm_cache

# Copy application code (no trailing slashes per v0.1.3 fix)
# Guide Ref: comprehensive-fix-pr.md (FIX #3)
COPY --chown=appuser:appgroup app/XNAi_rag_app /app/XNAi_rag_app
COPY --chown=appuser:appgroup entrypoint-api.sh /entrypoint-api.sh
RUN chmod +x /entrypoint-api.sh

# Verify critical files copied successfully (v0.1.3 pattern)
RUN test -f /app/XNAi_rag_app/main.py || \
    (echo "ERROR: main.py not found after COPY" && ls -la /app/XNAi_rag_app && exit 1)

# Environment variables (cross-ref to .env)
# Guide Ref: Artifact 3 (Configuration)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    # Ryzen optimization (from .env)
    LLAMA_CPP_N_THREADS=6 \
    LLAMA_CPP_F16_KV=true \
    LLAMA_CPP_USE_MLOCK=true \
    LLAMA_CPP_USE_MMAP=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=ZEN \
    MKL_DEBUG_CPU_TYPE=5 \
    # Memory management
    MALLOC_TRIM_THRESHOLD_=131072 \
    # Telemetry disables
    LLAMA_CPP_NO_TELEMETRY=true \
    LANGCHAIN_NO_TELEMETRY=true

# Expose ports (API + metrics)
EXPOSE 8000 8002

# Health check (90s start_period accounts for LLM load time)
# Guide Ref: Section 7.3 (Health Checks)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=90s \
    CMD python3 /app/XNAi_rag_app/healthcheck.py llm embeddings memory || exit 1

# Switch to non-root user (final security step)
USER appuser

# Entrypoint runs as appuser
ENTRYPOINT ["/entrypoint-api.sh"]
CMD ["uvicorn", "app.XNAi_rag_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Validation Commands**:

```bash
# Build image
sudo docker build -f Dockerfile.api -t xnai_rag_api:v0.1.3-beta .

# Verify multi-stage worked (runtime image should be <500MB)
sudo docker images xnai_rag_api:v0.1.3-beta
# Expected: SIZE ~450MB (vs 900MB+ for single-stage)

# Verify non-root user
sudo docker run --rm xnai_rag_api:v0.1.3-beta id
# Expected: uid=1001(appuser) gid=1001(appgroup)

# Verify files exist
sudo docker run --rm xnai_rag_api:v0.1.3-beta ls -la /app/XNAi_rag_app
# Expected: main.py, dependencies.py, healthcheck.py, etc.

# Verify directory ownership
sudo docker run --rm xnai_rag_api:v0.1.3-beta ls -la /app/XNAi_rag_app/logs
# Expected: drwxrwxrwx ... appuser appgroup ... logs/
```

---

### 7.3 Dockerfile.chainlit (Chainlit UI)

**Purpose**: Interactive UI (port 8001), session management, non-blocking command dispatch

**File**: `Dockerfile.chainlit` (similar pattern to API)

```dockerfile
# Xoe-NovAi v0.1.3-beta: Chainlit UI
# Guide Ref: Section 7.3

FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-chainlit.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements-chainlit.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appgroup && \
    useradd -m -u 1001 -g appgroup appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Create Chainlit-specific directories (includes .files for uploads)
RUN mkdir -p \
    /app/XNAi_rag_app/logs \
    /app/.files && \
    chown -R appuser:appgroup /app && \
    chmod 777 /app/XNAi_rag_app/logs /app/.files

COPY --chown=appuser:appgroup app/XNAi_rag_app /app/XNAi_rag_app
COPY --chown=appuser:appgroup entrypoint-chainlit.sh /entrypoint-chainlit.sh
RUN chmod +x /entrypoint-chainlit.sh

RUN test -f /app/XNAi_rag_app/chainlit_app.py || \
    (echo "ERROR: chainlit_app.py not found" && exit 1)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    CHAINLIT_NO_TELEMETRY=true \
    OPENAI_API_KEY=

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8001 || exit 1

USER appuser

ENTRYPOINT ["/entrypoint-chainlit.sh"]
CMD ["chainlit", "run", "/app/XNAi_rag_app/chainlit_app.py", "--host", "0.0.0.0", "--port", "8001"]
```

**Validation**:

```bash
sudo docker build -f Dockerfile.chainlit -t xnai_chainlit_ui:v0.1.3-beta .
sudo docker images xnai_chainlit_ui:v0.1.3-beta
# Expected: SIZE ~300MB

sudo docker run --rm xnai_chainlit_ui:v0.1.3-beta test -f /app/XNAi_rag_app/chainlit_app.py && echo "OK"
```

---

### 7.4 Dockerfile.crawl (CrawlModule)

**Purpose**: Web crawler (crawl4ai), curates documents to `/library/` and `/knowledge/curator/`

**File**: `Dockerfile.crawl` (playwright + crawl4ai dependencies)

```dockerfile
# Xoe-NovAi v0.1.3-beta: CrawlModule
# Guide Ref: Section 7.4

FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget gnupg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-crawl.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements-crawl.txt && \
    pip install --no-cache-dir --user playwright && \
    python3 -m playwright install --with-deps chromium

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # Playwright runtime dependencies
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appgroup && \
    useradd -m -u 1001 -g appgroup appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local
COPY --from=builder --chown=appuser:appgroup /root/.cache /home/appuser/.cache

# Crawler-specific directories (v0.1.3: explicit creation per comprehensive-fix-pr.md)
RUN mkdir -p \
    /app/XNAi_rag_app \
    /app/XNAi_rag_app/logs \
    /app/cache \
    /library \
    /knowledge/curator && \
    chown -R appuser:appgroup /app /library /knowledge && \
    chmod 777 /app/XNAi_rag_app/logs /app/cache

COPY --chown=appuser:appgroup app/XNAi_rag_app /app/XNAi_rag_app
RUN test -f /app/XNAi_rag_app/crawl.py || \
    (echo "ERROR: crawl.py not found" && ls -la /app/XNAi_rag_app && exit 1)

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    CRAWL4AI_NO_TELEMETRY=true \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python3 -c "import crawl4ai; print('OK')" || exit 1

USER appuser

CMD ["python3", "/app/XNAi_rag_app/crawl.py", "--help"]
```

**Validation**:

```bash
sudo docker build -f Dockerfile.crawl -t xnai_crawler:v0.1.3-beta .
sudo docker images xnai_crawler:v0.1.3-beta
# Expected: SIZE ~600MB (includes Playwright/Chromium)

# Test crawler init
sudo docker run --rm xnai_crawler:v0.1.3-beta python3 -c "import crawl4ai; print('OK')"
# Expected: OK
```

---

### 7.5 docker-compose.yml (Service Orchestration)

**Purpose**: 4-service stack (Redis, RAG API, Chainlit UI, Crawler) with health check dependencies

**File**: `docker-compose.yml` (Compose v2.29.2+ format — NO `version:` key)

```yaml
# Xoe-NovAi v0.1.3-beta: Service Orchestration
# Guide Ref: Section 7.5
# CRITICAL: Compose v2.29.2+ required (no `version:` key)

services:
  redis:
    image: redis:7.4.1-alpine
    container_name: xnai_redis
    user: "999:999"  # Non-root Redis user
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./entrypoint-redis.sh:/entrypoint-redis.sh:ro
    environment:
      # Cross-ref: .env REDIS_PASSWORD (must match ALL services)
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    command: /entrypoint-redis.sh
    healthcheck:
      # Auth-aware health check
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    networks:
      - xnai_network
    restart: unless-stopped
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true

  rag:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: xnai_rag_api:v0.1.3-beta
    container_name: xnai_rag_api
    user: "1001:1001"
    depends_on:
      redis:
        condition: service_healthy  # Wait for Redis ready
    ports:
      - "8000:8000"  # RAG API
      - "8002:8002"  # Prometheus metrics
    volumes:
      # CRITICAL: config.toml READ-ONLY mount (v0.1.3 fix)
      # Guide Ref: comprehensive-fix-pr.md (FIX #1)
      - ./config.toml:/app/XNAi_rag_app/config.toml:ro
      - ./models:/models:ro
      - ./embeddings:/embeddings:ro
      - ./library:/library:ro
      - ./knowledge:/knowledge
      - ./data/faiss_index:/app/XNAi_rag_app/faiss_index
      - ./backups:/backups
      - ./data/prometheus-multiproc:/prometheus_data
    tmpfs:
      - /tmp:mode=1777,size=512m
    environment:
      # Ryzen optimization (cross-ref: .env)
      - LLAMA_CPP_N_THREADS=${LLAMA_CPP_N_THREADS:-6}
      - LLAMA_CPP_F16_KV=${LLAMA_CPP_F16_KV:-true}
      - LLAMA_CPP_USE_MLOCK=${LLAMA_CPP_USE_MLOCK:-true}
      - LLAMA_CPP_USE_MMAP=${LLAMA_CPP_USE_MMAP:-true}
      - OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-ZEN}
      - MKL_DEBUG_CPU_TYPE=${MKL_DEBUG_CPU_TYPE:-5}
      
      # Telemetry disables (8 total, cross-ref: .env)
      - CHAINLIT_NO_TELEMETRY=true
      - CRAWL4AI_NO_TELEMETRY=true
      - LLAMA_CPP_NO_TELEMETRY=true
      - LANGCHAIN_NO_TELEMETRY=true
      - LANGCHAIN_TRACING_V2=false
      - LANGCHAIN_API_KEY=
      - OPENAI_API_KEY=
      - SCARF_NO_ANALYTICS=true
      
      # Models & paths
      - LLM_MODEL_PATH=${LLM_MODEL_PATH:-/models/gemma-3-4b-it-UD-Q5_K_XL.gguf}
      - EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-/embeddings/all-MiniLM-L12-v2.Q8_0.gguf}
      - LIBRARY_PATH=/library
      - KNOWLEDGE_PATH=/knowledge
      
      # Redis connection
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - REDIS_DB=0
      
      # Logging
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - LOG_FORMAT=json
      - PYTHONUNBUFFERED=1
    healthcheck:
      # Custom health script (7 targets: LLM, embeddings, memory, Redis, vectorstore, Ryzen, crawler)
      test: ["CMD", "python3", "/app/XNAi_rag_app/healthcheck.py"]
      interval: 30s
      timeout: 15s
      retries: 3
      start_period: 90s  # LLM load time ~7-10s
    networks:
      - xnai_network
    restart: unless-stopped
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    deploy:
      resources:
        limits:
          cpus: '6.0'
          memory: 6G
        reservations:
          cpus: '4.0'
          memory: 4G

  ui:
    build:
      context: .
      dockerfile: Dockerfile.chainlit
    image: xnai_chainlit_ui:v0.1.3-beta
    container_name: xnai_chainlit_ui
    user: "1001:1001"
    depends_on:
      rag:
        condition: service_healthy  # Wait for API ready
    ports:
      - "8001:8001"
    volumes:
      - ./config.toml:/app/XNAi_rag_app/config.toml:ro
      - ./library:/library:ro
      - ./knowledge:/knowledge
    environment:
      - CHAINLIT_HOST=0.0.0.0
      - CHAINLIT_PORT=8001
      - CHAINLIT_NO_TELEMETRY=true
      - OPENAI_API_KEY=
      - RAG_API_URL=http://rag:8000
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - xnai_network
    restart: unless-stopped
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true

  crawler:
    build:
      context: .
      dockerfile: Dockerfile.crawl
    image: xnai_crawler:v0.1.3-beta
    container_name: xnai_crawler
    user: "1001:1001"
    depends_on:
      rag:
        condition: service_healthy
    volumes:
      - ./config.toml:/app/XNAi_rag_app/config.toml:ro
      - ./library:/library
      - ./knowledge:/knowledge
      - ./data/cache:/app/cache
    environment:
      - CRAWL4AI_NO_TELEMETRY=true
      - CRAWL_SANITIZE_SCRIPTS=true
      - CRAWL_RATE_LIMIT_PER_MIN=30
      - CRAWL4AI_MAX_DEPTH=2
      - LIBRARY_PATH=/library
      - KNOWLEDGE_PATH=/knowledge
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "python3", "-c", "import crawl4ai; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - xnai_network
    restart: unless-stopped
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true

volumes:
  redis_data:
    driver: local

networks:
  xnai_network:
    driver: bridge
```

**Key Configuration Cross-References**:

| Environment Variable | Source | Usage |
|---------------------|--------|-------|
| `REDIS_PASSWORD` | `.env` | Used in redis, rag, ui services for auth |
| `LLAMA_CPP_N_THREADS` | `.env` | Ryzen optimization (6 cores) |
| `LLAMA_CPP_F16_KV` | `.env` | Memory saving (~50% KV cache) |
| `LLM_MODEL_PATH` | `.env` | Path to Gemma-3 model |
| `EMBEDDING_MODEL_PATH` | `.env` | Path to all-MiniLM model |
| `LIBRARY_PATH` | `.env` | Curated documents directory |
| `KNOWLEDGE_PATH` | `.env` | Agent knowledge metadata |

**Validation**:

```bash
# Verify compose syntax
sudo docker compose config > /dev/null && echo "Syntax OK"

# Verify no `version:` key (Compose v2 requirement)
grep -q "^version:" docker-compose.yml && echo "ERROR: Remove version key" || echo "OK"

# Verify health check dependencies
sudo docker compose config | grep -A 3 "depends_on:" | grep "condition: service_healthy"
# Expected: 3 matches (ui→rag, rag→redis, crawler→rag)
```

---

## Section 13.1: Pre-Deployment Checklist

**Purpose**: Validate environment BEFORE building/deploying (prevents deployment failures)

**Estimated Time**: 5 minutes

### Step 1: Configuration Validation (197 variables + 23 sections)

```bash
# Validate .env (197 variables, 8 telemetry disables)
python3 scripts/validate_config.py

# Expected output:
# âœ" Environment variables: 197/197 found
# âœ" Telemetry disables: 8/8 enabled
# âœ" Ryzen optimizations: LLAMA_CPP_N_THREADS=6, OPENBLAS_CORETYPE=ZEN
# âœ" Critical paths resolved: LLM_MODEL_PATH, EMBEDDING_MODEL_PATH
# Result: 16 checks passed âœ"
```

**Common Issues**:

| Issue | Symptom | Fix |
|-------|---------|-----|
| `REDIS_PASSWORD` not set | `ValueError: REDIS_PASSWORD not found` | Add to `.env`: `REDIS_PASSWORD=<16-char-min>` |
| Model paths incorrect | `FileNotFoundError: /models/*.gguf` | Verify: `ls -lh models/*.gguf embeddings/*.gguf` |
| UID/GID mismatch | Permission denied on volumes | Set in `.env`: `APP_UID=$(id -u)`, `APP_GID=$(id -g)` |

### Step 2: Directory Permissions

```bash
# Create directories with correct ownership (UID 1001 for appuser, 999 for Redis)
sudo mkdir -p \
  library/classical-works \
  library/physics \
  library/psychology \
  library/technical-manuals \
  library/esoteric \
  knowledge/curator \
  knowledge/coder \
  knowledge/editor \
  knowledge/manager \
  knowledge/learner \
  data/redis \
  data/faiss_index \
  data/cache \
  data/prometheus-multiproc \
  backups

# Set ownership
sudo chown -R 1001:1001 library knowledge data/faiss_index data/cache backups data/prometheus-multiproc
sudo chown -R 999:999 data/redis

# Verify
ls -la library knowledge data/ | grep -E "1001|999"
# Expected:
# drwxr-xr-x ... 1001 1001 ... library/
# drwxr-xr-x ... 1001 1001 ... knowledge/
# drwxr-xr-x ...  999  999 ... data/redis/
```

### Step 3: Model File Verification

```bash
# Verify model files downloaded and accessible
ls -lh models/*.gguf embeddings/*.gguf

# Expected:
# -rw-r--r-- ... 2.8G ... gemma-3-4b-it-UD-Q5_K_XL.gguf
# -rw-r--r-- ...  45M ... all-MiniLM-L12-v2.Q8_0.gguf

# Test model loading (optional, 10s)
python3 -c "
import sys
sys.path.insert(0, 'app/XNAi_rag_app')
from llama_cpp import Llama
llm = Llama(model_path='models/gemma-3-4b-it-UD-Q5_K_XL.gguf', n_ctx=512, verbose=False)
print('✓ LLM loads successfully')
"
```

### Step 4: Port Availability Check

```bash
# Check if required ports are available
for port in 8000 8001 6379 8002; do
  if sudo lsof -i :$port 2>/dev/null; then
    echo "⚠️  Port $port in use - stop service or change port"
  else
    echo "✓ Port $port available"
  fi
done

# Expected: All 4 ports available
```

### Step 5: Docker Version Verification

```bash
# Verify Docker version (27.3.1+)
docker version --format '{{.Server.Version}}'
# Expected: 27.3.1 or higher

# Verify Compose v2 (v2.29.2+, NO version key support)
docker compose version
# Expected: Docker Compose version v2.29.2 (or higher)

# CRITICAL: Check for old docker-compose (v1) - should NOT exist
which docker-compose 2>/dev/null && echo "⚠️  Remove old docker-compose v1" || echo "✓ No conflicting v1 install"
```

### Step 6: Disk Space Check

```bash
# Verify sufficient disk space (~10GB minimum)
df -h / | awk 'NR==2 {print "Available:", $4}'
# Expected: >10GB available

# Breakdown:
# - Docker images: ~2GB (all 4 services)
# - Models: ~3GB (LLM + embeddings)
# - FAISS index: ~1GB (10,000 vectors)
# - Redis: ~500MB
# - Logs/cache: ~1GB
# - Buffer: ~2.5GB
```

### Step 7: Pre-Build Image Cleanup (Optional)

```bash
# Remove old images to save space (only if rebuilding)
sudo docker image prune -a --filter "label=version=v0.1.2" --force

# Remove dangling images
sudo docker image prune --force

# Verify cleanup
sudo docker images | grep xnai
# Expected: Empty (if first deployment) or old versions removed
```

---

## Section 13.2: Deployment Execution

**Purpose**: Build and start all services with health check verification

**Estimated Time**: 3-5 minutes (depending on hardware)

### Step 1: Build Images (No Cache)

```bash
# Build all images from scratch (ensures clean build with v0.1.3 fixes)
sudo docker compose build --no-cache

# Expected output (watch for these):
# [+] Building 120.5s (24/24) FINISHED
# => [rag builder 1/8] FROM python:3.12-slim
# => [rag builder 8/8] RUN pip install --no-cache-dir --user -r requirements-api.txt
# => [rag stage-1 1/10] FROM python:3.12-slim
# => [rag stage-1 10/10] COPY --chown=appuser:appgroup app/XNAi_rag_app /app/XNAi_rag_app
# => => naming to docker.io/library/xnai-stack-rag
# Successfully tagged xnai-stack-rag:latest

# Repeat for ui, crawler services
```

**Build Time Breakdown** (Ryzen 7 5700U):
- `rag`: ~2-3 min (llama-cpp-python compilation)
- `ui`: ~1 min (Chainlit dependencies)
- `crawler`: ~2-3 min (Playwright browser download)
- Total: ~5-7 min

### Step 2: Verify Images Created

```bash
# Check images exist with correct tags
sudo docker images | grep xnai

# Expected:
# xnai-stack-rag        latest    ...    450MB    ...
# xnai-stack-ui         latest    ...    300MB    ...
# xnai-stack-crawler    latest    ...    600MB    ...
# redis                 7.4.1     ...     40MB    ...
```

### Step 3: Start Services (Background Mode)

```bash
# Start all services in detached mode
sudo docker compose up -d

# Expected output:
# [+] Running 4/4
#  ✔ Container xnai_redis         Created                 2.1s
#  ✔ Container xnai_rag_api       Created                 0.5s
#  ✔ Container xnai_chainlit_ui   Created                 0.3s
#  ✔ Container xnai_crawler       Created                 0.2s

# Note: Services will show "Created" then move to "Started" status
```

### Step 4: Monitor Startup Progress

```bash
# Watch service status (all should reach "Up (healthy)" within 90s)
watch -n 5 'sudo docker compose ps'

# Expected progression:
# t=0s:   All "Up" (not yet healthy)
# t=30s:  redis "Up (healthy)", others still starting
# t=60s:  rag "Up (healthy)" (LLM loaded)
# t=90s:  ui, crawler "Up (healthy)" (all services ready)

# OR use logs to watch health checks:
sudo docker compose logs -f rag | grep "HEALTHCHECK"
```

### Step 5: Verify All Services Healthy

```bash
# Check final status (after 90s wait)
sudo docker compose ps

# Expected (ALL must show "healthy"):
# NAME               IMAGE              STATUS
# xnai_redis         redis:7.4.1       Up (healthy)
# xnai_rag_api       xnai-stack-rag    Up (healthy)
# xnai_chainlit_ui   xnai-stack-ui     Up (healthy)
# xnai_crawler       xnai-stack-crawler Up (healthy)
```

**If Any Service Unhealthy**:

```bash
# Check specific service logs
sudo docker compose logs <service_name> -n 50

# Common issues:
# - Redis: Password mismatch → Check REDIS_PASSWORD in all services
# - RAG: LLM load failure → Verify model path, check memory
# - UI: API connection refused → Verify rag service healthy first
# - Crawler: Import error → Check crawl4ai version in requirements
```

---

## Section 13.3: Post-Deployment Validation

**Purpose**: Comprehensive validation that stack is production-ready

**Estimated Time**: 3 minutes

### Validation 1: Health Endpoints (7 Targets)

```bash
# Query health endpoint (returns JSON with 7 component checks)
curl -s http://localhost:8000/health | jq

# Expected output:
{
  "status": "healthy",
  "version": "v0.1.3-beta",
  "timestamp": "2025-10-22T14:32:15Z",
  "memory_gb": 4.2,
  "components": {
    "llm": true,
    "embeddings": true,
    "vectorstore": true,
    "redis": true,
    "crawler": true,
    "ryzen": true,
    "memory": true
  },
  "details": {
    "llm": "Gemma-3 4B (Q5_K_XL) loaded",
    "embeddings": "all-MiniLM-L12-v2 (Q8_0) loaded, 384 dims",
    "vectorstore": "FAISS index: 0 vectors (empty - expected on first run)",
    "redis": "Redis 7.4.1 - PING OK (2ms)",
    "crawler": "crawl4ai 0.7.3 initialized",
    "ryzen": "LLAMA_CPP_N_THREADS=6, OPENBLAS_CORETYPE=ZEN, F16_KV=true",
    "memory": "4.2GB / 6.0GB (70% utilized)"
  }
}

# All 7 components must be "true" ✓
```

### Validation 2: API Query Test (Without RAG)

```bash
# Test basic LLM query (no RAG context)
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Xoe-NovAi? Answer in one sentence.",
    "use_rag": false,
    "max_tokens": 100
  }' | jq

# Expected output:
{
  "response": "Xoe-NovAi is a local-first, privacy-preserving AI stack for CPU-optimized RAG and multi-agent workflows.",
  "tokens_generated": 23,
  "processing_time_ms": 847,
  "model": "gemma-3-4b-it-UD-Q5_K_XL",
  "rag_context": null,
  "rag_sources": [],
  "cache_hit": false
}

# Response should be coherent and complete ✓
```

### Validation 3: Chainlit UI Access

```bash
# Test Chainlit UI is accessible
curl -s http://localhost:8001 | grep -q "<!DOCTYPE" && echo "✓ UI loads" || echo "✗ UI failed"

# Open in browser (manual verification)
# Navigate to: http://localhost:8001
# Expected: Chainlit chat interface with welcome message
```

### Validation 4: Metrics Endpoint

```bash
# Check Prometheus metrics available
curl -s http://localhost:8002/metrics | grep "xnai_" | head -10

# Expected (sample output):
# xnai_memory_usage_gb{component="system"} 4.2
# xnai_memory_usage_gb{component="process"} 3.8
# xnai_token_rate_tps{model="gemma-3-4b-it"} 20.5
# xnai_active_sessions 0
# xnai_requests_total{endpoint="/query",method="POST",status="200"} 1
# xnai_response_latency_ms_bucket{le="100"} 0
# xnai_response_latency_ms_bucket{le="250"} 0
# xnai_response_latency_ms_bucket{le="500"} 0
# xnai_response_latency_ms_bucket{le="1000"} 1
# xnai_tokens_generated_total{model="gemma-3-4b-it"} 23
```

### Validation 5: Memory Usage Check

```bash
# Check memory usage is within target (<6GB for stack)
sudo docker stats --no-stream xnai_rag_api xnai_chainlit_ui xnai_crawler

# Expected output:
# CONTAINER            CPU %    MEM USAGE / LIMIT
# xnai_rag_api        45-65%   3.8GB / 6.0GB      ✓ (within limit)
# xnai_chainlit_ui     5-10%   0.5GB / 6.0GB      ✓
# xnai_crawler         2-5%    0.3GB / 6.0GB      ✓
# Total:                       ~4.6GB             ✓ (target: <6GB)
```

### Validation 6: Log Inspection

```bash
# Check for critical errors in logs (allow warnings, but no errors)
sudo docker compose logs rag | grep -i "error" | grep -v "expected\|startup" | wc -l
# Expected: 0 (or minimal startup warnings only)

sudo docker compose logs ui | grep -i "error" | wc -l
# Expected: 0

sudo docker compose logs crawler | grep -i "error" | wc -l
# Expected: 0
```

### Validation 7: Performance Benchmark

```bash
# Run token rate benchmark (target: 15-25 tok/s)
sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/scripts/query_test.py --queries 5

# Expected output:
# Query 1: 45 tokens in 2.1s = 21.4 tok/s ✓
# Query 2: 52 tokens in 2.4s = 21.7 tok/s ✓
# Query 3: 48 tokens in 2.3s = 20.9 tok/s ✓
# Query 4: 51 tokens in 2.5s = 20.4 tok/s ✓
# Query 5: 49 tokens in 2.2s = 22.3 tok/s ✓
# Mean: 21.3 tok/s (target: 15-25 tok/s) ✓
# p95 latency: 920ms ✓ (target: <1000ms)
```

---

## Deployment Success Criteria Checklist

**All must be ✓ before proceeding to production use:**

- [ ] Configuration validation passed (197/197 vars, 16/16 checks)
- [ ] All 4 services status "Up (healthy)" in `docker compose ps`
- [ ] Health endpoint returns 7/7 components true
- [ ] Query test returns coherent response
- [ ] Chainlit UI loads in browser
- [ ] Metrics endpoint returns Prometheus data
- [ ] Memory usage <6GB total
- [ ] Token rate 15-25 tok/s
- [ ] No critical errors in logs
- [ ] Disk space >5GB remaining

**If Any Item Fails**: See Section 13.4 (Troubleshooting) in Artifact 5

---

## Quick Reference: Deployment Commands

```bash
# Complete deployment sequence (copy-paste friendly)
# ============================================================

# 1. Pre-deployment validation
python3 scripts/validate_config.py
ls -lh models/*.gguf embeddings/*.gguf
for port in 8000 8001 6379 8002; do sudo lsof -i :$port 2>/dev/null || echo "✓ Port $port available"; done

# 2. Build images
sudo docker compose build --no-cache

# 3. Start services
sudo docker compose up -d

# 4. Wait for health checks (90 seconds)
sleep 90

# 5. Verify all healthy
sudo docker compose ps | grep "healthy"
# Expected: 4 matches (all services)

# 6. Test query
curl -s http://localhost:8000/health | jq '.components | map(select(. == false))'
# Expected: [] (empty array = all true)

# 7. Access UI
open http://localhost:8001
```

---

## Cross-References

- **Configuration**: Artifact 3 (.env template, config.toml structure)
- **Health Checks**: Artifact 5 (7-target validation, metrics)
- **Troubleshooting**: Artifact 5 Section 13.4 (common issues table)
- **Prerequisites**: Artifact 2 (hardware requirements, Docker installation)

---

## End of Artifact 4

**Sections Covered**: 7 (Docker), 13.1-13.3 (Deployment)  
**Token Count**: ~11,800  
**Next**: Artifact 5 (Health Monitoring + Troubleshooting)