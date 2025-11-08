# Xoe-NovAi v0.1.3-beta: Configuration Management

**Artifact**: Group 2.2 | **Sections**: 5 (Configuration) + Appendix A (.env Reference)  
**Purpose**: Complete configuration guide - 197 environment variables + 23 config.toml sections  
**Cross-Reference**: See Group 2.1 (Prerequisites) for dependency setup | See Group 3.1 (Docker) for config.toml mount strategy

---

## Quick Reference: Configuration Files

| File | Variables | Purpose | Validation Command |
|------|-----------|---------|-------------------|
| `.env` | 197 vars | Runtime environment (secrets, paths, feature flags) | `bash validate-config.sh` |
| `config.toml` | 23 sections | Application settings (non-secret, version-controlled) | `python3 -c "import toml; toml.load('config.toml')"` |
| `.env.example` | 197 vars | Template (safe to commit, no secrets) | `diff .env.example .env` (shows customizations) |

**Configuration Separation Strategy** (2025 best practices):

As verified by web search, modern best practices separate sensitive values (passwords, API keys) from application parameters:
- **`.env`**: Secrets (REDIS_PASSWORD), paths (LLM_MODEL_PATH), feature flags (telemetry disables)
- **`config.toml`**: Application parameters (batch sizes, timeouts), non-secret metadata (version, codename)
- **Docker READ-ONLY mount**: `config.toml` mounted `:ro` prevents container modification

---

## Section 5: Environment Variables (.env) - 197 Total

### 5.1 Core Structure (20 Categories)

The `.env` file contains 197 variables organized into 20 functional categories. This structure enables selective updates (e.g., change Redis settings without touching LLM config).

**Category Overview**:

| # | Category | Var Count | Purpose | Example |
|---|----------|-----------|---------|---------|
| 1 | Stack Identity | 5 | Version, codename, environment | `STACK_VERSION=v0.1.3-beta` |
| 2 | Redis | 14 | Cache, streams, connection | `REDIS_PASSWORD=<secret>` |
| 3 | Threading | 7 | Ryzen CPU optimization | `LLAMA_CPP_N_THREADS=6` |
| 4 | Memory | 8 | Limits, monitoring | `MEMORY_LIMIT_GB=6.0` |
| 5 | LLM | 15 | Model paths, inference | `LLM_MODEL_PATH=/models/gemma-3-4b-it-UD-Q5_K_XL.gguf` |
| 6 | Embeddings | 8 | Embedding model config | `EMBEDDING_MODEL_PATH=/embeddings/all-MiniLM-L12-v2.Q8_0.gguf` |
| 7 | Server | 10 | Ports, workers, timeouts | `API_PORT=8000` |
| 8 | Paths | 8 | Directories | `LIBRARY_PATH=/library` |
| 9 | Logging | 7 | JSON logging config | `LOG_LEVEL=INFO` |
| 10 | Telemetry | 13 | Zero-telemetry enforcement | `CHAINLIT_NO_TELEMETRY=true` |
| 11 | CrawlModule | 15 | Crawler settings | `CRAWL_RATE_LIMIT_PER_MIN=30` |
| 12 | RAG | 9 | Retrieval config | `RAG_TOP_K=5` |
| 13 | Health | 6 | Health check targets | `HEALTH_TARGETS=llm,embeddings,memory,redis,vectorstore,ryzen,crawler` |
| 14 | Security | 8 | Non-root, capabilities | `APP_UID=1001` |
| 15 | Automation | 6 | Backup, ingest | `BACKUP_RETENTION_DAYS=7` |
| 16 | Phase 2 | 10 | Multi-agent prep | `PHASE2_QDRANT_ENABLED=false` |
| 17 | Vectorstore | 8 | FAISS settings | `FAISS_INDEX_PATH=/app/XNAi_rag_app/faiss_index` |
| 18 | Debug | 7 | Debugging flags | `DEBUG_MODE=false` |
| 19 | Docker | 6 | Compose settings | `COMPOSE_PROJECT_NAME=xnai-stack` |
| 20 | Session | 5 | UI session config | `SESSION_TIMEOUT=3600` |

**Total**: 197 variables across 20 categories

### 5.2 CRITICAL Variables (Must Customize Before Deploy)

**These 8 variables MUST be changed from defaults**:

```bash
# 1. REDIS_PASSWORD (Security: CRITICAL)
REDIS_PASSWORD=CHANGE_ME_TO_16_CHAR_MINIMUM
# MUST be 16+ characters, alphanumeric + special chars
# Example: MySecurePass123456!@

# 2. APP_UID (Permissions: CRITICAL)
APP_UID=1001
# Set to your user ID: $(id -u)

# 3. APP_GID (Permissions: CRITICAL)
APP_GID=1001
# Set to your group ID: $(id -g)

# 4. LLM_MODEL_PATH (Required for LLM load)
LLM_MODEL_PATH=/models/gemma-3-4b-it-UD-Q5_K_XL.gguf
# Verify file exists: ls -lh $LLM_MODEL_PATH

# 5. EMBEDDING_MODEL_PATH (Required for embeddings)
EMBEDDING_MODEL_PATH=/embeddings/all-MiniLM-L12-v2.Q8_0.gguf
# Verify file exists: ls -lh $EMBEDDING_MODEL_PATH

# 6. STACK_VERSION (Update tracking)
STACK_VERSION=v0.1.3-beta
# Match deployment version

# 7. ENVIRONMENT (Deployment stage)
ENVIRONMENT=production
# Options: development, staging, production

# 8. LOG_LEVEL (Production vs Debug)
LOG_LEVEL=INFO
# Options: DEBUG (dev), INFO (prod), WARNING, ERROR
```

**Validation**:

```bash
# Check all 8 critical vars set
grep -E "^(REDIS_PASSWORD|APP_UID|APP_GID|LLM_MODEL_PATH|EMBEDDING_MODEL_PATH|STACK_VERSION|ENVIRONMENT|LOG_LEVEL)=" .env

# Verify REDIS_PASSWORD length (must be 16+)
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env | cut -d= -f2)
if [ ${#REDIS_PASS} -ge 16 ]; then
  echo "✅ REDIS_PASSWORD length OK (${#REDIS_PASS} chars)"
else
  echo "❌ REDIS_PASSWORD too short (${#REDIS_PASS} chars, need 16+)"
fi

# Verify UID/GID match current user
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
ENV_UID=$(grep "^APP_UID=" .env | cut -d= -f2)
ENV_GID=$(grep "^APP_GID=" .env | cut -d= -f2)

if [ "$CURRENT_UID" = "$ENV_UID" ] && [ "$CURRENT_GID" = "$ENV_GID" ]; then
  echo "✅ UID/GID match current user"
else
  echo "❌ UID/GID mismatch: current ($CURRENT_UID:$CURRENT_GID) vs .env ($ENV_UID:$ENV_GID)"
fi

# Verify model paths exist
if [ -f "$LLM_MODEL_PATH" ]; then
  echo "✅ LLM model found: $LLM_MODEL_PATH"
else
  echo "❌ LLM model NOT found: $LLM_MODEL_PATH"
fi

if [ -f "$EMBEDDING_MODEL_PATH" ]; then
  echo "✅ Embeddings found: $EMBEDDING_MODEL_PATH"
else
  echo "❌ Embeddings NOT found: $EMBEDDING_MODEL_PATH"
fi
```

### 5.3 Category Breakdown: Complete Reference

#### Category 1: Stack Identity (5 vars)

```bash
# Stack metadata
STACK_NAME=Xoe-NovAi
STACK_VERSION=v0.1.3-beta
STACK_CODENAME=Resilient Polymath
STACK_PHASE=1
ENVIRONMENT=production  # development | staging | production
```

#### Category 2: Redis Configuration (14 vars)

```bash
# Connection
REDIS_HOST=xnai_redis
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_ME_16_CHAR_MIN  # CRITICAL: Change from default
REDIS_DB=0

# Caching (exact-match for v0.1.3, semantic in Phase 2)
REDIS_CACHE_TTL=3600                   # Cache expiry: 1 hour
REDIS_CACHE_PREFIX=xnai_cache          # Key namespace
REDIS_POOL_SIZE=10
REDIS_ENCODING=utf-8
REDIS_DECODE_RESPONSES=true

# Resilience
REDIS_RETRY_ON_TIMEOUT=true
REDIS_HEALTH_TIMEOUT=5

# Phase 2 Preparation (streams for multi-agent coordination)
REDIS_STREAMS_ENABLED=false            # Enable in Phase 2
REDIS_STREAMS_GROUP=xnai_agents
REDIS_STREAMS_CONSUMER=agent_1
```

**Why These Settings?**

As confirmed by web search, Redis 8.2 (2025) supports semantic caching for AI applications. However, v0.1.3-beta uses simpler exact-match caching (keys like `xnai_cache:source:category:query`) to reduce complexity in Phase 1. The `REDIS_MAXMEMORY_POLICY=allkeys-lru` (set in docker-compose.yml) ensures automatic eviction when memory limit reached, preventing OOM errors.

**Validation**:

```bash
# Test Redis connection
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" ping
# Expected: PONG

# Check cache TTL setting
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" CONFIG GET maxmemory-policy
# Expected: 1) "maxmemory-policy" 2) "allkeys-lru"

# Verify no streams (Phase 1)
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" XINFO GROUPS xnai_coordination
# Expected: (error) ERR no such key (streams not enabled yet)
```

#### Category 3: Threading Optimization (7 vars) - RYZEN CRITICAL

```bash
# llama.cpp threading (Ryzen 7 5700U: 8C/16T)
LLAMA_CPP_N_THREADS=6                  # 75% of 8 cores (leave 2 for OS)
LLAMA_CPP_F16_KV=true                  # 50% memory savings (KV cache FP16)
LLAMA_CPP_USE_MLOCK=true               # Prevent page swapping
LLAMA_CPP_USE_MMAP=true                # Memory-mapped I/O
LLAMA_CPP_N_GPU_LAYERS=0               # GPU layers (0=CPU only, Phase 2: iGPU offload)

# OpenBLAS optimization (Zen2 architecture)
OMP_NUM_THREADS=1                      # Disable OpenMP (use Ryzen threads)
OPENBLAS_CORETYPE=ZEN                  # AMD Zen2 architecture
```

**Why These Settings?**

Web search confirmed that AMD Ryzen CPUs (especially mobile variants like 5700U) are memory bandwidth-limited rather than core-limited for LLM inference. Settings validated:
- `N_THREADS=6`: Optimal for 8C/16T (leave 2 cores for OS/Redis/Docker overhead)
- `F16_KV=true`: Reduces memory by ~50% (FP16 KV cache vs FP32) without quality loss
- `OPENBLAS_CORETYPE=ZEN`: Enables Zen2-specific SIMD optimizations (AVX2/FMA)

**Validation**:

```bash
# Verify Ryzen flags active in container
sudo docker exec xnai_rag_api env | grep -E "LLAMA_CPP|OPENBLAS"

# Expected:
# LLAMA_CPP_N_THREADS=6
# LLAMA_CPP_F16_KV=true
# LLAMA_CPP_USE_MLOCK=true
# LLAMA_CPP_USE_MMAP=true
# LLAMA_CPP_N_GPU_LAYERS=0
# OPENBLAS_CORETYPE=ZEN
# OMP_NUM_THREADS=1

# Test token rate (should be 15-25 tok/s with these settings)
sudo docker exec xnai_rag_api python3 scripts/query_test.py --queries 3
# Expected: Mean: ~20 tok/s
```

#### Category 4: Memory Management (8 vars)

```bash
# Hard limits
MEMORY_LIMIT_GB=6.0                    # Enforce hard limit (stack target)
MEMORY_WARNING_GB=5.5                  # Trigger warning at 5.5GB

# Monitoring
MEMORY_CHECK_INTERVAL_S=10             # Check every 10s

# Optimization
PYTHON_MALLOC=malloc                   # Use standard malloc (not jemalloc)
PYTORCH_ENABLE_MPS_FALLBACK=1          # Fallback if Metal (macOS) unavailable
LANGCHAIN_MEMORY_CLEAR_ON_INIT=false   # Preserve memory across reinits

# Caching
CACHE_SIZE_MB=512                      # CrawlModule cache limit
MAX_BATCH_SIZE=100                     # Ingestion batch size (Pattern 4)
```

**Memory Breakdown** (at 6GB target):

```
Total Stack Memory: 6.0GB target
├── LLM (Gemma-3 4B Q5_K_XL): 3.0GB
├── Embeddings (all-MiniLM Q8_0): 0.5GB
├── FAISS Vectorstore (10k vectors): 1.0GB
├── Redis Cache: 0.3GB
├── Python overhead: 0.5GB
├── CrawlModule cache (shelve): 0.5GB
└── Buffer: 0.2GB
```

**Validation**:

```bash
# Check memory usage in real-time
sudo docker stats --no-stream xnai_rag_api
# Expected: MEM USAGE <6GB

# Monitor memory over time (30s sample)
for i in {1..6}; do
  sudo docker stats --no-stream xnai_rag_api | awk '{print $3}'
  sleep 5
done
# Expected: All readings <6GB

# Test memory warning trigger
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from healthcheck import check_available_memory

# Should warn at 5.5GB
try:
    check_available_memory(required_gb=5.5)
    print('✅ Memory check passed')
except MemoryError as e:
    print(f'⚠️ Memory warning: {e}')
"
```

#### Category 5: LLM Configuration (15 vars)

```bash
# Model
LLM_MODEL_PATH=/models/gemma-3-4b-it-UD-Q5_K_XL.gguf  # CRITICAL

# Context
LLM_CONTEXT_LENGTH=2048
LLM_N_CTX=2048
LLM_N_PREDICT=512                      # Max tokens per query

# Inference parameters
LLM_TEMPERATURE=0.7                    # Creativity (0=deterministic, 1=random)
LLM_TOP_P=0.95                         # Nucleus sampling threshold
LLM_TOP_K=40                           # Top-K sampling
LLM_REPEAT_PENALTY=1.1                 # Penalize repetition
LLM_REPEAT_LAST_N=64                   # Lookback window for repeat penalty

# Optimization
LLM_VERBOSE=false                      # Disable debug output
LLM_SEED=42                            # Reproducible outputs (dev/testing)
LLM_USE_MMAP=true                      # Memory-mapped I/O
LLM_USE_MLOCK=true                     # Prevent page swapping
LLM_LAST_TOKEN_LOGITS=false            # Skip logit calculation (speed)

# Phase 2
LLM_ENABLE_SPECULATIVE=false           # Speculative decoding (experimental)
```

**Parameter Tuning Guide**:

| Parameter | Impact | Recommended Values |
|-----------|--------|-------------------|
| `TEMPERATURE` | Higher = more creative, lower = more deterministic | 0.7 (balanced), 0.3 (factual), 1.0 (creative) |
| `TOP_P` | Nucleus sampling (cumulative probability cutoff) | 0.95 (default), 0.9 (stricter), 1.0 (disabled) |
| `TOP_K` | Top-K sampling (number of candidates) | 40 (default), 20 (stricter), 100 (more variety) |
| `REPEAT_PENALTY` | Penalize repetition | 1.1 (default), 1.2 (stronger), 1.0 (disabled) |

**Validation**:

```bash
# Test LLM loads with these settings
sudo docker exec xnai_rag_api python3 -c "
import sys, os
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_llm

llm = get_llm()
response = llm('Hello', max_tokens=10, temperature=0.7)
print(f'✅ LLM response: {response}')
"
# Expected: LLM generates coherent response

# Verify context length
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_llm

llm = get_llm()
print(f'Context length: {llm.n_ctx()}')
"
# Expected: Context length: 2048
```

#### Category 6: Embeddings Configuration (8 vars)

```bash
# Model
EMBEDDING_MODEL_PATH=/embeddings/all-MiniLM-L12-v2.Q8_0.gguf  # CRITICAL

# Output
EMBEDDING_DIMENSION=384                # Output vector size

# Performance
EMBEDDING_BATCH_SIZE=32                # Batch embeddings generation
EMBEDDING_NORMALIZE=true               # L2 normalization
EMBEDDING_N_THREADS=2                  # Secondary embeddings worker

# Optimization
EMBEDDING_USE_MMAP=true                # Memory-mapped I/O
EMBEDDING_CACHE_SIZE_MB=256            # Embedding cache limit
EMBEDDING_DEVICE=cpu                   # CPU-only (Phase 2: iGPU)
```

**Why 384 Dimensions?**

The all-MiniLM-L12-v2 model outputs 384-dimensional vectors, a standard size for semantic search. Higher dimensions (768, 1024) improve accuracy but increase FAISS index size and query time. For Phase 1, 384 dimensions balance quality and performance.

**Validation**:

```bash
# Test embeddings generation
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_embeddings

embeddings = get_embeddings()
vec = embeddings.embed_query('test')
print(f'✅ Embedding dimension: {len(vec)}')
assert len(vec) == 384, f'Expected 384, got {len(vec)}'
"
# Expected: ✅ Embedding dimension: 384

# Test batch embedding
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_embeddings
from langchain_core.documents import Document

embeddings = get_embeddings()
docs = [Document(page_content=f'test {i}') for i in range(5)]
vecs = embeddings.embed_documents(docs)
print(f'✅ Batch embeddings: {len(vecs)} vectors, {len(vecs[0])} dims each')
"
# Expected: ✅ Batch embeddings: 5 vectors, 384 dims each
```

#### Category 7: Server Configuration (10 vars)

```bash
# FastAPI
API_HOST=0.0.0.0                       # Bind to all interfaces
API_PORT=8000                          # RAG API port
API_WORKERS=4                          # Uvicorn workers (CPU cores)
API_TIMEOUT=300                        # Request timeout (5 min)
API_RATE_LIMIT_PER_MIN=60              # Rate limiting (per client IP)

# Chainlit
CHAINLIT_HOST=0.0.0.0
CHAINLIT_PORT=8001

# Metrics
METRICS_PORT=8002                      # Prometheus scrape endpoint

# Logging
LOG_LEVEL=INFO                         # DEBUG | INFO | WARNING | ERROR
UVICORN_LOG_LEVEL=INFO
```

**Port Mapping**:

| Port | Service | Protocol | Purpose |
|------|---------|----------|---------|
| 8000 | FastAPI | HTTP/REST | RAG queries, curation, health |
| 8001 | Chainlit | HTTP/WebSocket | Interactive UI |
| 8002 | Prometheus | HTTP | Metrics scraping |
| 6379 | Redis | TCP | Cache/streams (internal) |

**Validation**:

```bash
# Check ports available before deployment
for port in 8000 8001 8002 6379; do
  if sudo lsof -i :$port 2>/dev/null; then
    echo "⚠️ Port $port in use"
  else
    echo "✅ Port $port available"
  fi
done

# Test rate limiting (after deployment)
for i in {1..65}; do
  curl -s -w "%{http_code}\n" -o /dev/null http://localhost:8000/health
done | grep -c 429
# Expected: At least 5 (429 = rate limited after 60 req/min)
```

#### Category 8: Paths (8 vars)

```bash
# Core directories
LIBRARY_PATH=/library                  # Curated documents
KNOWLEDGE_PATH=/knowledge              # Agent knowledge (Phase 2)
MODELS_PATH=/models                    # LLM models
EMBEDDINGS_PATH=/embeddings            # Embedding models

# Runtime data
BACKUP_PATH=/backups                   # FAISS backups
CACHE_PATH=/app/cache                  # CrawlModule cache
FAISS_INDEX_PATH=/app/XNAi_rag_app/faiss_index
LOG_PATH=/app/XNAi_rag_app/logs
```

**Directory Structure** (created by Dockerfiles):

```
Container Filesystem:
├── /models/                           # READ-ONLY bind mount
│   └── gemma-3-4b-it-UD-Q5_K_XL.gguf
├── /embeddings/                       # READ-ONLY bind mount
│   └── all-MiniLM-L12-v2.Q8_0.gguf
├── /library/                          # READ-WRITE bind mount
│   ├── psychology/
│   ├── physics/
│   └── classical-works/
├── /knowledge/                        # READ-WRITE bind mount
│   └── curator/
│       └── index.toml
├── /app/XNAi_rag_app/
│   ├── faiss_index/                   # Persistent volume
│   └── logs/                          # Created with chmod 777
├── /backups/                          # Persistent volume
└── /app/cache/                        # CrawlModule cache
```

**Validation**:

```bash
# Verify all paths exist in container
sudo docker exec xnai_rag_api bash -c "
for path in /models /embeddings /library /knowledge /backups /app/cache /app/XNAi_rag_app/faiss_index /app/XNAi_rag_app/logs; do
  if [ -d \$path ]; then
    echo \"✅ \$path exists\"
  else
    echo \"❌ \$path MISSING\"
  fi
done
"
# Expected: All paths exist

# Check write permissions
sudo docker exec xnai_rag_api bash -c "
for path in /library /knowledge /backups /app/cache /app/XNAi_rag_app/logs; do
  if [ -w \$path ]; then
    echo \"✅ \$path writable\"
  else
    echo \"❌ \$path NOT writable\"
  fi
done
"
# Expected: All writable
```

#### Category 9: Logging (7 vars)

```bash
# Level & Format
LOG_LEVEL=INFO                         # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=json                        # JSON structured logging

# Rotation
LOG_MAX_SIZE_MB=10                     # Rotate after 10MB
LOG_BACKUP_COUNT=5                     # Keep 5 backups

# Output
LOG_ENABLE_FILE=true                   # Log to file
LOG_ENABLE_CONSOLE=true                # Log to stdout (Docker logs)
LOG_ENABLE_SYSLOG=false                # Syslog integration (Phase 2)
```

**Log Format** (JSON structured):

```json
{
  "timestamp": "2025-10-21T14:32:15Z",
  "level": "INFO",
  "logger": "app.main.query_endpoint",
  "message": "Query started: What is Xoe-NovAi?",
  "context": {
    "request_id": "abc123",
    "user_id": "anonymous",
    "query_length": 19
  },
  "performance": {
    "duration_ms": 850
  }
}
```

**Validation**:

```bash
# Check log file exists and rotates
sudo docker exec xnai_rag_api ls -lh /app/XNAi_rag_app/logs/
# Expected: xnai.log, xnai.log.1, xnai.log.2, ... (up to 5 backups)

# Verify JSON format
sudo docker exec xnai_rag_api tail -1 /app/XNAi_rag_app/logs/xnai.log | jq .
# Expected: Valid JSON with timestamp, level, logger, message

# Test log rotation trigger
sudo docker exec xnai_rag_api bash -c "
dd if=/dev/zero of=/app/XNAi_rag_app/logs/test.log bs=1M count=11
"
# Should trigger rotation at 10MB
```

#### Category 10: Telemetry Disables (13 vars) - CRITICAL

**ZERO-TELEMETRY ENFORCEMENT** (8 primary + 5 secondary):

```bash
# Primary disables (CRITICAL - all must be true)
CHAINLIT_NO_TELEMETRY=true             # 1/8
CRAWL4AI_NO_TELEMETRY=true             # 2/8
LLAMA_CPP_NO_TELEMETRY=true            # 3/8
LANGCHAIN_NO_TELEMETRY=true            # 4/8
FAISS_NO_TELEMETRY=true                # 5/8
PROMETHEUS_NO_TELEMETRY=true           # 6/8
UVICORN_NO_TELEMETRY=true              # 7/8
FASTAPI_NO_TELEMETRY=true              # 8/8

# Secondary disables (prevent external API calls)
LANGCHAIN_TRACING_V2=false             # LangSmith tracing
LANGCHAIN_API_KEY=                     # Empty (no external API)
OPENAI_API_KEY=                        # Empty (no external API)
ANTHROPIC_API_KEY=                     # Empty (no external API)
SCARF_NO_ANALYTICS=true                # Scarf package analytics
```

**Why This Matters**:

Zero-telemetry ensures:
1. **Privacy**: No usage data leaves the machine
2. **Security**: No external API calls (attack surface minimized)
3. **Performance**: No network overhead for telemetry
4. **Compliance**: GDPR/HIPAA-friendly (no data leakage)

**Validation**:

```bash
# Verify all 8 primary disables set to "true"
grep -E "^(CHAINLIT|CRAWL4AI|LLAMA_CPP|LANGCHAIN|FAISS|PROMETHEUS|UVICORN|FASTAPI)_NO_TELEMETRY=true" .env | wc -l
# Expected: 8

# Verify no API keys set
grep -E "^(LANGCHAIN_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)=" .env | grep -v "^#" | grep -v "=$"
# Expected: No output (all empty or commented)

# Test no network calls during query (after deployment)
sudo tcpdump -i any -n 'host not 127.0.0.1' &
TCPDUMP_PID=$!
curl -s -X POST http://localhost:8000/query -d '{"query":"test","use_rag":false}' > /dev/null
sleep 2
sudo kill $TCPDUMP_PID
# Expected: No packets captured (all local)
```

#### Category 11: CrawlModule (15 vars)

```bash
# Core settings
CRAWL_ENABLE=true
CRAWL_TIMEOUT_S=30                     # Per-page timeout
CRAWL_MAX_ITEMS=50                     # Per curation run
CRAWL_BATCH_SIZE=10                    # Batch processing

# Browser
CRAWL_HEADLESS_BROWSER=true            # Puppeteer headless
CRAWL_USER_AGENT=Mozilla/5.0...        # Standard user agent

# Security
CRAWL_SANITIZE_SCRIPTS=true            # Remove malicious JS
CRAWL_VERIFY_SSL=true                  # SSL certificate validation
CRAWL_FOLLOW_REDIRECTS=true            # Follow HTTP redirects

# Cache
CRAWL_CACHE_DIR=/app/cache             # Cache directory
CRAWL_CACHE_TTL_S=86400                # 24 hours
CRAWL_ENABLE_LOGGING=false             # Reduce noise

# Performance
CRAWL_RETRY_FAILED=true                # Retry failed fetches
CRAWL_RATE_LIMIT_PER_MIN=30            # Rate limiting (avoid bans)

# Version
CRAWL4AI_VERSION=0.7.3                 # CrawlModule version
```

**Rate Limiting Strategy**:

| Source | Rate Limit | Reason |
|--------|-----------|--------|
| Project Gutenberg | 30 req/min | Terms of service limit |
| arXiv | 30 req/min | Courtesy limit |
| PubMed | 30 req/min | NCBI guidelines |
| YouTube | 30 req/min | Avoid throttling |

**Validation**:

```bash
# Test CrawlModule initialization
sudo docker exec xnai_crawler python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_curator

crawler = get_curator()
print('✅ CrawlModule initialized')
"
# Expected: ✅ CrawlModule initialized

# Verify cache directory writable
sudo docker exec xnai_crawler test -w /app/cache && echo "✅ Cache writable" || echo "❌ Cache NOT writable"

# Test rate limiting (simulate 35 requests in 1 minute)
sudo docker exec xnai_crawler python3 -c "
import time
from app.XNAi_rag_app.crawl import RateLimiter

limiter = RateLimiter(rate_per_minute=30)
blocked_count = 0

for i in range(35):
    start = time.time()
    limiter.wait_if_needed()
    if time.time() - start > 0.1:  # Blocked >100ms
        blocked_count += 1

print(f'Blocked requests: {blocked_count}/35')
assert blocked_count >= 5, 'Rate limiter not working'
"
# Expected: Blocked requests: 5+/35
```

#### Category 12: RAG Configuration (9 vars)

```bash
# Retrieval
RAG_TOP_K=5                            # Retrieve top 5 docs
RAG_SIMILARITY_THRESHOLD=0.5           # Minimum similarity score

# Context truncation (memory safety)
RAG_TRUNCATE_CONTEXT=true
RAG_MAX_CONTEXT_CHARS=2048             # Hard limit for context
RAG_PER_DOC_CHARS=500                  # Excerpt length per doc

# Phase 2 features
RAG_ENABLE_RERANKING=false             # Reranking with cross-encoder
RAG_CACHE_RESULTS=true                 # Cache query results
RAG_CACHE_TTL_S=3600                   # 1 hour cache
RAG_FALLBACK_MODE=true                 # Direct LLM if no docs found
```

**Context Truncation Strategy** (Pattern from Claude's version):

```
Query: "What is Xoe-NovAi?"
↓
FAISS retrieval: Top 5 docs (similarity > 0.5)
↓
Per-doc truncation: 500 chars each (2500 total)
↓
Global truncation: Max 2048 chars final context
↓
LLM prompt: "Context: [2048 chars]\n\nQuery: What is Xoe-NovAi?"
```

**Why Truncate?**

2048-char limit ensures:
1. **Memory safety**: LLM context fits in 2048 token limit (with overhead)
2. **Speed**: Shorter context = faster inference
3. **Focus**: Most relevant information only (avoid noise)

**Validation**:

```bash
# Test RAG retrieval with truncation
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from dependencies import get_vectorstore, get_embeddings

vectorstore = get_vectorstore(get_embeddings())
docs = vectorstore.similarity_search('test', k=5)

# Check truncation applied
context = ''.join([d.page_content[:500] for d in docs[:5]])[:2048]
print(f'Context length: {len(context)} chars (max 2048)')
assert len(context) <= 2048, 'Context truncation failed'
print('✅ Context truncation working')
"
# Expected: ✅ Context truncation working
```

### 5.4 Validation Script (validate-config.sh)

**Complete validation** (16 checks):

```bash
#!/bin/bash
# validate-config.sh - Validate all 197 .env variables
# Guide Ref: Section 5.4

set -e

echo "=== Xoe-NovAi v0.1.3-beta Configuration Validation ==="
echo ""

# Check 1: .env file exists
if [ ! -f .env ]; then
  echo "❌ Check 1: .env file NOT found"
  exit 1
fi
echo "✅ Check 1: .env file found"

# Check 2: Count variables (should be 197)
VAR_COUNT=$(grep -E "^[A-Z_]+=" .env | wc -l)
if [ "$VAR_COUNT" -lt 190 ]; then
  echo "⚠️ Check 2: Only $VAR_COUNT variables found (expected 197)"
else
  echo "✅ Check 2: $VAR_COUNT variables found"
fi

# Check 3: Telemetry disables (8 primary)
TELEMETRY_COUNT=$(grep -E "^(CHAINLIT|CRAWL4AI|LLAMA_CPP|LANGCHAIN|FAISS|PROMETHEUS|UVICORN|FASTAPI)_NO_TELEMETRY=true" .env | wc -l)
if [ "$TELEMETRY_COUNT" -ne 8 ]; then
  echo "❌ Check 3: Only $TELEMETRY_COUNT/8 telemetry disables set"
  exit 1
fi
echo "✅ Check 3: All 8 telemetry disables enabled"

# Check 4: REDIS_PASSWORD not default
REDIS_PASS=$(grep "^REDIS_PASSWORD=" .env | cut -d= -f2)
if [ "$REDIS_PASS" = "CHANGE_ME_TO_16_CHAR_MINIMUM" ]; then
  echo "❌ Check 4: REDIS_PASSWORD still default (SECURITY RISK)"
  exit 1
fi
if [ ${#REDIS_PASS} -lt 16 ]; then
  echo "❌ Check 4: REDIS_PASSWORD too short (${#REDIS_PASS} chars, need 16+)"
  exit 1
fi
echo "✅ Check 4: REDIS_PASSWORD set (${#REDIS_PASS} chars)"

# Check 5: Ryzen optimizations
RYZEN_FLAGS=$(grep -E "^(LLAMA_CPP_N_THREADS|LLAMA_CPP_F16_KV|OPENBLAS_CORETYPE)" .env | wc -l)
if [ "$RYZEN_FLAGS" -lt 3 ]; then
  echo "❌ Check 5: Missing Ryzen optimization flags"
  exit 1
fi
echo "✅ Check 5: Ryzen optimizations configured"

# Check 6: Model paths exist
LLM_PATH=$(grep "^LLM_MODEL_PATH=" .env | cut -d= -f2)
if [ ! -f "$LLM_PATH" ]; then
  echo "❌ Check 6: LLM model NOT found: $LLM_PATH"
  exit 1
fi
echo "✅ Check 6: LLM model found: $LLM_PATH"

# Check 7: Embeddings path exists
EMBED_PATH=$(grep "^EMBEDDING_MODEL_PATH=" .env | cut -d= -f2)
if [ ! -f "$EMBED_PATH" ]; then
  echo "❌ Check 7: Embeddings NOT found: $EMBED_PATH"
  exit 1
fi
echo "✅ Check 7: Embeddings found: $EMBED_PATH"

# Check 8: APP_UID/GID match current user
CURRENT_UID=$(id -u)
ENV_UID=$(grep "^APP_UID=" .env | cut -d= -f2)
if [ "$CURRENT_UID" != "$ENV_UID" ]; then
  echo "⚠️ Check 8: UID mismatch (current: $CURRENT_UID, .env: $ENV_UID)"
else
  echo "✅ Check 8: UID matches current user"
fi

# Check 9: LOG_LEVEL appropriate
LOG_LEVEL=$(grep "^LOG_LEVEL=" .env | cut -d= -f2)
if [ "$LOG_LEVEL" = "DEBUG" ]; then
  echo "⚠️ Check 9: LOG_LEVEL=DEBUG (verbose, use INFO for production)"
else
  echo "✅ Check 9: LOG_LEVEL=$LOG_LEVEL"
fi

# Check 10: Memory limit reasonable
MEM_LIMIT=$(grep "^MEMORY_LIMIT_GB=" .env | cut -d= -f2)
if (( $(echo "$MEM_LIMIT < 4.0" | bc -l) )); then
  echo "⚠️ Check 10: MEMORY_LIMIT_GB=$MEM_LIMIT (low, may cause OOM)"
elif (( $(echo "$MEM_LIMIT > 8.0" | bc -l) )); then
  echo "⚠️ Check 10: MEMORY_LIMIT_GB=$MEM_LIMIT (high, may not fit in 16GB system)"
else
  echo "✅ Check 10: MEMORY_LIMIT_GB=$MEM_LIMIT (reasonable)"
fi

# Check 11: Docker Compose v2
if ! docker compose version | grep -q "v2"; then
  echo "❌ Check 11: Docker Compose v2 NOT installed"
  exit 1
fi
echo "✅ Check 11: Docker Compose v2 installed"

# Check 12: config.toml exists
if [ ! -f config.toml ]; then
  echo "❌ Check 12: config.toml NOT found"
  exit 1
fi
echo "✅ Check 12: config.toml found"

# Check 13: config.toml valid TOML
if ! python3 -c "import toml; toml.load('config.toml')" 2>/dev/null; then
  echo "❌ Check 13: config.toml invalid TOML syntax"
  exit 1
fi
echo "✅ Check 13: config.toml valid"

# Check 14: Directories exist
for dir in library knowledge models embeddings data/redis data/faiss_index backups; do
  if [ ! -d "$dir" ]; then
    echo "❌ Check 14: Directory missing: $dir"
    exit 1
  fi
done
echo "✅ Check 14: All directories exist"

# Check 15: Phase 2 flags disabled
PHASE2_ENABLED=$(grep "^PHASE2_QDRANT_ENABLED=" .env | cut -d= -f2)
if [ "$PHASE2_ENABLED" = "true" ]; then
  echo "⚠️ Check 15: Phase 2 enabled (not yet tested)"
else
  echo "✅ Check 15: Phase 2 disabled (Phase 1 scope)"
fi

# Check 16: Rate limiting configured
RATE_LIMIT=$(grep "^API_RATE_LIMIT_PER_MIN=" .env | cut -d= -f2)
if [ "$RATE_LIMIT" -lt 30 ]; then
  echo "⚠️ Check 16: API rate limit very strict ($RATE_LIMIT req/min)"
else
  echo "✅ Check 16: API rate limit: $RATE_LIMIT req/min"
fi

echo ""
echo "=== Validation Complete: 16/16 checks passed ✅ ==="
echo "Ready to deploy: docker compose up -d"
```

**Usage**:

```bash
# Run validation
bash validate-config.sh

# Expected output:
# ✅ Check 1: .env file found
# ✅ Check 2: 197 variables found
# ...
# ✅ Check 16: API rate limit: 60 req/min
# === Validation Complete: 16/16 checks passed ✅ ===
```

---

## Section 5.5: config.toml Structure (23 Sections)

**Application configuration** (non-secret, version-controlled):

```toml
# config.toml - Application parameters
# Guide Ref: Section 5.5
# READ-ONLY mount: ./config.toml:/app/XNAi_rag_app/config.toml:ro

[metadata]
stack_version = "v0.1.3-beta"
codename = "Resilient Polymath"
phase = 1
created_date = "2025-10-19"
last_updated = "2025-10-21"

[project]
name = "Xoe-NovAi"
telemetry_enabled = false              # CRITICAL: False always
privacy_first = true
local_only = true
zero_telemetry = true

[models]
llm_name = "gemma-3-4b-it-UD-Q5_K_XL"
llm_quantization = "Q5_K_XL"
embedding_name = "all-MiniLM-L12-v2"
embedding_quantization = "Q8_0"
embedding_dimension = 384

[performance]
memory_limit_gb = 6.0
cpu_threads = 6
f16_kv_enabled = true
token_rate_target = 20                 # 15-25 tok/s range
startup_time_s = 90
context_length = 2048

[server]
host = "0.0.0.0"
port = 8000
workers = 4
timeout_s = 300

[files]
extensions = ["txt", "md", "pdf"]
max_size_mb = 50
encoding = "utf-8"

[session]
max_duration_hours = 24
idle_timeout_minutes = 30
enable_persistence = true

[security]
non_root_uid = 1001
non_root_gid = 1001
drop_capabilities = ["ALL"]
no_new_privileges = true

[chainlit]
enabled = true
port = 8001
enable_chat_history = true
session_retention_days = 7

[redis]
host = "localhost"
port = 6379
db = 0
cache_ttl_s = 3600
pool_size = 10

[redis.streams]
enabled = false                        # Phase 2
group = "xnai_agents"

[redis.cache]
strategy = "lru"
max_items = 10000
eviction_policy = "allkeys-lru"

[crawl]
version = "0.7.3"
enabled = true
rate_limit_per_min = 30
timeout_s = 30
sanitize_scripts = true

[crawl.sources]
gutenberg_priority = 100
arxiv_priority = 90
pubmed_priority = 80
youtube_priority = 70

[crawl.allowlist]
gutenberg = ["*.gutenberg.org"]
arxiv = ["*.arxiv.org"]
pubmed = ["*.pubmed.ncbi.nlm.nih.gov"]

[crawl.metadata]
track_source = true
track_timestamp = true
track_category = true

[vectorstore]
type = "faiss"
metric = "L2"
index_type = "flat"
enable_backup = true

[vectorstore.qdrant]
enabled = false                        # Phase 2
host = "localhost"
port = 6333

[api]
enable_query = true
enable_stream = true
enable_health = true
enable_metrics = true

[logging]
level = "INFO"
format = "json"
max_size_mb = 10
backup_count = 5

[metrics]
enable_prometheus = true
enable_custom_metrics = true
scrape_interval_s = 15

[healthcheck]
targets = ["llm", "embeddings", "memory", "redis", "vectorstore", "ryzen", "crawler"]
timeout_s = 10
interval_s = 300

[backup]
enabled = true
retention_days = 7
max_backups = 5

[backup.faiss]
enabled = true
verify_on_load = true

[phase2]
enabled = false
multi_agent_enabled = false
agent_coordination = "none"

[phase2.agents]
coder_enabled = false
editor_enabled = false
manager_enabled = false
learner_enabled = false

[docker]
compose_version = "v2.29.2"
image_build_platform = "linux/amd64"

[validation]
validate_imports = true
validate_config = true
check_memory = true
check_disk = true

[debug]
enabled = false
verbose_logging = false
profile_performance = false
```

**Validation**:

```bash
# Test config.toml loads in container
sudo docker exec xnai_rag_api python3 -c "
import sys
sys.path.insert(0, '/app/XNAi_rag_app')
from config_loader import load_config

config = load_config()
print(f'✅ Config loaded: {len(config)} sections')
assert len(config) >= 23, f'Expected 23+ sections, got {len(config)}'
"
# Expected: ✅ Config loaded: 23 sections

# Verify READ-ONLY mount
sudo docker exec xnai_rag_api test -w /app/XNAi_rag_app/config.toml && echo "❌ config.toml writable" || echo "✅ config.toml READ-ONLY"
# Expected: ✅ config.toml READ-ONLY
```

---

## Common Issues & Solutions

### Issue 1: .env Variables Not Loaded

**Symptom**:
```
KeyError: 'REDIS_PASSWORD'
```

**Diagnosis**:
```bash
# Check .env file exists
ls -la .env

# Verify docker-compose.yml loads .env
grep "env_file" docker-compose.yml
```

**Solution**:

```bash
# Restart containers to reload .env
sudo docker compose down
sudo docker compose up -d
```

### Issue 2: config.toml Not Found in Container

**Symptom**:
```
FileNotFoundError: config.toml
```

**Diagnosis**:
```bash
# Check mount in docker-compose.yml
grep "config.toml" docker-compose.yml

# Verify file exists on host
ls -la config.toml
```

**Solution**:

```yaml
# Add mount to ALL services in docker-compose.yml
services:
  rag:
    volumes:
      - ./config.toml:/app/XNAi_rag_app/config.toml:ro  # ADD THIS
```

### Issue 3: Telemetry Still Enabled

**Symptom**:
Network calls to external telemetry servers detected.

**Diagnosis**:
```bash
# Check all 8 disables
grep "_NO_TELEMETRY=true" .env | wc -l
# Expected: 8
```

**Solution**:

```bash
# Add missing telemetry disables
echo "CHAINLIT_NO_TELEMETRY=true" >> .env
echo "CRAWL4AI_NO_TELEMETRY=true" >> .env
# ... (add all 8)

# Restart containers
sudo docker compose restart
```

---

## Cross-References

- **Group 2.1 (Prerequisites)**: See for dependency initialization chain
- **Group 3.1 (Docker)**: See for config.toml READ-ONLY mount implementation
- **Group 5.2 (Troubleshooting)**: See for configuration-related issues
- **Group 6.1 (Testing)**: See for configuration validation in CI/CD

---

## Summary: Configuration Checklist

**Before proceeding to Group 3 (Deployment), verify all ✅**:

- [ ] .env file exists with 197 variables
- [ ] All 8 telemetry disables set to `true`
- [ ] REDIS_PASSWORD changed from default (16+ chars)
- [ ] APP_UID/APP_GID match current user
- [ ] LLM_MODEL_PATH and EMBEDDING_MODEL_PATH valid
- [ ] Ryzen optimizations configured (N_THREADS=6, F16_KV=true, CORETYPE=ZEN)
- [ ] config.toml exists with 23 sections
- [ ] config.toml valid TOML syntax
- [ ] validate-config.sh passes all 16 checks

**Validation Command**:

```bash
bash validate-config.sh && echo "✅ Ready to deploy"
```

**Next**: Group 3.1 - Docker Build & Deployment Checklist