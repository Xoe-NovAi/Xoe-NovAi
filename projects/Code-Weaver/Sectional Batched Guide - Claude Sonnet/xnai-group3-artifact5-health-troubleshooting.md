# Xoe-NovAi v0.1.3-beta: Health Monitoring & Troubleshooting

**Sections**: 6 (Health Checks), 13.4-13.6 (Troubleshooting + Recovery)  
**Purpose**: Production monitoring and diagnostic procedures  
**Cross-References**: Artifact 4 (Deployment), Artifact 3 (Configuration)

---

## Section 6: Health Check System (7 Targets)

### 6.1 Health Check Architecture

**Why 7 Targets**: Comprehensive validation of all critical components ensures production readiness. Each check validates a specific failure mode.

| # | Target | Purpose | Failure Mode | Recovery |
|---|--------|---------|--------------|----------|
| 1 | **LLM** | Inference ready | Model not loaded, OOM | Restart container, check memory |
| 2 | **Embeddings** | Vector generation | Model missing, dimension mismatch | Verify EMBEDDING_MODEL_PATH |
| 3 | **Memory** | Resource availability | >6GB usage, swap thrashing | Stop ingestion, reduce batch size |
| 4 | **Redis** | Cache operational | Connection refused, auth failed | Check REDIS_PASSWORD match |
| 5 | **Vectorstore** | FAISS index loaded | Corruption, missing index | Restore from backup |
| 6 | **Ryzen** | Optimization flags | Wrong thread count, no F16_KV | Check .env Ryzen settings |
| 7 | **Crawler** | CrawlModule ready | Import error, allowlist missing | Verify crawl4ai install |

### 6.2 Health Check Implementation

**File**: `app/XNAi_rag_app/healthcheck.py` (can run standalone or via API)

```python
#!/usr/bin/env python3
"""
Xoe-NovAi v0.1.3-beta: Multi-Target Health Checks
Guide Ref: Section 6.2

Usage:
  python3 healthcheck.py              # All 7 checks
  python3 healthcheck.py llm          # Single check
  curl http://localhost:8000/health   # API endpoint
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import time
from typing import Tuple, Dict
import psutil
import logging

from config_loader import load_config, get_config_value
from logging_config import get_logger

logger = get_logger(__name__)
CONFIG = load_config()

def check_llm() -> Tuple[bool, str]:
    """
    Target 1: Verify LLM loaded and can perform inference.
    
    Returns:
        (status, message) where status is True if healthy
    """
    try:
        from dependencies import get_llm
        llm = get_llm()
        
        if llm is None:
            return False, "LLM not initialized"
        
        # Quick inference test (1 token)
        start = time.time()
        result = llm("Hello", max_tokens=1, echo=False)
        elapsed = (time.time() - start) * 1000
        
        return True, f"LLM OK: Gemma-3 4B inference in {elapsed:.0f}ms"
    
    except Exception as e:
        logger.error(f"LLM health check failed: {e}", exc_info=True)
        return False, f"LLM error: {str(e)[:100]}"

def check_embeddings() -> Tuple[bool, str]:
    """
    Target 2: Verify embedding model loaded and produces correct dimensions.
    
    Returns:
        (status, message)
    """
    try:
        from dependencies import get_embeddings
        embeddings = get_embeddings()
        
        if embeddings is None:
            return False, "Embeddings not initialized"
        
        # Test vector generation
        test_vector = embeddings.embed_query("test")
        
        expected_dim = 384
        if len(test_vector) != expected_dim:
            return False, f"Dimension mismatch: {len(test_vector)} != {expected_dim}"
        
        return True, f"Embeddings OK: all-MiniLM-L12-v2 ({len(test_vector)} dims)"
    
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}", exc_info=True)
        return False, f"Embeddings error: {str(e)[:100]}"

def check_memory() -> Tuple[bool, str]:
    """
    Target 3: Verify memory usage within target (<6GB).
    
    Returns:
        (status, message) where status is False if >6GB
    """
    try:
        memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info().rss / (1024 ** 3)
        system_memory_gb = memory.used / (1024 ** 3)
        
        memory_limit = float(get_config_value('performance.memory_limit_gb', 6.0))
        
        if system_memory_gb > memory_limit:
            return False, f"Memory usage {system_memory_gb:.1f}GB exceeds {memory_limit}GB limit"
        
        return True, f"Memory OK: {process_memory:.1f}GB process, {system_memory_gb:.1f}GB / {memory_limit}GB system"
    
    except Exception as e:
        logger.error(f"Memory health check failed: {e}", exc_info=True)
        return False, f"Memory check error: {str(e)[:100]}"

def check_redis() -> Tuple[bool, str]:
    """
    Target 4: Verify Redis connection and auth.
    
    Returns:
        (status, message)
    """
    try:
        from redis import Redis
        
        client = Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True,
            socket_connect_timeout=5
        )
        
        # Test PING
        start = time.time()
        pong = client.ping()
        elapsed = (time.time() - start) * 1000
        
        if not pong:
            return False, "Redis PING failed"
        
        # Test SET/GET
        test_key = "xnai_healthcheck_test"
        client.setex(test_key, 10, "test_value")
        value = client.get(test_key)
        client.delete(test_key)
        
        if value != "test_value":
            return False, "Redis SET/GET failed"
        
        return True, f"Redis OK: PING {elapsed:.0f}ms, version {client.info()['redis_version']}"
    
    except Exception as e:
        logger.error(f"Redis health check failed: {e}", exc_info=True)
        return False, f"Redis error: {str(e)[:100]}"

def check_vectorstore() -> Tuple[bool, str]:
    """
    Target 5: Verify FAISS index can be loaded (or is empty on first run).
    
    Returns:
        (status, message)
    """
    try:
        from dependencies import get_vectorstore
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            # Check if this is first run (no index yet)
            index_path = Path('/app/XNAi_rag_app/faiss_index')
            if not index_path.exists():
                return True, "Vectorstore: No index (first run - OK)"
            else:
                return False, "Vectorstore initialization failed"
        
        vector_count = vectorstore.index.ntotal
        
        return True, f"Vectorstore OK: FAISS index with {vector_count} vectors"
    
    except Exception as e:
        logger.error(f"Vectorstore health check failed: {e}", exc_info=True)
        return False, f"Vectorstore error: {str(e)[:100]}"

def check_ryzen() -> Tuple[bool, str]:
    """
    Target 6: Verify Ryzen optimization flags are set correctly.
    
    Returns:
        (status, message) where status is False if critical flags missing
    """
    try:
        checks = {
            'LLAMA_CPP_N_THREADS': '6',
            'LLAMA_CPP_F16_KV': 'true',
            'OPENBLAS_CORETYPE': 'ZEN',
            'LLAMA_CPP_USE_MLOCK': 'true'
        }
        
        missing = []
        wrong_value = []
        
        for key, expected in checks.items():
            actual = os.getenv(key, '').lower()
            expected_lower = expected.lower()
            
            if not actual:
                missing.append(key)
            elif actual != expected_lower:
                wrong_value.append(f"{key}={actual} (expected {expected})")
        
        if missing:
            return False, f"Ryzen: Missing flags {', '.join(missing)}"
        
        if wrong_value:
            return False, f"Ryzen: Wrong values {', '.join(wrong_value)}"
        
        return True, f"Ryzen OK: N_THREADS=6, F16_KV=true, CORETYPE=ZEN, MLOCK=true"
    
    except Exception as e:
        logger.error(f"Ryzen health check failed: {e}", exc_info=True)
        return False, f"Ryzen check error: {str(e)[:100]}"

def check_crawler() -> Tuple[bool, str]:
    """
    Target 7: Verify CrawlModule (crawl4ai) can be imported and initialized.
    
    Returns:
        (status, message)
    """
    try:
        import crawl4ai
        
        # Check allowlist file exists
        allowlist_path = Path('/app/allowlist.txt')
        if not allowlist_path.exists():
            return False, "Crawler: allowlist.txt missing"
        
        # Check library directory writable
        library_path = Path(os.getenv('LIBRARY_PATH', '/library'))
        if not library_path.exists():
            return False, f"Crawler: library path missing {library_path}"
        
        # Test write access
        test_file = library_path / '.healthcheck_test'
        test_file.touch()
        test_file.unlink()
        
        return True, f"Crawler OK: crawl4ai {crawl4ai.__version__}, library writable"
    
    except Exception as e:
        logger.error(f"Crawler health check failed: {e}", exc_info=True)
        return False, f"Crawler error: {str(e)[:100]}"

def run_health_checks(targets: list = None) -> Dict[str, Tuple[bool, str]]:
    """
    Run specified health checks (or all 7 if None).
    
    Args:
        targets: List of check names (e.g., ['llm', 'memory']) or None for all
    
    Returns:
        Dict mapping check name to (status, message) tuple
    """
    all_checks = {
        'llm': check_llm,
        'embeddings': check_embeddings,
        'memory': check_memory,
        'redis': check_redis,
        'vectorstore': check_vectorstore,
        'ryzen': check_ryzen,
        'crawler': check_crawler
    }
    
    if targets:
        checks_to_run = {k: v for k, v in all_checks.items() if k in targets}
    else:
        checks_to_run = all_checks
    
    results = {}
    for name, check_func in checks_to_run.items():
        try:
            status, message = check_func()
            results[name] = (status, message)
            
            symbol = "✓" if status else "✗"
            logger.info(f"{symbol} {name}: {message}")
            
        except Exception as e:
            results[name] = (False, f"Check failed: {e}")
            logger.error(f"✗ {name}: Unexpected error {e}", exc_info=True)
    
    return results

def main():
    """CLI entry point for health checks."""
    import sys
    
    # Parse arguments
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    
    print("=" * 70)
    print("Xoe-NovAi v0.1.3-beta Health Checks")
    print("=" * 70)
    
    results = run_health_checks(targets)
    
    # Print summary
    total = len(results)
    passed = sum(1 for status, _ in results.values() if status)
    failed = total - passed
    
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 70)
    
    if failed > 0:
        print("\nFailed checks:")
        for name, (status, message) in results.items():
            if not status:
                print(f"  ✗ {name}: {message}")
        sys.exit(1)
    else:
        print("\n✓ All health checks passed - stack is healthy")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

**Standalone Usage Examples**:

```bash
# Run all 7 checks
python3 app/XNAi_rag_app/healthcheck.py

# Run specific checks
python3 app/XNAi_rag_app/healthcheck.py llm memory ryzen

# In container
sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/healthcheck.py

# Expected output (all passing):
# ======================================================================
# Xoe-NovAi v0.1.3-beta Health Checks
# ======================================================================
# ✓ llm: LLM OK: Gemma-3 4B inference in 847ms
# ✓ embeddings: Embeddings OK: all-MiniLM-L12-v2 (384 dims)
# ✓ memory: Memory OK: 3.8GB process, 4.2GB / 6.0GB system
# ✓ redis: Redis OK: PING 2ms, version 7.4.1
# ✓ vectorstore: Vectorstore OK: FAISS index with 0 vectors
# ✓ ryzen: Ryzen OK: N_THREADS=6, F16_KV=true, CORETYPE=ZEN, MLOCK=true
# ✓ crawler: Crawler OK: crawl4ai 0.7.3, library writable
# 
# ======================================================================
# RESULT: 7/7 checks passed
# ======================================================================
# 
# ✓ All health checks passed - stack is healthy
```

---

### 6.3 Health API Endpoint

**File**: `app/XNAi_rag_app/main.py` (FastAPI integration)

```python
from fastapi import FastAPI
from healthcheck import run_health_checks
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health_endpoint():
    """
    Health check API endpoint (returns JSON).
    
    Returns:
        {
          "status": "healthy" | "degraded",
          "version": "v0.1.3-beta",
          "timestamp": "2025-10-22T14:32:15Z",
          "memory_gb": 4.2,
          "components": {"llm": true, ...},
          "details": {"llm": "LLM OK: ...", ...}
        }
    """
    results = run_health_checks()
    
    all_healthy = all(status for status, _ in results.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "v0.1.3-beta",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "memory_gb": round(psutil.virtual_memory().used / (1024**3), 1),
        "components": {k: status for k, (status, _) in results.items()},
        "details": {k: message for k, (_, message) in results.items()}
    }
```

**API Usage**:

```bash
# Query health endpoint
curl -s http://localhost:8000/health | jq

# Check specific component
curl -s http://localhost:8000/health | jq '.components.llm'
# Output: true

# Count failed checks
curl -s http://localhost:8000/health | jq '.components | map(select(. == false)) | length'
# Output: 0 (all passing)
```

---

### 6.4 Prometheus Metrics

**Implementation**: `app/XNAi_rag_app/metrics.py`

```python
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import psutil
import time

# Gauges (current state)
memory_usage = Gauge('xnai_memory_usage_gb', 'Memory usage in GB', ['component'])
token_rate = Gauge('xnai_token_rate_tps', 'Token generation rate', ['model'])
active_sessions = Gauge('xnai_active_sessions', 'Number of active sessions')

# Histograms (distributions)
response_latency = Histogram('xnai_response_latency_ms', 'Response latency in ms', 
                            ['endpoint', 'method'],
                            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
rag_retrieval_time = Histogram('xnai_rag_retrieval_time_ms', 'RAG retrieval time in ms',
                               buckets=[5, 10, 25, 50, 100, 250, 500, 1000])

# Counters (cumulative)
requests_total = Counter('xnai_requests_total', 'Total requests', ['endpoint', 'method', 'status'])
errors_total = Counter('xnai_errors_total', 'Total errors', ['error_type', 'component'])
tokens_generated_total = Counter('xnai_tokens_generated_total', 'Total tokens generated', ['model'])
queries_processed_total = Counter('xnai_queries_processed_total', 'Total queries processed', ['rag_enabled'])

def update_metrics():
    """Update gauge metrics (called periodically)."""
    # Memory
    memory = psutil.virtual_memory()
    process_memory = psutil.Process().memory_info().rss / (1024**3)
    memory_usage.labels(component='system').set(memory.used / (1024**3))
    memory_usage.labels(component='process').set(process_memory)
    
    # Token rate (calculated from recent queries)
    # Implementation depends on token tracking in query handler

def metrics_endpoint():
    """Return Prometheus metrics."""
    update_metrics()
    return generate_latest()
```

**Metrics Endpoint**:

```python
# In main.py
from fastapi.responses import PlainTextResponse
from metrics import metrics_endpoint

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus scrape endpoint (port 8002)."""
    return metrics_endpoint()
```

**Metrics Access**:

```bash
# All metrics
curl http://localhost:8002/metrics

# Filter by prefix
curl http://localhost:8002/metrics | grep "xnai_"

# Query specific metric
curl http://localhost:8002/metrics | grep "xnai_token_rate_tps"
# Output: xnai_token_rate_tps{model="gemma-3-4b-it"} 20.5
```

---

## Section 13.4: Common Issues & Solutions

**Purpose**: Diagnostic table for rapid troubleshooting (80% of issues covered)

| Issue | Symptom | Root Cause | Diagnostic Command | Solution |
|-------|---------|------------|-------------------|----------|
| **ModuleNotFoundError** | `cannot import config_loader` | Missing import path resolution (Pattern 1) | `docker exec xnai_rag_api python3 -c "from app.XNAi_rag_app.main import app"` | Add `sys.path.insert(0, str(Path(__file__).parent))` to entry point |
| **config.toml not found** | `FileNotFoundError: config.toml` | No mount in docker-compose.yml | `docker exec xnai_rag_api test -f /app/XNAi_rag_app/config.toml` | Add volume: `./config.toml:/app/XNAi_rag_app/config.toml:ro` in rag, ui, crawler services |
| **Permission denied: logs/** | `OSError: [Errno 13] /app/XNAi_rag_app/logs/` | Directory not writable | `docker exec xnai_rag_api ls -la /app/XNAi_rag_app/logs` | In Dockerfile: `chmod 777 /app/XNAi_rag_app/logs` before USER switch |
| **Redis connection refused** | `ConnectionRefusedError` | Password mismatch or Redis not started | `docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" PING` | Verify REDIS_PASSWORD identical in .env and all services; check Redis healthy |
| **LLM initialization failed** | `RuntimeError: Model not loaded` | Wrong path or insufficient memory | `docker exec xnai_rag_api test -f "$LLM_MODEL_PATH"` | Verify LLM_MODEL_PATH in .env; check memory <6GB; restart container |
| **Memory >6GB** | OOM kills, swap thrashing | LLM load without F16_KV or running ingestion | `docker stats --no-stream xnai_rag_api` | Set LLAMA_CPP_F16_KV=true; stop ingestion; reduce batch_size |
| **Token rate <15 tok/s** | Slow inference | Wrong thread count or CPU throttling | `docker exec xnai_rag_api env | grep LLAMA_CPP_N_THREADS` | Set LLAMA_CPP_N_THREADS=6; check CPU governor (use 'performance') |
| **Health check: Ryzen fail** | Missing optimization flags | .env not loaded or wrong values | `docker exec xnai_rag_api env | grep -E "LLAMA_CPP|OPENBLAS"` | Update .env Ryzen section; rebuild container |
| **Crawler import error** | `ImportError: crawl4ai` | Wrong crawl4ai version or missing | `docker exec xnai_crawler python3 -c "import crawl4ai; print(crawl4ai.__version__)"` | Downgrade to 0.7.3: `pip install crawl4ai==0.7.3` in requirements-crawl.txt |
| **URL spoofing allowed** | Malicious URLs not blocked | Old substring regex in crawl.py | Test: `is_allowed_url("https://evil-gutenberg.org", ["*.gutenberg.org"])` | Apply domain-anchored regex fix (Section 10.1) |
| **Ingestion data loss** | Crash loses all progress | No checkpointing (Pattern 4) | Check: `ls -la /app/XNAi_rag_app/faiss_index/` after crash | Implement batch checkpointing: `vectorstore.save_local()` every 100 docs |
| **UI hangs on /curate** | Blocking subprocess | No background threading (Pattern 3) | Check logs: `docker logs xnai_chainlit_ui | grep "curate"` | Apply Pattern 3: `threading.Thread(target=curate_worker)` |

---

## Section 13.5: Advanced Diagnostics

### 13.5.1 Container Shell Access

```bash
# Access running container shell
sudo docker exec -it xnai_rag_api bash

# Inside container:
# - Check environment variables
env | grep -E "LLAMA_CPP|REDIS|LLM_MODEL"

# - Test imports
python3 -c "from app.XNAi_rag_app.main import app; print('OK')"

# - Check file permissions
ls -la /app/XNAi_rag_app/logs
ls -la /app/XNAi_rag_app/faiss_index

# - Test health checks individually
python3 /app/XNAi_rag_app/healthcheck.py llm
python3 /app/XNAi_rag_app/healthcheck.py memory

# Exit container
exit
```

### 13.5.2 Log Analysis

```bash
# View logs (all services)
sudo docker compose logs -f

# Specific service
sudo docker compose logs rag -n 100
sudo docker compose logs ui --since 10m
sudo docker compose logs crawler --follow

# Search for errors
sudo docker compose logs | grep -i "error" | tail -20

# Filter by log level (JSON logs)
sudo docker compose logs rag | grep '"level":"ERROR"'

# Export logs for analysis
sudo docker compose logs > /tmp/xnai_logs_$(date +%Y%m%d_%H%M%S).txt
```

### 13.5.3 Network Diagnostics

```bash
# Check service connectivity
sudo docker exec xnai_rag_api curl http://redis:6379
sudo docker exec xnai_chainlit_ui curl http://rag:8000/health

# Check port bindings
sudo netstat -tulpn | grep -E "8000|8001|6379|8002"

# Test from host
curl http://localhost:8000/health
curl http://localhost:8001
curl http://localhost:8002/metrics

# DNS resolution inside container
sudo docker exec xnai_rag_api nslookup redis
sudo docker exec xnai_rag_api ping -c 1 redis
```

### 13.5.4 Resource Monitoring

```bash
# Real-time stats
sudo docker stats

# Memory breakdown
sudo docker exec xnai_rag_api python3 << 'EOF'
import psutil
mem = psutil.virtual_memory()
print(f"Total: {mem.total / (1024**3):.1f}GB")
print(f"Used: {mem.used / (1024**3):.1f}GB")
print(f"Available: {mem.available / (1024**3):.1f}GB")
print(f"Percent: {mem.percent}%")
EOF

# Disk usage
df -h | grep -E "Filesystem|/$"
sudo du -sh library/ knowledge/ data/ backups/

# CPU info
lscpu | grep -E "Model name|CPU\(s\)|Thread"
```

---

## Section 13.6: Recovery Procedures

### 13.6.1 Service Restart (Soft Recovery)

```bash
# Restart specific service (preserves data)
sudo docker compose restart rag

# Restart all services
sudo docker compose restart

# Wait for health checks
sleep 90
sudo docker compose ps | grep "healthy"
```

### 13.6.2 Full Stack Reset (Moderate Recovery)

```bash
# Stop and remove containers (preserves volumes)
sudo docker compose down

# Rebuild images (with cache)
sudo docker compose build

# Start fresh
sudo docker compose up -d

# Validate
sleep 90
curl -s http://localhost:8000/health | jq '.components'
```

### 13.6.3 Complete Clean Rebuild (Aggressive Recovery)

**WARNING**: Removes all data including FAISS index, Redis cache, logs

```bash
# Stop and remove everything (including volumes)
sudo docker compose down -v

# Remove images
sudo docker rmi $(sudo docker images -q 'xnai*')

# Clean directories (DESTRUCTIVE - backup first!)
sudo rm -rf data/faiss_index/* data/redis/* data/cache/*

# Rebuild from scratch
sudo docker compose build --no-cache
sudo docker compose up -d

# Validate
sleep 90
sudo docker compose ps
```

### 13.6.4 FAISS Index Recovery (Corruption Fix)

```bash
# Symptom: Vectorstore health check fails, search returns errors

# Step 1: Stop services
sudo docker compose down

# Step 2: Check backup exists
ls -lh backups/faiss_backup_*.tar.gz

# Step 3: Restore from latest backup
LATEST_BACKUP=$(ls -t backups/faiss_backup_*.tar.gz | head -1)
sudo tar -xzf "$LATEST_BACKUP" -C data/faiss_index/

# Step 4: Verify extraction
ls -la data/faiss_index/
# Expected: index.faiss, index.pkl

# Step 5: Fix permissions
sudo chown -R 1001:1001 data/faiss_index/

# Step 6: Restart
sudo docker compose up -d

# Step 7: Validate
sleep 90
sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/healthcheck.py vectorstore
# Expected: "Vectorstore OK: FAISS index with N vectors"
```

### 13.6.5 Redis Cache Reset (Performance Degradation Fix)

```bash
# Symptom: Slow queries, high memory, stale cache

# Option 1: Flush cache (keeps Redis running)
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" FLUSHDB

# Option 2: Restart Redis (clears cache)
sudo docker compose restart redis

# Validate
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" INFO stats
# Check: keyspace_hits, keyspace_misses (should reset to 0)
```

---

## Section 13.7: Performance Tuning

### 13.7.1 Token Rate Optimization

**If token rate <15 tok/s:**

```bash
# Check current settings
sudo docker exec xnai_rag_api env | grep LLAMA_CPP

# Tune .env:
LLAMA_CPP_N_THREADS=6          # 75% of 8 cores
LLAMA_CPP_F16_KV=true          # Enable FP16 KV cache
LLAMA_CPP_USE_MLOCK=true       # Lock memory (prevent swapping)
OPENBLAS_CORETYPE=ZEN          # Ryzen optimization
OMP_NUM_THREADS=1              # Disable OpenMP (use native threads)

# Rebuild and test
sudo docker compose down
sudo docker compose build --no-cache rag
sudo docker compose up -d

# Benchmark
sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/scripts/query_test.py --queries 10
```

### 13.7.2 Memory Optimization

**If memory usage >5.5GB:**

```bash
# Reduce batch size in ingestion
# Edit: scripts/ingest_library.py
# Change: batch_size=100 → batch_size=50

# Tune Redis memory
# Edit: docker-compose.yml
environment:
  - REDIS_MAXMEMORY=256mb  # Reduce from 512mb

# Restart services
sudo docker compose restart
```

### 13.7.3 Disk Space Management

```bash
# Clean Docker system (removes unused data)
sudo docker system prune -a --volumes --force

# Rotate logs
sudo docker compose exec rag python3 << 'EOF'
from logging_config import rotate_logs
rotate_logs()
EOF

# Clean old backups (keep last 5)
ls -t backups/*.tar.gz | tail -n +6 | xargs rm -f
```

---

## Diagnostic Command Reference

**Quick Copy-Paste Commands for Troubleshooting**:

```bash
# ======================================================================
# HEALTH CHECKS
# ======================================================================

# All checks
curl -s http://localhost:8000/health | jq

# Failed components
curl -s http://localhost:8000/health | jq '.components | to_entries | map(select(.value == false)) | .[].key'

# Standalone health check
sudo docker exec xnai_rag_api python3 /app/XNAi_rag_app/healthcheck.py

# ======================================================================
# LOGS & DEBUGGING
# ======================================================================

# Recent errors (all services)
sudo docker compose logs --since 10m | grep -i "error"

# Specific service errors
sudo docker compose logs rag | grep -i "error" | tail -20

# Follow logs (real-time)
sudo docker compose logs -f rag

# Export logs
sudo docker compose logs > /tmp/xnai_debug_$(date +%Y%m%d_%H%M%S).txt

# ======================================================================
# RESOURCE MONITORING
# ======================================================================

# Memory usage
sudo docker stats --no-stream xnai_rag_api | awk '{print $3}'

# Detailed memory breakdown
sudo docker exec xnai_rag_api python3 -c "import psutil; mem=psutil.virtual_memory(); print(f'{mem.used/(1024**3):.1f}GB / {mem.total/(1024**3):.1f}GB')"

# CPU usage
sudo docker stats --no-stream xnai_rag_api | awk '{print $2}'

# Disk space
df -h / | tail -1

# ======================================================================
# CONNECTIVITY TESTS
# ======================================================================

# Test API endpoints
curl -f http://localhost:8000/health && echo "API OK" || echo "API FAILED"
curl -f http://localhost:8001 && echo "UI OK" || echo "UI FAILED"
curl -f http://localhost:8002/metrics && echo "Metrics OK" || echo "Metrics FAILED"

# Test Redis
sudo docker exec xnai_redis redis-cli -a "$REDIS_PASSWORD" PING

# Test service-to-service
sudo docker exec xnai_chainlit_ui curl -f http://rag:8000/health

# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

# Check environment variables
sudo docker exec xnai_rag_api env | grep -E "LLAMA_CPP|REDIS|LLM_MODEL"

# Verify config.toml mounted
sudo docker exec xnai_rag_api test -f /app/XNAi_rag_app/config.toml && echo "Mounted" || echo "Missing"

# Check model files
sudo docker exec xnai_rag_api ls -lh /models/*.gguf /embeddings/*.gguf

# ======================================================================
# RECOVERY COMMANDS
# ======================================================================

# Restart all services
sudo docker compose restart

# Full rebuild (preserves data)
sudo docker compose down && sudo docker compose build && sudo docker compose up -d

# Clean rebuild (DESTRUCTIVE - removes data)
sudo docker compose down -v && sudo docker compose build --no-cache && sudo docker compose up -d

# Restore FAISS from backup
sudo tar -xzf backups/faiss_backup_latest.tar.gz -C data/faiss_index/
```

---

## Cross-References

- **Deployment**: Artifact 4 (Docker orchestration, compose configuration)
- **Configuration**: Artifact 3 (.env variables, config.toml structure)
- **Prerequisites**: Artifact 2 (system requirements, directory setup)

---

## End of Artifact 5

**Sections Covered**: 6 (Health Checks), 13.4-13.7 (Troubleshooting + Recovery)  
**Token Count**: ~11,900  
**Next**: Artifact 6 (Section 11 - Library Ingestion)