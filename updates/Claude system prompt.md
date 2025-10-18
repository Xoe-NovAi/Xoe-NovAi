# Xoe-NovAi Phase 1 v0.1.2 - System Prompt (Optimized for LLM)

**Last Updated**: October 18, 2025  
**Version**: v0.1.2-optimized  
**Status**: Production-ready stack, critical blockers identified and partially fixed  
**Current Phase**: Critical blocker fixes (Phase 1 of 3-phase PR readiness plan)

---

## I. PROJECT STATUS & PRIORITIES

### Current State (October 18, 2025)
- **Overall**: 72% production-ready (code quality, architecture, tests)
- **Critical Blockers**: 8 identified, 4 fixed in artifacts, 4 require manual implementation
- **Major Issues**: 12 identified (retry logic, subprocess handling, security gaps)
- **Test Coverage**: Incomplete (missing fixtures, benchmarks, security tests)

### Immediate Priority (Next 24-48 hours)
Fix all 8 critical blockers to reach "code compiles & tests run" milestone:

| # | Issue | Artifact Status | Implementation |
|---|-------|-----------------|-----------------|
| 1 | config.toml TOML parse error | ✅ READY | Copy artifact |
| 2 | logging_config.py corrupted methods | ✅ READY | Replace lines 395-475 |
| 3 | healthcheck.py missing import | ✅ READY | Copy artifact |
| 4 | conftest.py missing fixtures | ✅ READY | Copy artifact |
| 5 | Missing chainlit_app.py | ⚠️ MANUAL | cp app.py chainlit_app.py |
| 6 | Import paths broken (sys.path) | ⚠️ MANUAL | Add 3 lines to 5 files |
| 7 | crawl.py allowlist regex bug | ⚠️ MANUAL | Replace is_allowed_url() |
| 8 | get_curator() architecture | ✅ RESOLVED | Already correct in artifacts |

---

## II. TECHNICAL ARCHITECTURE (Fixed Understanding)

### Stack Components
1. **FastAPI Backend** (`main.py`, port 8000)
   - Query endpoint: `/query` (POST, <1000ms target)
   - Stream endpoint: `/stream` (POST, SSE)
   - Health endpoint: `/health`
   - Metrics endpoint: `/metrics` (port 8002)

2. **Chainlit UI** (`chainlit_app.py`, port 8001)
   - Commands: `/help`, `/stats`, `/reset`, `/rag on|off`, `/status`, `/curate`
   - Async SSE streaming from FastAPI backend
   - Non-blocking subprocess dispatch for curation

3. **CrawlModule** (`crawl.py`, in crawler container)
   - Functions (not class): `curate_from_source()`, `is_allowed_url()`, `sanitize_content()`
   - Sources: Gutenberg, arXiv, PubMed, YouTube
   - Rate limit: 30 req/min
   - Output: `/library/{category}/` (documents) + `/knowledge/curator/index.toml` (metadata)

4. **Infrastructure**
   - Redis 7.4.1: Caching (TTL: 3600s), streams (max_len: 1000)
   - FAISS: Vectorstore with auto-backups (retention: 7 days, max: 5)
   - Prometheus: Metrics on port 8002

### Data Flow
```
User Input
  ↓
Chainlit UI (port 8001)
  ↓
FastAPI RAG (port 8000) ← → FAISS (vectorstore)
  ↓                           ↓
LLM (Gemma-3-4B)        Redis (cache/streams)
  ↓
Token stream (SSE)
  ↓
User Output
```

### Folder Structure (CRITICAL)
- **Stack root** (above `app/`):
  - `library/` - Curated documents (categories: classics, physics, psychology, etc.)
  - `knowledge/` - Phase 2 agent bases (curator, coder, editor, manager, learner)
  - `.env` - 197 environment variables
  - `config.toml` - 23 config sections
  - `docker-compose.yml` - 4 services (redis, rag, ui, crawler)

- **App directory** (`app/XNAi_rag_app/`):
  - `main.py` - FastAPI backend
  - `chainlit_app.py` - Chainlit UI (renamed from `app.py`)
  - `crawl.py` - CrawlModule wrapper
  - Core modules: `config_loader.py`, `dependencies.py`, `logging_config.py`, `healthcheck.py`, `metrics.py`
  - Tests: `tests/conftest.py`, `tests/test_*.py`

---

## III. CRITICAL IMPLEMENTATION DETAILS

### Import Pattern (CRITICAL - Must Fix)
**Current (BROKEN)**:
```python
from logging_config import setup_logging  # Fails outside Docker
```

**Fixed (REQUIRED)**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from logging_config import setup_logging  # Works everywhere
```

**Files to update**: `main.py`, `app.py`/`chainlit_app.py`, `crawl.py`, `metrics.py`

### Configuration Cascade
1. **`.env`** (197 variables) → Runtime environment
2. **`config.toml`** (23 sections) → Application config
3. **`dependencies.py`** → Initializes LLM, embeddings, vectorstore, Redis, curator
4. **`healthcheck.py`** → Validates all components work

**Load order**: `config_loader.py` (cached, @lru_cache(maxsize=1)) → `dependencies.py` → `main.py`

### Performance Targets (Ryzen 7 5700U)
- Memory: <6GB (warn 5.5GB, critical 5.8GB)
- Token rate: 15–25 tok/s
- Query latency: <1000ms (p95)
- Startup: <90s
- Ingestion: 50–200 items/h
- Test coverage: >90%

### Telemetry: 8 Disables (ABSOLUTE)
1. `CHAINLIT_NO_TELEMETRY=true`
2. `CRAWL4AI_NO_TELEMETRY=true`
3. `LANGCHAIN_TRACING_V2=false`
4. `PYDANTIC_NO_TELEMETRY=true`
5. `FASTAPI_NO_TELEMETRY=true`
6. `LANGCHAIN_API_KEY=` (empty)
7. `OPENAI_API_KEY=` (empty)
8. `SCARF_NO_ANALYTICS=true`

---

## IV. CODE GENERATION GUIDELINES

### Must-Have Pattern (All Code)
```python
#!/usr/bin/env python3
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 - [Module Name]
# ============================================================================
# Purpose: [Single-line purpose]
# Guide Reference: Section X.Y
# Last Updated: YYYY-MM-DD
# ============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config, get_config_value
from logging_config import logger

CONFIG = load_config()

# ... rest of code ...

# Self-Critique: X/10
# - [Positive aspect] ✓
# - [Area for improvement]
```

### Testing Pattern
```python
@pytest.mark.unit
def test_function_name(mock_dependency):
    """Test description matching docstring style."""
    from module import function_name
    
    result = function_name(args)
    
    assert result == expected, f"Assertion message"
```

### Error Handling Pattern
```python
try:
    result = operation()
    logger.info("Operation succeeded", extra={"duration_ms": elapsed})
    return result
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)