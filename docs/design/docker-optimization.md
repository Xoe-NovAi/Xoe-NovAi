# Docker Image Size Optimization Strategy
## Xoe-NovAi Phase 1 v0.1.4-stable

**Status**: Analysis Complete | **Date**: 2026-01-09  
**Target Reductions**: API (2.84GB → ~1.2GB), UI (800MB → ~350MB), Crawler (1.3GB → ~600MB)

---

## 1. CURRENT STATE ANALYSIS

### Image Size Breakdown
| Service    | Current | Target | Reduction | Status |
|------------|---------|--------|-----------|--------|
| API        | 2.84 GB | 1.2 GB | 58%       | ⚠️ CRITICAL |
| Chainlit   | 800 MB  | 350 MB | 56%       | ⚠️ HIGH PRIORITY |
| Crawler    | 1.3 GB  | 600 MB | 54%       | ⚠️ HIGH PRIORITY |
| Curation   | TBD     | ~500 MB| TBD       | ⏳ PENDING |

### Known Heavy Dependencies
1. **`faiss-cpu` (250-400MB)**
   - Pure C++ linear algebra library
   - No CPU-only alternative available
   - Optimization: Use smaller model sizes, quantization

2. **`torch` & `torchaudio` (800MB-1.5GB combined)**
   - Chainlit UI currently includes PyTorch (unnecessary)
   - Recommendation: Remove from Chainlit, use lightweight API calls instead

3. **`openai-whisper` (100-150MB + model cache)**
   - Large transformer model
   - Alternative: Use `faster-whisper` with CTranslate2 (3-4x faster, 50% smaller)

4. **`llama-cpp-python` (builds from source, 200-300MB)**
   - Can be optimized with wheel caching

5. **`crawl4ai` dependencies**
   - Browser engine bloat (Playwright/Puppeteer)
   - Recommendation: Modularize, use lightweight extraction

---

## 2. ROOT CAUSE ANALYSIS

### API Image (2.84GB - CRITICAL)
**Primary Contributors**:
1. `faiss-cpu` full binary build: ~350-400MB
2. Python build artifacts & __pycache__: ~300-400MB  
3. `langchain-community` + transitive deps: ~400-500MB
4. System libraries (apt packages left behind): ~200-300MB
5. Multiple model caches: ~300-500MB (if loaded)
6. `torch` + dependencies (if pulling audio): ~800MB+

**Key Issues**:
- Multi-stage build exists but may not be cleaning site-packages properly
- Large transitive dependencies from langchain ecosystem
- No `.dockerignore` file to exclude build artifacts from context
- Possible redundant/unused dependencies in requirements

### Chainlit Image (800MB - HIGH)
**Primary Contributors**:
1. `torch` + `torchaudio` (UNNECESSARY): ~1.0-1.5GB total
2. `chainlit` + UI dependencies: ~150-200MB
3. Python site-packages bloat: ~150-200MB
4. System libraries: ~100-150MB

**Quick Win**: Remove PyTorch entirely (~1GB reduction)

### Crawler Image (1.3GB - HIGH)
**Primary Contributors**:
1. `crawl4ai` + Playwright/Puppeteer engines: ~400-600MB
2. Chromium binary (if included): ~200-300MB
3. System libraries for browser support: ~150-200MB
4. Standard library + site-packages: ~200-300MB

---

## 3. OPTIMIZATION STRATEGIES (PRIORITY ORDER)

### 🔴 CRITICAL: Chainlit - Remove PyTorch (56% reduction, ~450MB)
**Issue**: Chainlit UI doesn't need PyTorch, it calls the RAG API via HTTP  
**Solution**:
```dockerfile
# Dockerfile.chainlit BEFORE:
torch>=2.2.0 torchaudio>=2.2.0  # 1GB+

# AFTER (in requirements-chainlit.txt):
# Remove torch, torchaudio completely
# Chainlit calls RAG API via httpx/requests only
```

**Expected Impact**: UI image: 800MB → ~280MB (65% reduction)  
**Risk**: Low - Chainlit doesn't perform ML inference locally

---

### 🟠 HIGH: API - Replace `openai-whisper` with `faster-whisper`
**Current**: 
```
openai-whisper (+ transformers, torch): ~150MB + 800MB dependencies
```

**Replacement**:
```dockerfile
# requirements-api.txt
faster-whisper==1.2.1  # 50MB
ctranslate2==4.6.2     # 100MB
# Replaces: openai-whisper, transformers, torch audio stack
```

**Benefits**:
- 3-4x faster transcription (CPU)
- 50% smaller library footprint
- Compatible with existing Whisper models
- Better CPU optimization (CTranslate2 uses quantization)

**Migration Cost**: Minimal - API code change only
```python
# Before
from whisper import load_model
model = load_model("base")
result = model.transcribe("audio.mp3")

# After
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.mp3")
result = {"text": " ".join([s.text for s in segments])}
```

**Expected Impact**: API image: 2.84GB → ~2.2GB (22% reduction, 640MB saved)

---

### 🟠 HIGH: Crawler - Modularize Browser Engine
**Current**: crawl4ai brings entire Playwright/Puppeteer stack (~400-600MB)

**Strategy**:
1. **Option A (RECOMMENDED)**: Use lightweight HTML parsing only
   ```dockerfile
   # Keep: beautifulsoup4, lxml, httpx
   # Remove: crawl4ai (brings Playwright)
   # Add: trafilatura (lightweight content extraction, 2MB)
   ```
   - Trafilatura: 2MB, excellent content extraction
   - beautifulsoup4: Already present
   - Loss: JavaScript rendering (but most knowledge base content is static)

2. **Option B**: Keep crawl4ai but disable browser
   ```dockerfile
   # requirements-crawl.txt
   crawl4ai==0.7.3
   # Environment variable in docker-entrypoint:
   CRAWL4AI_NO_BROWSER=true  # Use requests only, not Playwright
   ```

**Expected Impact**: Crawler: 1.3GB → ~650MB (50% reduction)

---

### 🟡 MEDIUM: API - Aggressive Site-Packages Cleanup
**Techniques**:

1. **Remove Test Files & Documentation**
```dockerfile
# In multi-stage build, STAGE 1 builder → STAGE 2 runtime:

# Copy only needed files from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
RUN find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "*.dist-info/tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyo" -delete && \
    du -sh /usr/local/lib/python3.12/site-packages
```

2. **Use Slim Base Image Optimizations**
```dockerfile
# Current: FROM python:3.12-slim
# Already good, but ensure:
RUN apt-get update && \
    apt-get install --no-install-recommends \
    libgomp1 \  # For OpenBLAS threads only
    libopenblas0 \  # FAISS dependency
    ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```

**Expected Impact**: ~100-150MB per image

---

### 🟡 MEDIUM: API/Crawler - Use `.dockerignore`
**Create** `.dockerignore`:
```
__pycache__
*.pyc
*.pyo
.pytest_cache
.mypy_cache
tests/
docs/
examples/
README.md
*.md
.git
.gitignore
.venv
venv/
*.egg-info
.tox/
.coverage
htmlcov/
dist/
build/
*.egg
```

**Expected Impact**: Faster builds, slightly smaller context (~50MB)

---

### 🟡 MEDIUM: API - Dependency Audit & Trimming
**Current Issues**:
```
langchain-community (pulls 200+ transitive deps)
  ├── pydantic (also explicit)
  ├── numpy (via faiss/langchain)
  ├── sqlalchemy (not needed for RAG)
  └── ... 100+ others
```

**Recommendations**:

1. **Replace heavy langchain-community with targeted imports**
   ```python
   # Current: from langchain_community import ...
   # Better: Direct Hugging Face, Anthropic, OpenAI imports
   
   # Before (langchain-community adds 400MB+ of bloat):
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_community.llms import LlamaCpp
   
   # After (direct imports):
   from huggingface_hub import hf_hub_download
   from llama_cpp import Llama  # Already have llama-cpp-python
   ```

2. **Audit explicit dependencies**
   - ✅ Keep: fastapi, redis, faiss-cpu, llama-cpp-python, requests
   - ❓ Review: langchain-community, langchain-core (often redundant)
   - ❌ Remove: unnecessary monitoring, unused API integrations

**Expected Impact**: ~200-300MB reduction in API image

---

### 🟡 MEDIUM: Crawler - Remove Unnecessary HTML Parsing
```dockerfile
# Current requirements-crawl.txt:
beautifulsoup4==4.12.3  # 200KB - KEEP
lxml==5.3.0             # 2MB - KEEP
crawl4ai==0.7.3         # 500MB - REMOVE if not needed
requests==2.32.5        # Already in API
httpx==0.27.2           # Redundant with requests?

# Optimized:
beautifulsoup4==4.12.3
lxml==5.3.0
trafilatura==2.0.0      # 2MB, excellent extraction
httpx==0.27.2
```

**Expected Impact**: ~300MB

---

### 🟢 LOW: Layer Caching Optimization
**Ensure requirements files are copied early**:
```dockerfile
# ✅ GOOD (current approach):
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# ❌ BAD (entire app copied before install):
COPY . .
RUN pip install -r requirements-api.txt
```

Status: Already implemented correctly ✅

---

## 4. IMPLEMENTATION ROADMAP

### Phase 1 (Immediate - 48 hours)
**Actions**:
1. Remove PyTorch from Chainlit image
   - Expected: 800MB → 280MB (65% reduction)
   - File: `requirements-chainlit.txt`, `Dockerfile.chainlit`
   - Risk: Very low

2. Create `.dockerignore` file
   - File: `.dockerignore` (at repo root)
   - Risk: None

3. Audit langchain-community usage
   - Search: `grep -r "langchain_community" app/`
   - Goal: Replace with direct imports
   - Expected: 150-200MB savings

**Estimated Total Savings**: ~700MB across images

---

### Phase 2 (Short-term - 1-2 weeks)
**Actions**:
1. Replace `openai-whisper` with `faster-whisper`
   - Files: `requirements-api.txt`, `app/XNAi_rag_app/` (search for whisper imports)
   - Expected: 640MB saved
   - Risk: Low (API-only change)

2. Modularize crawler (Option A: Remove crawl4ai)
   - Files: `requirements-crawl.txt`, `app/crawler/` (refactor)
   - Expected: 400-600MB saved
   - Risk: Medium (feature change)

3. Implement site-packages cleanup in Dockerfiles
   - Files: `Dockerfile.api`, `Dockerfile.crawl`, `Dockerfile.chainlit`
   - Expected: 150-200MB per image

**Estimated Total Savings**: ~1.1GB

---

### Phase 3 (Medium-term - 2-4 weeks)
**Actions**:
1. Replace heavy dependencies (langchain-community)
   - Goal: Direct library imports
   - Expected: 200-300MB saved
   - Risk: Medium (code refactoring)

2. Optimize base images
   - Evaluate: alpine alternatives
   - Current: python:3.12-slim (good choice)
   - Savings: Minimal (slim is already optimized)

**Estimated Total Savings**: ~200-300MB

---

## 5. IMPLEMENTATION CHECKLIST

### Immediate Actions (Today)
- [ ] Create `.dockerignore`
- [ ] Remove torch/torchaudio from requirements-chainlit.txt
- [ ] Test Chainlit image build
- [ ] Document PyTorch removal rationale

### Week 1 Actions
- [ ] Replace openai-whisper with faster-whisper in requirements-api.txt
- [ ] Update whisper import statements in API code
- [ ] Test faster-whisper integration
- [ ] Add site-packages cleanup to all Dockerfiles
- [ ] Test API image build and functionality

### Week 2 Actions
- [ ] Decide on crawler approach (lightweight HTML vs crawl4ai no-browser)
- [ ] Implement chosen approach
- [ ] Test crawler image build
- [ ] Audit langchain-community usage

### Week 3-4 Actions
- [ ] Refactor to replace langchain-community imports
- [ ] Final testing across all services
- [ ] Document changes in deployment guide
- [ ] Create optimization summary for PR

---

## 6. VALIDATION & TESTING

### Build Verification
```bash
# API Image
docker build -f Dockerfile.api -t xoe-novai-api:optimized .
docker images | grep xoe-novai-api
# Expected: ~1.2GB (from 2.84GB)

# Chainlit Image
docker build -f Dockerfile.chainlit -t xoe-novai-ui:optimized .
docker images | grep xoe-novai-ui
# Expected: ~280MB (from 800MB)

# Crawler Image
docker build -f Dockerfile.crawl -t xoe-novai-crawler:optimized .
docker images | grep xoe-novai-crawler
# Expected: ~650MB (from 1.3GB)
```

### Functionality Testing
```bash
# Test API imports
docker run --rm xoe-novai-api:optimized python -c "
import faiss
import fastapi
import faster_whisper
import redis
print('✓ All imports successful')
"

# Test Chainlit (no PyTorch)
docker run --rm xoe-novai-ui:optimized python -c "
import chainlit
import httpx
print('✓ Chainlit ready (no PyTorch)')
"

# Test Crawler
docker run --rm xoe-novai-crawler:optimized python -c "
import httpx
import beautifulsoup4
import trafilatura
print('✓ Crawler dependencies ready')
"
```

### Performance Testing
```bash
# Faster-whisper vs openai-whisper
docker run --rm -v $(pwd)/test_audio:/audio xoe-novai-api:optimized python -c "
from faster_whisper import WhisperModel
import time

model = WhisperModel('base', device='cpu', compute_type='int8')
start = time.time()
segments, _ = model.transcribe('/audio/sample.mp3')
print(f'Transcribed in {time.time()-start:.2f}s')
"
```

---

## 7. RISK ASSESSMENT & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Breaking Chainlit without PyTorch | Low | High | Test UI thoroughly |
| faster-whisper API incompatibility | Low | Medium | Keep old code path, gradual migration |
| Crawler feature loss | Medium | Medium | Comprehensive testing; Option B if needed |
| Build time increase | Low | Low | Utilize layer caching |
| Runtime performance regression | Low | Low | Benchmark before/after |

---

## 8. EXPECTED OUTCOMES

### Target Image Sizes (Post-Optimization)
| Service | Before | After | Reduction | Status |
|---------|--------|-------|-----------|--------|
| API | 2.84 GB | 1.1 GB | **61%** | 🎯 TARGET |
| Chainlit | 800 MB | 280 MB | **65%** | 🎯 TARGET |
| Crawler | 1.3 GB | 620 MB | **52%** | 🎯 TARGET |
| **TOTAL** | **4.94 GB** | **2.0 GB** | **60%** | **🎯 TARGET** |

### Benefits
1. **Deployment**: 3x faster image pull, 4x less storage
2. **Scaling**: Faster container startup, reduced resource usage
3. **Maintainability**: Cleaner dependencies, fewer vulnerabilities
4. **Performance**: faster-whisper provides 3-4x speed improvement

---

## 9. REFERENCES & RESOURCES

### Useful Tools
- **Docker Analyzer**: `dive` (scan image layers)
  ```bash
  dive xoe-novai-api:optimized
  ```

- **Layer History**: `docker history`
  ```bash
  docker history xoe-novai-api:optimized --human --no-trunc
  ```

- **Disk Space**: `docker system df`

### Documentation
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Multi-stage Builds: https://docs.docker.com/build/building/multi-stage/
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://opennmt.net/CTranslate2/

---

## 10. NEXT STEPS

1. **Review** this document with the team
2. **Prioritize** based on effort vs. impact
3. **Create** GitHub issues for each optimization task
4. **Assign** work to team members
5. **Track** progress in project board
6. **Test** thoroughly before merging
7. **Document** changes in deployment guide

---

**Created**: 2026-01-09  
**Updated**: [To be updated as optimizations are completed]  
**Owner**: DevOps/Engineering Team
