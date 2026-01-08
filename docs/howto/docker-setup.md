# Docker Optimization Implementation Guide
## Xoe-NovAi Phase 1 v0.1.4-stable - QUICK START

**Status**: Ready for Implementation  
**Date**: 2026-01-09  
**Estimated Time**: 2-4 weeks  
**Expected Savings**: 2.94 GB (60% reduction) across all images

---

## 🚀 QUICK START (First 4 Hours)

### Step 1: Remove PyTorch from Chainlit (15 minutes)
**Impact**: 800MB → 280MB (65% reduction)

1. **Edit** `requirements-chainlit.txt`:
```bash
cd /home/arcana-novai/Documents/GitHub/Xoe-NovAi
nano requirements-chainlit.txt
```

2. **Remove these lines**:
```bash
torch>=2.2.0
torchaudio>=2.2.0
# torchvision>=0.17.0  # (if present)
```

3. **Verify change**:
```bash
grep -n "torch" requirements-chainlit.txt
# Should return nothing (empty)
```

4. **Build and test**:
```bash
docker build -f Dockerfile.chainlit -t xoe-novai-ui:test .
docker images xoe-novai-ui:test --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
# Expected: ~280-350MB (was 800MB)
```

---

### Step 2: Create `.dockerignore` Review (5 minutes)
**File exists at**: `/home/arcana-novai/Documents/GitHub/Xoe-NovAi/.dockerignore`

**Verify key exclusions**:
```bash
grep -E "(__pycache__|\.pyc|tests/|docs/)" .dockerignore
# Should see all these patterns

# Verify not excluding needed files:
grep -E "(requirements|Dockerfile)" .dockerignore
# Should return nothing (we KEEP these)
```

---

### Step 3: Replace Whisper with faster-whisper (30 minutes)

#### 3a. Update requirements-api.txt

**Locate and remove**:
```bash
nano requirements-api.txt
# Find and DELETE these lines (if present):
openai-whisper>=20231117  
transformers>=4.36.0
torch  # (any torch dependency)
```

**Add faster-whisper**:
```bash
cat >> requirements-api.txt << 'EOF'

# ============================================================================
# SPEECH-TO-TEXT (OPTIMIZED - replaced openai-whisper)
# ============================================================================
# faster-whisper: 3-4x faster, 50% smaller than openai-whisper
# Uses CTranslate2 for optimized CPU inference
faster-whisper==1.2.1
ctranslate2==4.6.2
EOF
```

#### 3b. Update API code imports

**Find all whisper references**:
```bash
find app/ -name "*.py" -exec grep -l "whisper\|from openai" {} \;
# Likely files: app/XNAi_rag_app/api_routes.py, app/services/transcription.py
```

**Update imports in found files**:
```python
# BEFORE:
from whisper import load_model
from openai import AzureOpenAI
import openai_whisper

# AFTER:
from faster_whisper import WhisperModel
# Remove openai imports if not using OpenAI API for other features
```

**Update transcription code**:
```python
# BEFORE:
model = load_model("base")
result = model.transcribe("audio.mp3")
text = result["text"]

# AFTER:
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.mp3")
text = " ".join([segment.text for segment in segments])
```

#### 3c. Test API image
```bash
docker build -f Dockerfile.api -t xoe-novai-api:test .
docker run --rm xoe-novai-api:test python -c "
from faster_whisper import WhisperModel
import ctranslate2
print('✓ faster-whisper imported successfully')
print(f'CTranslate2 version: {ctranslate2.__version__}')
"
```

---

### Step 4: Add Site-Packages Cleanup (45 minutes)

Update all three Dockerfiles to clean up unnecessary files in the runtime stage.

#### For Dockerfile.api (add in STAGE 2, after COPY --from=builder):

```dockerfile
# After copying site-packages:
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# ADD THIS BLOCK:
# Clean up unnecessary files to reduce image size
RUN find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type f -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12/site-packages -type f -name "*.pyo" -delete && \
    find /usr/local/lib/python3.12/site-packages -type d -name "*.dist-info/tests" -exec rm -rf {} + 2>/dev/null || true && \
    echo "✓ Site-packages cleanup complete (saved ~50-100MB)" && \
    du -sh /usr/local/lib/python3.12/site-packages
```

#### Apply to: 
- `Dockerfile.api` (STAGE 2)
- `Dockerfile.chainlit` (STAGE 2)
- `Dockerfile.crawl` (STAGE 2)

---

## 📊 VERIFICATION CHECKLIST

After implementing the quick start changes:

```bash
# 1. Check image sizes
docker images | grep -E "xoe-novai|test"

# Expected:
# API:       2.84GB → ~1.8-2.0GB (after faster-whisper + cleanup)
# Chainlit:  800MB  → ~280-350MB (after PyTorch removal)
# Crawler:   1.3GB  → unchanged (for now)

# 2. Verify no torch in Chainlit
docker run --rm xoe-novai-ui:test python -c "import torch" 2>&1 | grep -i "no module"
# Should show error (torch not found) ✓

# 3. Test faster-whisper
docker run --rm xoe-novai-api:test python -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny', device='cpu')
print('✓ faster-whisper working')
"

# 4. Verify Chainlit still works
docker run -d --name chainlit-test xoe-novai-ui:test chainlit run app.py
docker logs chainlit-test | grep -i "server running"
docker stop chainlit-test
# Should see server startup messages ✓

# 5. Check for remaining large files
docker run --rm xoe-novai-api:test du -sh /usr/local/lib/python3.12/site-packages
# Should be <800MB total (was likely 1.2-1.5GB)
```

---

## 📋 FOLLOW-UP TASKS (Week 2-3)

Once quick start is complete and tested:

### Task 1: Audit langchain-community (Medium Priority)
```bash
# Find all langchain imports
grep -r "from langchain_community" app/
grep -r "from langchain import" app/

# For each import, check if direct import is available:
# Example: Don't use langchain_community.embeddings.HuggingFaceEmbeddings
#          Instead: from huggingface_hub import hf_hub_download
```

### Task 2: Optimize Crawler (High Priority)
Two options:

**Option A (Recommended)**: Replace crawl4ai with trafilatura + beautifulsoup
```bash
# Update requirements-crawl.txt
cat requirements-crawl.txt | grep -v crawl4ai > requirements-crawl.txt.tmp
echo "trafilatura==2.0.0" >> requirements-crawl.txt.tmp
mv requirements-crawl.txt.tmp requirements-crawl.txt

# Update crawler code:
# from crawl4ai import AsyncWebCrawler
# to:
# from trafilatura import extract
```

**Option B**: Keep crawl4ai but disable browser
```bash
# Set environment variable in Dockerfile:
ENV CRAWL4AI_NO_BROWSER=true
```

### Task 3: Additional Cleanup (Optional)
- Remove unused transitive dependencies
- Analyze remaining heavy libraries with `docker history`
- Implement startup performance monitoring

---

## 🧪 FULL INTEGRATION TEST

After all changes, run comprehensive test:

```bash
#!/bin/bash
set -e

echo "🧪 Docker Optimization Integration Test"
echo "========================================"

# Build all images
echo "📦 Building images..."
docker build -f Dockerfile.api -t xoe-novai-api:optimized .
docker build -f Dockerfile.chainlit -t xoe-novai-ui:optimized .
docker build -f Dockerfile.crawl -t xoe-novai-crawler:optimized .

# Show sizes
echo ""
echo "📊 Final Image Sizes:"
docker images | grep optimized

# Test imports
echo ""
echo "✅ Testing API imports..."
docker run --rm xoe-novai-api:optimized python -c "
import faiss
import fastapi
import faster_whisper
import redis
print('✓ All API dependencies loaded')
"

echo "✅ Testing Chainlit imports..."
docker run --rm xoe-novai-ui:optimized python -c "
import chainlit
import httpx
try:
    import torch
    print('❌ PyTorch still present!')
    exit(1)
except ImportError:
    print('✓ PyTorch correctly removed')
"

echo "✅ Testing Crawler imports..."
docker run --rm xoe-novai-crawler:optimized python -c "
import httpx
import beautifulsoup4
print('✓ All crawler dependencies loaded')
"

echo ""
echo "🎉 All tests passed! Optimization complete."
```

---

## 📝 DOCUMENTATION

Update these files after changes:

1. **DOCKER_OPTIMIZATION_STRATEGY.md** ← Created above
2. **IMPLEMENTATION_STATUS.txt** (update summary)
3. **DEPLOYMENT_GUIDE.md** (if exists, add optimization notes)
4. **Commit message for Git**:
   ```
   docs: Docker image optimization (60% size reduction)
   
   - Remove PyTorch from Chainlit UI (unnecessary HTTP client)
   - Replace openai-whisper with faster-whisper (3x faster, 50% smaller)
   - Add site-packages cleanup to all Dockerfiles
   - Total size reduction: 4.94GB → ~2.0GB (60%)
   
   Images:
   - API:     2.84GB → ~1.8GB (36% reduction)
   - Chainlit: 800MB → 280MB (65% reduction)
   - Crawler: 1.3GB → ~620MB (52% reduction)
   ```

---

## ⚠️ ROLLBACK PROCEDURE

If issues arise:

```bash
# 1. Revert git changes
git checkout requirements-api.txt requirements-chainlit.txt Dockerfile.*

# 2. Rebuild old images
docker build -f Dockerfile.api -t xoe-novai-api:v0.1.4 .

# 3. Switch docker-compose to use old image
# Edit docker-compose.yml: change image tag back to :v0.1.4
```

---

## 📞 SUPPORT & QUESTIONS

For issues during implementation:

1. **Build fails**: Check Docker logs: `docker build --progress=plain`
2. **Import errors**: Verify package installed: `docker run --rm image pip show faster-whisper`
3. **Runtime errors**: Test in isolation: `docker run -it image python`
4. **Size not reducing**: Verify cleanup ran: `docker history image`

---

## 📈 SUCCESS METRICS

Target completion metrics:

- ✅ All images build successfully
- ✅ All services start and respond to health checks
- ✅ No test failures  
- ✅ Image size reductions within 10% of targets:
  - API: 1.8-2.2GB (target 1.2GB, acceptable 1.8-2.2GB)
  - Chainlit: 280-350MB (target 280MB)
  - Crawler: 600-700MB (target 600MB)
- ✅ Documentation updated
- ✅ Changes merged to main branch

---

**Created**: 2026-01-09  
**Estimated Duration**: 2-4 weeks  
**Quick Start Time**: 4 hours  
**Follow-up Time**: 5-10 hours

**Good Luck! 🚀**
