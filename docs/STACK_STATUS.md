---
status: active
last_updated: 2026-01-09
category: reference
auto_update: true
---

# Current Stack Status

**Purpose:** Single source of truth for current Xoe-NovAi stack implementation status.  
**Update Frequency:** Updated automatically when stack changes are made.  
**For:** AI coding assistants and developers to understand current state.

---

## Core Stack Components

### LLM Backend
- **Status:** ✅ Implemented
- **Provider:** llama-cpp-python (native GGUF)
- **Format:** GGUF quantized models
- **Version:** v0.1.4-stable

### Vector Database
- **Status:** ✅ Implemented
- **Primary:** FAISS (Meta/Facebook)
- **Future:** Qdrant (migration guide available)
- **Version:** Current

### Speech-to-Text (STT)
- **Status:** ✅ Implemented
- **Primary:** Faster Whisper v1.2.1 (SYSTRAN)
- **Backend:** CTranslate2 v4.0+
- **Models:** tiny, base, small, medium, large-v3, distil-large-v3
- **GPU:** CUDA 12.x with cuBLAS + cuDNN 9
- **Performance:** 4x faster than OpenAI Whisper

### Text-to-Speech (TTS)
- **Status:** ✅ Implemented
- **Primary:** Piper ONNX (torch-free, CPU-optimized)
  - Repository: rhasspy/piper
  - Quality: 7.8/10
  - Latency: Real-time CPU synthesis (<100ms)
  - Footprint: ~21MB total
  - Languages: 50+ supported
- **Fallback:** XTTS V2 (torch-dependent, GPU-preferred)
  - Repository: coqui-ai/TTS
  - Quality: Production-grade
  - Latency: <200ms (GPU)
  - Features: Voice cloning available
- **Future:** Fish-Speech (SOTA, GPU-required, 9.8/10 quality)

### Design Patterns
- **Status:** ✅ All 5 patterns implemented
- **Pattern 1:** Import Path Resolution (sys.path.insert)
- **Pattern 2:** Retry Logic with Exponential Backoff (tenacity)
- **Pattern 3:** Non-Blocking Subprocess (subprocess.Popen)
- **Pattern 4:** Batch Checkpointing (Atomic writes, fsync, Redis)
- **Pattern 5:** Circuit Breaker (pybreaker library, fail_max=3, reset_timeout=60s)

### Services
- **Status:** ✅ 4 persistent services + healthcheck
- **Services:**
  - FastAPI RAG service
  - Chainlit UI
  - CrawlModule
  - Curation Worker
  - Healthcheck (8 targets)

### Ingestion System
- **Status:** ✅ Enhanced (Enterprise-Grade)
- **Location:** app/XNAi_rag_app/ingest_library.py
- **Features:**
  - Scholarly text curation with classical language detection
  - Multi-domain support (8 domains: science, technology, occult, spiritual, astrology, esoteric, science_fiction, youtube)
  - Domain knowledge base construction for LLM experts
  - Citation network analysis and cross-referencing
  - Hardware optimization for AMD Ryzen 7 5700U (6 cores, 12GB memory)
  - Enterprise error handling and quality assurance
  - 11 library APIs with enhanced scholarly enrichment

### Configuration
- **Status:** ✅ Two-tier system
- **Default:** config.toml
- **Overrides:** .env
- **Sections:** 23 configuration sections

### Build System
- **Status:** ✅ 3-stage offline build
- **Components:** Makefile, wheelhouse, Docker
- **Mode:** Offline-first

---

## Implementation Phases

### Phase 1
- **Status:** ✅ Complete
- **Components:** Metadata Enricher, Semantic Chunker, Delta Detector, Groundedness Scorer
- **Impact:** 25-40% precision improvement

### Phase 1.5
- **Status:** ✅ Complete
- **Components:** Quality Scoring, Specialized Retrievers, Query Router
- **Impact:** +10-15% precision improvement

### Phase 2-3
- **Status:** 🔄 In Progress / Planned
- **Components:** Multi-Adapter Retrieval, LLM Reranking, Monitoring, Versioning

---

## Technology Decisions

### TTS Decision (2026-01-09)
- **Decision:** Piper ONNX as primary (torch-free)
- **Rationale:** 
  - No PyTorch dependency (reduces footprint)
  - Real-time CPU synthesis (suitable for Ryzen 7)
  - Small footprint (~21MB)
  - Good quality (7.8/10)
- **Fallback:** XTTS V2 for GPU systems or voice cloning needs
- **Future:** Fish-Speech when GPU available

### LLM Decision
- **Decision:** llama-cpp-python (native GGUF)
- **Rationale:**
  - No external dependencies
  - Offline-first
  - CPU-optimized
  - GGUF format support

### Vector DB Decision
- **Current:** FAISS
- **Future:** Qdrant (migration guide available)
- **Rationale:** FAISS for current needs, Qdrant for scale

---

## Version Information

- **Stack Version:** v0.1.4-stable
- **Release Date:** 2025-11-08
- **Last Updated:** 2026-01-09
- **Documentation Version:** v0.2.1

---

## Quick Reference

| Component | Status | Version | Location |
|-----------|--------|---------|----------|
| LLM Backend | ✅ | llama-cpp-python | app/XNAi_rag_app/dependencies.py |
| STT | ✅ | Faster Whisper 1.2.1 | app/XNAi_rag_app/voice_interface.py |
| TTS Primary | ✅ | Piper ONNX | app/XNAi_rag_app/voice_interface.py |
| TTS Fallback | ✅ | XTTS V2 | app/XNAi_rag_app/voice_interface.py |
| Vector DB | ✅ | FAISS | app/XNAi_rag_app/dependencies.py |
| Design Patterns | ✅ | All 5 | Various files |
| Services | ✅ | 4 + healthcheck | docker-compose.yml |

---

## Documentation References

- **Architecture:** `reference/blueprint.md`
- **Implementation Guides:** `implementation/`
- **Voice Setup:** `howto/voice-setup.md`
- **TTS Options:** `howto/tts-options.md`
- **Release Notes:** `releases/CHANGELOG.md`

---

**Last Verified:** 2026-01-09  
**Next Review:** When stack changes are made  
**Maintained By:** Automated + Manual review

---

## Release Status

### v0.1.4-stable Release Readiness
- **Status:** 🔄 Pre-Release (85% Ready)
- **Audit Date:** 2026-01-09
- **Critical Issues:** 4 minor issues (estimated 1-2 hours to fix)
- **See:** [Release Readiness Audit](releases/v0.1.4-stable-release-readiness-audit.md)

### Project Status Tracking
- **See:** [Project Status Tracker](PROJECT_STATUS_TRACKER.md) for phase status and roadmap

