# Archived: WHEELHOUSE_BUILD_TRACKING

This file has been archived and consolidated.

- **Canonical (active):** `docs/WHEELHOUSE_BUILD_TRACKING.md` ✅
- **Archived snapshot:** `docs/archived/WHEELHOUSE_BUILD_TRACKING_archive - 01_04_2026.md` 📚

If you need the full historical version, open the archived snapshot above.
---

## Awaiting in Wheels (Priority)

### Piper ONNX Integration (CRITICAL)
- 🔄 **piper-tts==1.3.0** (14 MB, torch-free, PRIMARY TTS)
- 🔄 **onnxruntime>=1** (6.8 MB, ONNX Runtime backend)

### Voice Processing
- 🔄 **faster-whisper-1.2.1** (STT - Faster Whisper)
- 🔄 **librosa** (Audio processing)
- 🔄 **scipy** (Scientific computing)

### Core RAG Dependencies
- 🔄 **langchain-core** (RAG framework)
- 🔄 **langchain-community** (Community integrations)
- 🔄 **langchain-retriever** (Retrieval chains)

---

## Critical Verification Points (When Build Completes)

### ✅ Check 1: Piper-TTS Present
- Expected file: `piper-tts-1.3.0-py3-none-any.whl` (14 MB)
- Verify: No torch-related files inside
- Success indicator: Presence of piper_models.onnx (4.7 MB)

### ✅ Check 2: NO Torch in Wheelhouse
- Verify: `torch` NOT in wheelhouse directory
- Verify: No `torch-*.whl` files present
- Expected: Zero torch wheels (confirms our Piper ONNX decision)

### ✅ Check 3: ONNX Runtime Present
- Expected file: `onnxruntime-*.whl` (6.8 MB)
- Verify: C++ backend included
- Critical: This is Piper's inference backend (NOT torch)

### ✅ Check 4: Fast-Whisper Available
- Expected file: `faster-whisper-*.whl`
- Verify: CTranslate2 backend (not torch)
- Success: STT processing ready to go

### ✅ Check 5: Size Verification
- **Expected Total:** ~795 MB (no torch)
- **With torch:** ~2.5+ GB (avoided by using Piper!)
- **Space Saved:** ~1.7 GB by using Piper ONNX ✅

---

## Expected Final Wheelhouse Size Breakdown

```
├─ Core API (fastapi, pydantic, uvicorn):     150 MB
├─ RAG Stack (langchain, llama-index):        250 MB
├─ FAISS + deps (vector database):            100 MB
├─ Voice Processing (whisper, scipy):          50 MB
├─ TTS Stack (piper-tts + onnxruntime):        21 MB ✅ NO TORCH!
├─ Chainlit UI (web interface):                75 MB
└─ Other deps (Click, attrs, urllib3, etc):   150 MB
───────────────────────────────────────────────────────
TOTAL EXPECTED:                               ~795 MB
```

### Torch Comparison
- **Current approach (Piper ONNX):** ~800 MB
- **With PyTorch approach:** ~2,500+ MB
- **Savings:** ~1,700 MB (64% reduction!) 🎉

---

## What Happens When Build Completes

1. **Automatic:** Script will finish collecting all wheels
2. **Output:** Terminal will show "Wheelhouse build complete!"
3. **Directory:** `/wheelhouse/` will contain all `.whl` files
4. **Next Step:** Manual verification of critical packages
5. **Deployment:** Ready for offline installation on target systems

---

## Installation from Wheelhouse (When Ready)

Once build completes, offline installation will be possible:

```bash
# Install all packages from wheelhouse (no internet needed)
pip install --no-index --find-links=/home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/ \
    -r requirements-api.txt \
    -r requirements-chainlit.txt \
    -r requirements-crawl.txt \
    -r requirements-curation_worker.txt
```

---

## Estimated Timeline

| Task | Status | Time |
|------|--------|------|
| Build started | ✅ | 15:13 UTC |
| Current progress | 🔄 | ~73 wheels collected |
| Estimated completion | ⏳ | ~15:25-15:35 UTC (10-20 min total) |
| Manual verification | ⏳ | 2-3 minutes |
| Documentation update | ⏳ | 1 minute |

---

## Important Notes

✅ **Code Implementation:** COMPLETE  
✅ **Syntax Verification:** PASSED  
✅ **Dependency Analysis:** PASSED (confirmed no torch in runtime)  
🔄 **Wheelhouse Build:** IN PROGRESS  
   - Will include piper-tts==1.3.0 (torch-free)
   - Will verify zero torch in final distribution
⏳ **Functional Testing:** Deferred (piper import test when ready)  
⏳ **Docker Testing:** Deferred (as user requested)  

---

## Related Documentation

- [PIPER_ONNX_IMPLEMENTATION_SUMMARY.md](PIPER_ONNX_IMPLEMENTATION_SUMMARY.md) - Implementation guide
- [IMPLEMENTATION_COMPLETE_PIPER_ONNX.md](IMPLEMENTATION_COMPLETE_PIPER_ONNX.md) - Status report
- [LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025.md](LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025.md) - Research background

---

## Quick Commands for Monitoring

```bash
# Check wheelhouse directory
ls -lh /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/ | head -20

# Count wheels
ls -1 /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/ | wc -l

# Total size
du -sh /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/

# Check for piper-tts
ls -lh /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/piper*

# Check for torch (should find NOTHING)
ls -lh /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/torch*

# Check for onnxruntime
ls -lh /home/arcana-novai/Documents/GitHub/Xoe-NovAi/wheelhouse/onnx*
```

---

**Last Updated:** January 4, 2025 15:18 UTC  
**Next Review:** When build script completes (~15:25-15:35 UTC)
