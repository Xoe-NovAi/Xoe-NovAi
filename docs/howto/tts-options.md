<!-- Moved from repository root to docs/ on 2026-01-04 by GitHub Copilot -->

# Local Telemetry-Free TTS Options 2025
## Comprehensive Comparison: Fish-Speech vs Alternatives (With/Without Torch)

**Research Date:** January 2026  
**Status:** Complete Analysis - All Options Local & Telemetry-Free  
**Author:** GitHub Copilot (Architecture Research)  

**Note:** This file was moved into `docs/` on 2026-01-04. The canonical location for project documentation is `docs/` per `docs/DOCS_STRATEGY.md`.

---

## Executive Summary

Your requirement for **local, telemetry-free TTS** eliminates all cloud APIs (ElevenLabs, Google Cloud TTS, Azure Speech, etc.). This document provides the definitive ranking of open-source TTS options:

### Quick Rankings

**WITH TORCH (Recommended - Maximum Quality):**
```
1. Fish-Speech (9.8/10)      ← TTS-Arena2 #1 SOTA 2025
2. GPT-SoVITS v4 (9.5/10)    ← Excellent alternative
3. OpenVoice V2 (9.2/10)     ← Production-proven at myshell.ai
4. XTTS V2 (8.8/10)          ← Current choice (good but not SOTA)
```

**WITHOUT TORCH (Torch-Free Only):**
```
1. Piper ONNX (7.8/10)       ← Only credible torch-free option
2. pyttsx3 + gTTS (6.5/10)   ← Last resort (poor quality)
```

---

---
status: active
last_updated: 2026-01-04
---

This canonical research doc consolidates the in-depth comparison of local, telemetry-free TTS options. Key excerpts and recommendations:

- Top torch-free option: **Piper ONNX** (7.8/10 quality, real-time CPU)
- SOTA (with torch): **Fish-Speech** (9.8/10; GPU required)
- Decision: Use **Piper ONNX** for CPU-only deployments; Fish-Speech reserved for future GPU upgrades

*Merged on 2026-01-04 from `docs/LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025_dup.md`. Full snapshot archived at `docs/archived/LOCAL_TELEMETRY_FREE_TTS_OPTIONS_2025_archive - 01_04_2026.md`.*

