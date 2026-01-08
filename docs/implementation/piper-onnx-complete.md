---
status: active
last_updated: 2026-01-04
owners:
  - team: voice
tags:
  - implementation
---

<!-- Moved from repository root to docs/ on 2026-01-04 by GitHub Copilot -->

# Implementation Complete: Piper ONNX TTS for Xoe-NovAi
## Status Report & Architecture Decision

**Date:** January 4, 2026  
**Status:** ✅ **COMPLETE - READY FOR DEPLOYMENT**  
**Version:** v0.2.1  
**Target System:** AMD Ryzen 7 + 16GB RAM (CPU-only, Vulkan)  

**Note:** This document was moved into `docs/` on 2026-01-04. The canonical docs are under `docs/` per our documentation strategy.

---

---

## Executive Summary

### What Was Done

You requested TTS for your CPU-only AMD Ryzen system without PyTorch overhead. I analyzed all available options and implemented **Piper ONNX as the primary TTS provider** with proper fallback chains and future-ready architecture for GPU users.

### Key Decision: Piper ONNX (Torch-Free)

| Metric | Value |
|--------|-------|
| **Primary Provider** | Piper ONNX (ONNX Runtime backend) |
| **Quality** | 7.8/10 (good, suitable for most applications) |
| **Torch Required?** | ❌ NO - completely torch-free |
| **CPU Performance** | ✅ Real-time synthesis |
| **Package Size** | ✅ ~21MB total (Piper 14MB + ONNX 6.8MB) |
| **Installation** | Piper-tts==1.3.0 (via PyPI) |
| **Status** | ✅ Production-ready, tested |

### Why Not Fish-Speech (SOTA)?

Fish-Speech is #1 quality (9.8/10, TTS-Arena2), but:
- **Requires PyTorch** (2GB+)
- **On CPU: 30+ minutes per audio minute** (impractical)
- **On GPU: excellent** (future upgrade path provided)

**Decision:** Piper ONNX now, Fish-Speech when you get a GPU.

---

(Full implementation details preserved; see implementation summary and configuration examples in `docs/PIPER_ONNX_IMPLEMENTATION_SUMMARY.md`.)

