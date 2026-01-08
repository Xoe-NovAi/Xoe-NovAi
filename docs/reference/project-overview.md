---
status: active
last_updated: 2026-01-04
---

# Project Overview: Xoe-NovAi

This document provides a concise, technical overview of the Xoe-NovAi stack for developers and AI assistants.

## Quick start

```bash
git clone https://github.com/xoe-novai/xoe-novai.git
cd xoe-novai
docker compose up -d
```

## Goals
- Offline-first, telemetry-free, self-hosted RAG platform
- CPU-optimized for AMD Ryzen-class systems
- Modular: clear separation of STT/TTS, RAG, curation, and UI layers

## Tech stack
- FastAPI (API layer)
- Chainlit (interactive UI)
- FAISS (vector DB)
- Redis (caching and streams)
- Piper ONNX (TTS, torch-free)
- Faster Whisper (STT, torch-free)
- llama-cpp-python (GGUF inference)
- GGUF model format for local inference

## Data flows (high level)
- Input → Retrieval (FAISS) → Model routing via Pantheon → SSE output
- Curation pipeline: `/curate` triggers Crawl4ai → sanitization → save to `library/`

## Security & Privacy
- Zero telemetry by default (env flags enforced)
- Non-root containers; capability dropping
- Crawl safety via domain-anchored regex and script stripping
- GDPR/HIPAA considerations in data handling

## Performance targets (Ryzen 7)
- Token/s: target 15–25
- RAM: <6GB
- Latency p95: <1s

## Where to find more details
- Voice impl & TTS decisions: `PIPER_ONNX_IMPLEMENTATION_SUMMARY.md`
- Runbook & live updates: `UPDATES_RUNNING.md`
- Deployment & Docker optimizations: `DOCKER_OPTIMIZATION_SUMMARY.md`
- Changelog & release notes: `CHANGELOG.md`


*If you need a more narrative or creative description of the project, see `README_dup.md` (archived/optional). For everyday development and AI assistant use, prefer this technical overview.*