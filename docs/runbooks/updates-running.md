```markdown
# Project Runbook & Live Updates

---
status: active
last_updated: 2026-01-06
---

This is the canonical, active runbook for short-term session logs and live update notes.

Recent focus (summary):
- **Critical Security Fixes:** Implemented 5 major security vulnerability remediations (command injection, path traversal, Redis security, async framework, health check optimization) - see `docs/runbooks/security-fixes-runbook.md`
- **Documentation consolidation:** canonicalized root docs into `docs/`, removed duplicates, added templates and owners.
- **Wheelhouse build:** Offline wheel collection in progress—`WHEELHOUSE_BUILD_TRACKING.md` contains live status and verification steps.
- **Voice Implementation:** Piper ONNX integrated as primary TTS for CPU systems; implementation and testing notes in `PIPER_ONNX_IMPLEMENTATION_SUMMARY.md` and `IMPLEMENTATION_COMPLETE_PIPER_ONNX.md`.

Usage:
- Keep this file for active session notes (status, quick commands, blockers).
- For historical session logs and large session transcripts, see `docs/CHANGES.md` and `docs/CHANGELOG.md`.

Last updated: January 4, 2026

See also:
- `WHEELHOUSE_BUILD_TRACKING.md`
- `DOCS_STRATEGY.md`
- `POLICIES.md`

```
