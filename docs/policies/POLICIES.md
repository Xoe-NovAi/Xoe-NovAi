---
status: active
last_updated: 2026-01-04
owners:
  - team: policy
tags:
  - policy
---

# Project Policies

**Last updated:** 2026-01-04

These are mandatory constraints and rules the project follows. All contributors (human and AI) must follow these.

1. No Torch in default installation
   - Primary installation artifacts (requirements files and wheelhouse) MUST NOT include `torch` or its wheels.
   - GPU/torch is allowed only as an optional configuration and must be documented in `docs/policies/` with explicit install steps and a separate optional requirements file (e.g., `requirements-gpu.txt`).

2. No Telemetry
   - No code or third-party dependency may phone home or collect telemetry by default.
   - Any dependency that includes telemetry must be disabled or removed. If telemetry is unavoidable, document it and get explicit approval.

3. Documentation Strategy Enforcement
   - Canonical docs live under `docs/`.
   - Top-level directory should not accumulate ad-hoc docs; any `.md` created at the root must be moved to `docs/` or be an allowed root document (`README.md`, `LICENSE`, `CHANGELOG.md`).

4. Doc Frontmatter & Status
   - All major docs should include frontmatter with `status` (draft|active|archived) and `last_updated`.
   - `archived` docs go under `docs/archive/` with an archive index explaining why archived.

5. Pull Request & Review
   - Docs changes that alter policy or behavior require two maintainer approvals.
   - Small fixes (typos, grammar) can be merged by a single maintainer.

6. Automated Checks
   - Run markdown linting and a lightweight policy check on PRs (examples: no `torch` keyword in primary requirements, no `telemetry` in dependencies).

7. Templates
   - Use `docs/templates` for any new major document type to ensure consistency.

8. AI assistance and Copilot
   - AI (Copilot) may be used to draft docs, but a human must review and sign-off.
   - Policy edits suggested by AI require explicit human approval and a `policy-review` label on PR.

9. Archival & Deprecation
   - When archiving a doc, create an entry in `docs/archive/INDEX.md` with: filename, date archived, reason, and replacement link if any.

10. Emergency Overrides
   - For emergency reasons (security, licensing), maintainers may temporarily relax a policy but must document the reason and file an issue for permanent resolution.

