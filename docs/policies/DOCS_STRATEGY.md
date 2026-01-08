# Documentation Strategy for Xoe-NovAi

**Last updated:** 2026-01-09 (Reorganized structure implemented)

Purpose
- Provide a single, maintainable source of truth for project documentation.
- Make it easy for humans and the AI (Copilot/GitHub Copilot) to find, update, and enforce policies.
- Enforce project-wide policies (No Torch, No Telemetry, documentation workflow).

Principles
- Single source of truth: canonical docs live under `docs/`.
- Keep root directory minimal: only `README.md`, `LICENSE`, and operational scripts.
- Immutable history: archived/ contains archived or superseded docs (never deleted).
- Policy-first: project policies are explicit in `docs/POLICIES.md` and enforced by review.
- Machine-friendly: each doc should include structured metadata (YAML frontmatter) when appropriate.

Current Structure (Implemented 2026-01-09)
```
/docs
  /archive              # Historical archive (duplicates, old-versions, sessions, historical)
  /design               # Architecture & strategy documents
  /howto                # Step-by-step guides and quickstarts
  /implementation       # Phase guides, checklists, code skeletons
  /policies             # Project policies and governance
  /reference            # Technical references and blueprints
  /releases             # Changelogs and release notes
  /runbooks             # Operational guides and live updates
  /templates            # Document templates
  README.md             # Main navigation hub
  START_HERE.md         # Quick onboarding guide
  CHANGES.md            # Top-level changes log
```

Folder Layout and Purpose
- `reference/` — Technical references, architecture blueprints, API specs, canonical guides
- `howto/` — Step-by-step guides, quick starts, setup instructions, migration guides
- `design/` — Architectural design docs, strategies, optimization plans, research-backed approaches
- `implementation/` — Phase-by-phase implementation guides, checklists, code skeletons, execution plans
- `runbooks/` — Operational guides, troubleshooting, live updates, build tracking
- `releases/` — Changelogs, release notes, version summaries, delivery reports
- `policies/` — Project policies, documentation strategy, ownership, governance
- `templates/` — Document templates and contributor checklists
- `archive/` — Superseded docs organized by type (duplicates, old-versions, sessions, historical)

Filenames & Frontmatter
- Filenames: use UPPER_SNAKE or kebab-case consistently, no spaces.
- Frontmatter (optional YAML) for each major doc:
  ```yaml
  ---
  title: ""
  status: draft|active|archived
  authors: ["Name"]
  last_updated: 2026-01-04
  tags: [policy, tts, piper]
  ---
  ```
- Include `status` to make automation easy (e.g., archive scripts).

Templates
- `docs/templates/guide_template.md` — for new how-to guides.
- `docs/templates/design_template.md` — for architecture/design docs.
- `docs/templates/release_notes_template.md` — for release notes.

Policies and Enforcement
- `docs/POLICIES.md` defines mandatory project rules (see file).
- PRs that change code or dependencies must include docs updates when behavior changes.
- CI: Add lightweight checks (MD lint, presence of frontmatter, no forbidden keywords) optionally via GitHub Actions.

VS Code & Copilot Workflow (for human + AI collaboration)
- Use `docs/` as working area—open `docs/` folder in VS Code.
- Use Copilot to propose edits, but require a human review for policy changes.
- Suggest the following VS Code extensions: Markdown All in One, Markdownlint, GitLens, GitHub Copilot.
- Use `editor.formatOnSave` and `prettier`/`markdownlint` rules.

Pull Request & Review Process
- Doc changes must include: short PR description, link to issue (if any), and checklist:
  - [ ] Added/updated frontmatter
  - [ ] Updated docs index `docs/README.md` if needed
  - [ ] Tests/lint passed
  - [ ] Policy changes reviewed and approved
- For critical policy updates (e.g., toggling No Torch), require two maintainers sign-off.

Maintenance & Cadence
- Weekly doc review by maintainers (short triage of outstanding doc PRs).
- Quarterly audit of `docs/archive` to prune or re-activate items.

Automation ideas
- GitHub Action that checks:
  - All top-level `.md` files are either whitelisted (`README.md`, `CHANGELOG.md`) or located in `docs/`.
  - `docs/POLICIES.md` not changed without `policy-review` label.
  - No `torch` occurrences in primary requirements (or flag to PR reviewers).

Onboarding New Contributors
- Add `docs/templates/contributing_doc.md` that includes doc writing guidelines.
- Provide a `docs/quick_start_for_contributors.md` explaining the docs workflow.

Document Ownership
- Keep a `docs/OWNERS.md` listing maintainers for each section (design, howto, reference, policies).

Organization Status (2026-01-09)
- ✅ Complete folder structure implemented
- ✅ All 110+ documents organized into categories
- ✅ Index files (README.md) created for each folder
- ✅ Duplicates archived in `archive/duplicates/`
- ✅ Historical content preserved in `archive/`
- ✅ Top-level reduced from 79 files to 4 essential files
- ✅ Navigation updated in `docs/README.md`
- ✅ Templates available in `templates/`

See `docs/ORGANIZATION_COMPLETE.md` for complete migration summary.

