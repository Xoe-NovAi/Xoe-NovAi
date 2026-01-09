---
status: active
last_updated: 2026-01-08
category: navigation
---

# Xoe-NovAi Documentation

**Welcome to the Xoe-NovAi documentation hub.**

This directory contains all project documentation, organized for efficient navigation by both humans and AI coding assistants.

---

## 🚀 Quick Start

**New to the project?** → Start with [`START_HERE.md`](START_HERE.md)

**Looking for something specific?** → Use the category indexes below

**AI Assistant?** → Focus on the relevant category folder for your task

---

## 📚 Documentation Structure

### [reference/](reference/) - Technical References
API references, architecture blueprints, technical specifications, and canonical guides.

**Key Documents:**
- [`blueprint.md`](reference/blueprint.md) - Complete v0.1.4-stable blueprint
- [`condensed-guide.md`](reference/condensed-guide.md) - Condensed technical guide
- [`architecture.md`](reference/architecture.md) - Architecture audit and design

**Use When:** You need technical specifications, API details, or architectural understanding.

---

### [howto/](howto/) - Step-by-Step Guides
Quick starts, setup guides, migration instructions, and operational procedures.

**Key Documents:**
- [`quick-start.md`](howto/quick-start.md) - Get started in 5 minutes
- [`docker-setup.md`](howto/docker-setup.md) - Docker deployment guide
- [`voice-setup.md`](howto/voice-setup.md) - Voice interface setup
- [`makefile-usage.md`](howto/makefile-usage.md) - Complete build system guide
- [`github-protocol-guide.md`](howto/github-protocol-guide.md) - GitHub workflow guide

**Use When:** You need to perform a specific task or set up a component.

---

### [design/](design/) - Architecture & Strategy
Design decisions, architectural strategies, optimization plans, and research-backed approaches.

**Key Documents:**
- [`rag-refinements.md`](design/rag-refinements.md) - Advanced RAG architecture
- [`enterprise-strategy.md`](design/enterprise-strategy.md) - Enterprise configuration strategy
- [`docker-optimization.md`](design/docker-optimization.md) - Docker optimization strategy

**Use When:** You need to understand design decisions, architectural patterns, or strategic planning.

---

### [implementation/](implementation/) - Implementation Guides
Phase-by-phase implementation guides, checklists, code skeletons, and execution plans.

**Key Documents:**
- [`phase-1.md`](implementation/phase-1.md) - Phase 1 implementation guide
- [`phase-1-5/`](implementation/phase-1-5/) - Phase 1.5 complete package
- [`phase-2-3.md`](implementation/phase-2-3.md) - Advanced implementation

**Use When:** You're implementing features or following development phases.

---

### [runbooks/](runbooks/) - Operational Runbooks
Live updates, troubleshooting guides, operational procedures, and build tracking.

**Key Documents:**
- [`security-fixes-runbook.md`](runbooks/security-fixes-runbook.md) - Critical security fixes (2026-01-06)
- [`updates-running.md`](runbooks/updates-running.md) - Live session updates
- [`docker-testing.md`](runbooks/docker-testing.md) - Docker testing procedures
- [`build-tools.md`](runbooks/build-tools.md) - Build system documentation

**Use When:** You're operating the system, debugging issues, or tracking ongoing work.

**New Runbooks:**
- [`ingestion-system-enhancements.md`](runbooks/ingestion-system-enhancements.md) - Scholarly curation and multi-domain support (2026-01-08)

---

### [releases/](releases/) - Release Information
Changelogs, release notes, version summaries, and delivery reports.

**Key Documents:**
- [`CHANGELOG.md`](releases/CHANGELOG.md) - Complete changelog
- [`v0.1.4-stable.md`](releases/v0.1.4-stable.md) - Production release summary
- [`v0.1.5.md`](releases/v0.1.5.md) - Phase 1.5 release notes

**Use When:** You need to understand what changed, when, and why.

---

### [policies/](policies/) - Project Policies
Project policies, documentation strategy, ownership, and governance.

**Key Documents:**
- [`POLICIES.md`](policies/POLICIES.md) - Project policies
- [`DOCS_STRATEGY.md`](policies/DOCS_STRATEGY.md) - Documentation strategy
- [`OWNERS.md`](policies/OWNERS.md) - Document ownership

**Use When:** You need to understand project rules, documentation standards, or find document owners.

---

### [archive/](archive/) - Historical Archive
Archived documents, duplicates, old versions, and historical session logs.

**Organization:**
- [`duplicates/`](archive/duplicates/) - Superseded duplicate files
- [`historical/`](archive/historical/) - Historical documentation versions
- [`code-review-sessions/`](archive/code-review-sessions/) - Archived code review sessions and patches

**Use When:** You need historical reference or understanding project evolution.

---

### [templates/](templates/) - Document Templates
Templates for creating new documentation following project standards.

**Use When:** Creating new documentation.

## 🎯 Role-Based Entry Points

### For Executives/Stakeholders
→ [`START_HERE.md`](START_HERE.md) → [`releases/delivery-complete.md`](releases/delivery-complete.md)

### For Technical Leads/Architects
→ [`reference/blueprint.md`](reference/blueprint.md) → [`design/rag-refinements.md`](design/rag-refinements.md)

### For Developers
→ [`howto/quick-start.md`](howto/quick-start.md) → [`implementation/phase-1.md`](implementation/phase-1.md)

### For Operations/DevOps
→ [`runbooks/updates-running.md`](runbooks/updates-running.md) → [`howto/docker-setup.md`](howto/docker-setup.md)

### For AI Coding Assistants
→ Focus on the relevant category folder for your task. Each folder has a README.md index.

---

## 📋 Documentation Standards

All documentation follows these standards:
- **Frontmatter:** YAML metadata with status, last_updated, tags
- **Naming:** kebab-case for active docs
- **Structure:** Clear headings, table of contents for long docs
- **Links:** Relative paths within docs/

See [`policies/DOCS_STRATEGY.md`](policies/DOCS_STRATEGY.md) for complete standards.

---

## 🔍 Finding Documents

### By Topic
- **Voice/TTS:** [`howto/voice-setup.md`](howto/voice-setup.md), [`design/audio-strategy.md`](design/audio-strategy.md)
- **Docker:** [`howto/docker-setup.md`](howto/docker-setup.md), [`design/docker-optimization.md`](design/docker-optimization.md)
- **RAG/Vector DB:** [`design/rag-refinements.md`](design/rag-refinements.md), [`howto/qdrant-migration.md`](howto/qdrant-migration.md)
- **Implementation:** [`implementation/`](implementation/) folder

### By Phase
- **Phase 1:** [`implementation/phase-1.md`](implementation/phase-1.md)
- **Phase 1.5:** [`implementation/phase-1-5/`](implementation/phase-1-5/)
- **Phase 2-3:** [`implementation/phase-2-3.md`](implementation/phase-2-3.md)

### By Type
- **Quick Start:** [`howto/quick-start.md`](howto/quick-start.md)
- **Architecture:** [`reference/blueprint.md`](reference/blueprint.md)
- **Strategy:** [`design/`](design/) folder
- **Operations:** [`runbooks/`](runbooks/) folder

---

## 📝 Contributing

When adding new documentation:
1. Use appropriate template from [`templates/`](templates/)
2. Place in correct category folder
3. Update relevant README.md index
4. Add frontmatter with metadata
5. Follow naming conventions (kebab-case)

See [`policies/DOCS_STRATEGY.md`](policies/DOCS_STRATEGY.md) for details.

---

## 📊 Current Stack Status

**For current stack implementation status, see [`STACK_STATUS.md`](STACK_STATUS.md).**

This is the single source of truth for:
- Current technology stack
- Implementation status
- Version information
- Technology decisions and rationale

**AI Assistants:** Always check `STACK_STATUS.md` first before making assumptions about the stack.

---

## 🔄 Recent Changes

- **2026-01-08:** Enterprise ingestion system with scholarly curation and multi-domain support implemented
- **2026-01-06:** Critical security fixes implemented - command injection, path traversal, Redis security, health check optimization
- **2026-01-09:** Complete documentation reorganization, STACK_STATUS.md created
- **2026-01-04:** Initial consolidation and archive structure
- **2026-01-03:** Phase 1.5 implementation package delivery

See [`releases/CHANGELOG.md`](releases/CHANGELOG.md) for complete history.

---

**Last Updated:** 2026-01-09  
**Maintained By:** Xoe-NovAi Team  
**Questions?** See [`policies/OWNERS.md`](policies/OWNERS.md) for document owners.
