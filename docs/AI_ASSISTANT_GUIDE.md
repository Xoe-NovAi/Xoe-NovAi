---
status: active
last_updated: 2026-01-09
category: meta
---

# AI Assistant Guide

**Purpose:** Help AI coding assistants efficiently navigate and use Xoe-NovAi documentation.

---

## 🎯 Quick Strategy

### Context Window Management
- **Don't load entire docs/ directory** - too many files (110+)
- **Load specific category folders** based on your task
- **Use index files** (README.md) for quick overviews
- **Focus on relevant subfolders** when possible

### Recommended Approach
1. **Start:** Read `docs/STACK_STATUS.md` for current stack (SINGLE SOURCE OF TRUTH)
2. **Then:** Read `docs/README.md` for navigation
3. **Identify:** Which category folder matches your task
4. **Load:** Only that category folder + its README.md
5. **Deep dive:** Load specific files as needed

### Critical: Always Check STACK_STATUS.md First
- **Before making assumptions** about the stack
- **Before suggesting implementations**
- **Before updating documentation**
- **STACK_STATUS.md** is the single source of truth for current implementations

---

## 📁 Category Guide for AI Assistants

### When You Need: Technical Specifications
**Load:** `docs/reference/` folder
- **Key files:** blueprint.md, condensed-guide.md, architecture.md
- **Use for:** Understanding system architecture, API specs, design patterns

### When You Need: Implementation Steps
**Load:** `docs/implementation/` folder
- **Key files:** phase-1.md, phase-1-5/, phase-2-3.md
- **Use for:** Implementing features, following phase guides, code skeletons

### When You Need: Setup/Configuration
**Load:** `docs/howto/` folder
- **Key files:** quick-start.md, docker-setup.md, voice-setup.md
- **Use for:** Setting up components, configuring services, migrations

### When You Need: Design Decisions
**Load:** `docs/design/` folder
- **Key files:** rag-refinements.md, enterprise-strategy.md, docker-optimization.md
- **Use for:** Understanding design decisions, architectural strategies, optimization

### When You Need: Current Status
**Load:** `docs/runbooks/` folder
- **Key files:** updates-running.md, build-tools.md
- **Use for:** Current work status, operational procedures, troubleshooting

### When You Need: Release Information
**Load:** `docs/releases/` folder
- **Key files:** CHANGELOG.md, v0.1.4-stable.md
- **Use for:** Understanding what changed, version history, release notes

### When You Need: Project Rules
**Load:** `docs/policies/` folder
- **Key files:** POLICIES.md, DOCS_STRATEGY.md
- **Use for:** Project policies, documentation standards, ownership

---

## 🔍 Efficient Search Strategy

### Step 1: Identify Task Type
- **Implementation?** → `implementation/`
- **Setup/Config?** → `howto/`
- **Architecture?** → `reference/` or `design/`
- **Operations?** → `runbooks/`
- **Policies?** → `policies/`

### Step 2: Load Category Folder
Load the entire category folder (usually 5-15 files) rather than individual files.

### Step 3: Use Index Files
Each category has a README.md with:
- Quick summaries
- File purposes
- Read time estimates
- Use case guidance

### Step 4: Load Specific Files
Only load specific files when you need detailed information.

---

## 📊 File Size Guidelines

| Category | Typical Files | Total Size | Load Strategy |
|----------|---------------|------------|--------------|
| reference/ | 6 files | ~500KB | Load all for architecture tasks |
| howto/ | 13 files | ~300KB | Load specific guide for task |
| design/ | 11 files | ~800KB | Load all for design decisions |
| implementation/ | 10 files | ~600KB | Load phase-specific files |
| runbooks/ | 5 files | ~200KB | Load updates-running.md first |
| releases/ | 9 files | ~400KB | Load CHANGELOG.md + specific version |
| policies/ | 4 files | ~50KB | Load all (small, important) |

**Recommendation:** Load entire category folder if <500KB, otherwise load specific files.

---

## 🚫 What NOT to Load

### Avoid Loading:
- ❌ Entire `docs/` directory (110+ files, too large)
- ❌ `archive/` folder (historical, not needed for active work)
- ❌ Multiple category folders simultaneously (unless necessary)
- ❌ Duplicate files (all in archive/duplicates/)

### Instead:
- ✅ Load one category folder at a time
- ✅ Use index files for navigation
- ✅ Load specific files when needed
- ✅ Reference archive only for historical context

---

## 💡 Best Practices

### 1. Start Small
Begin with the category README.md to understand what's available.

### 2. Load Incrementally
Load category folder → identify specific files → load those files.

### 3. Use Indexes
Category README.md files provide quick summaries and navigation.

### 4. Focus on Active Docs
Archive folder is for historical reference only.

### 5. Check Frontmatter
Files have YAML frontmatter with status, tags, last_updated.

---

## 📋 Common Tasks & File Mapping

| Task | Primary Folder | Key Files |
|------|----------------|-----------|
| Understand architecture | reference/ | blueprint.md, architecture.md |
| Set up Docker | howto/ | docker-setup.md |
| Implement Phase 1.5 | implementation/ | phase-1-5/index.md |
| Fix a bug | runbooks/ | updates-running.md |
| Understand design | design/ | rag-refinements.md |
| Check policies | policies/ | POLICIES.md |
| See what changed | releases/ | CHANGELOG.md |

---

## 🎯 Quick Reference

**Navigation Hub:** `docs/README.md`  
**Onboarding:** `docs/START_HERE.md`  
**This Guide:** `docs/AI_ASSISTANT_GUIDE.md` (you are here)

**Category Indexes:**
- `docs/reference/README.md`
- `docs/howto/README.md`
- `docs/design/README.md`
- `docs/implementation/README.md`
- `docs/runbooks/README.md`
- `docs/releases/README.md`
- `docs/policies/README.md`

---

**Last Updated:** 2026-01-09  
**Purpose:** Optimize AI assistant efficiency and context window usage

