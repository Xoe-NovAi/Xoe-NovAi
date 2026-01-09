---
status: active
last_updated: 2026-01-08
category: policy
---

# Version Management Policy for Xoe-NovAi

**Standardized approach for tracking and updating version information across all project files.**

---

## 🎯 Overview

This policy establishes best practices for version management across the Xoe-NovAi codebase. Proper version tracking ensures consistency, enables reliable deployments, and maintains clear release history.

---

## 📋 Version Format Standards

### Semantic Versioning (SemVer)
```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

#### Components
- **MAJOR**: Breaking changes (API incompatibilities)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)
- **PRERELEASE**: `alpha`, `beta`, `rc` (release candidates)
- **BUILD**: Build metadata (timestamps, commit hashes)

#### Examples
- `0.1.4` - Stable release
- `0.1.5-alpha` - Pre-release
- `0.1.5-rc.1` - Release candidate
- `0.1.5+20260108` - Build metadata

---

## 📁 Files Requiring Version Updates

### Core Configuration Files

#### 1. `config.toml`
```toml
[metadata]
stack_version = "v0.1.5"          # ← Primary version source
release_date = "2026-01-08"       # ← Update release date
codename = "Voice Integration"    # ← Update codename
```

#### 2. `config_loader.py`
```python
# Validation checks - update expected versions
if version not in ["v0.1.4-stable", "v0.1.5"]:
    logger.warning(f"Unexpected stack_version: {version}")
```

#### 3. `versions.toml`
```toml
# Dependency versions (updated separately from stack version)
[versions]
# Version constraints for Python packages
```

### Build & Container Files

#### 4. `Dockerfile.*`
```dockerfile
LABEL maintainer="Xoe-NovAi Team"
LABEL version="0.1.5"                    # ← Update version label
LABEL description="Xoe-NovAi v0.1.5"    # ← Update description
```

#### 5. `Makefile`
```makefile
# Xoe-NovAi Makefile (Full Version)
# Last Updated: 2026-01-08 (Voice-to-Voice + Build Tracking v0.1.5)
```

### Documentation Files

#### 6. `docs/STACK_STATUS.md`
```markdown
**Stack Version:** v0.1.5
**Release Date:** 2026-01-08
```

#### 7. `docs/README.md`
```markdown
**Last Updated:** 2026-01-08
**Stack Version:** v0.1.5
```

#### 8. `docs/releases/v0.1.5.md` *(New file)*
```markdown
# Xoe-NovAi v0.1.5 Release Notes

## Overview
Voice-to-Voice Conversation System & Build Optimization Enhancement

## Release Date
2026-01-08

## Major Features
- Voice-to-Voice conversation with <500ms latency
- 85% faster Docker builds with wheelhouse caching
- Enhanced Makefile with 27 comprehensive targets
```

### Code Files

#### 9. `app/XNAi_rag_app/main.py`
```python
title="Xoe-NovAi RAG API",
version="0.1.5",  # ← Update FastAPI version
```

#### 10. `setup.py` / `pyproject.toml` *(If exists)*
```python
version = "0.1.5"
```

---

## 🔄 Version Update Workflow

### Pre-Release Checklist

#### 1. Code Complete
- [ ] All features implemented and tested
- [ ] Integration tests passing
- [ ] Performance benchmarks completed
- [ ] Security review completed

#### 2. Documentation Updated
- [ ] Release notes written (`docs/releases/v0.1.5.md`)
- [ ] STACK_STATUS.md updated
- [ ] README.md updated
- [ ] API documentation updated

#### 3. Version References Updated
- [ ] `config.toml` - metadata.stack_version
- [ ] `config_loader.py` - validation checks
- [ ] `Dockerfile.*` - version labels
- [ ] `Makefile` - header comments
- [ ] FastAPI app version
- [ ] Documentation files

### Release Process

#### Phase 1: Preparation (1-2 days)
```bash
# 1. Create release branch
git checkout -b release/v0.1.5

# 2. Update version in all files
# Use the checklist above

# 3. Run comprehensive tests
make test
make voice-test
make build-analyze

# 4. Update changelog
vim docs/releases/CHANGELOG.md
```

#### Phase 2: Validation (1 day)
```bash
# 1. Build and test containers
make build
make up
make health

# 2. Test voice functionality
make voice-test
make voice-up

# 3. Validate build tracking
make build-tracking
make build-report
```

#### Phase 3: Release (1 day)
```bash
# 1. Final version bump
# Update config.toml stack_version to "v0.1.5" (remove -alpha/beta)

# 2. Tag release
git tag -a v0.1.5 -m "Release v0.1.5: Voice Integration & Build Optimization"

# 3. Push to main
git checkout main
git merge release/v0.1.5
git push origin main --tags

# 4. Deploy to production
make build
make up
```

---

## 🏷️ Version Tracking Strategy

### Single Source of Truth
**`config.toml`** serves as the primary version source:
```toml
[metadata]
stack_version = "v0.1.5"  # ← Authoritative version
```

### Automated Propagation
Use scripts to update version across files:
```bash
# Example version update script
#!/bin/bash
NEW_VERSION="v0.1.5"
RELEASE_DATE="2026-01-08"

# Update config.toml
sed -i "s/stack_version = \".*\"/stack_version = \"$NEW_VERSION\"/" config.toml
sed -i "s/release_date = \".*\"/release_date = \"$RELEASE_DATE\"/" config.toml

# Update Dockerfiles
find . -name "Dockerfile*" -exec sed -i "s/LABEL version=.*/LABEL version=\"$NEW_VERSION\"/" {} \;

# Update documentation
find docs/ -name "*.md" -exec sed -i "s/v0\.1\.4-stable/$NEW_VERSION/g" {} \;
```

### Validation Checks
Config loader validates version consistency:
```python
def validate_config():
    version = config["metadata"]["stack_version"]
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version: {version}")
```

---

## 📊 Version History Tracking

### Release Tracking
Maintain version history in `docs/releases/`:

```
docs/releases/
├── v0.1.4-stable.md
├── v0.1.5.md          # ← New release notes
├── v0.1.6.md          # ← Future releases
└── CHANGELOG.md       # ← Cumulative changelog
```

### Git Tags
Use annotated tags for releases:
```bash
# Tag format: vMAJOR.MINOR.PATCH
git tag -a v0.1.5 -m "Release v0.1.5: Voice Integration & Build Optimization"
git push origin --tags
```

### CHANGELOG.md Format
```markdown
# Changelog

## [v0.1.5] - 2026-01-08
### Added
- Voice-to-Voice conversation system
- Build dependency tracking and wheel caching
- Enhanced Makefile with 27 targets

### Changed
- Improved build performance (85% faster)
- Updated Docker optimization strategies

### Fixed
- Memory limit issues in Chainlit
- Build tracking integration

## [v0.1.4-stable] - 2025-11-08
<!-- Previous release -->
```

---

## 🚨 Breaking Changes Policy

### Major Version Changes
- **Breaking API changes** → Increment MAJOR version
- **Database schema changes** → Increment MAJOR version
- **Configuration format changes** → Increment MAJOR version

### Minor Version Changes
- **New features** (backward compatible) → Increment MINOR version
- **New API endpoints** → Increment MINOR version
- **Configuration additions** → Increment MINOR version

### Patch Version Changes
- **Bug fixes** → Increment PATCH version
- **Security fixes** → Increment PATCH version
- **Performance improvements** → Increment PATCH version

### Pre-Release Versions
- **Alpha**: `0.1.5-alpha` - Early testing, may have bugs
- **Beta**: `0.1.5-beta` - Feature complete, stability testing
- **RC**: `0.1.5-rc.1` - Release candidate, final validation

---

## 🔍 Version Validation

### Automated Checks

#### 1. Build-time Validation
```bash
# In Dockerfile
RUN python3 -c "
import toml
config = toml.load('config.toml')
version = config['metadata']['stack_version']
print(f'Building Xoe-NovAi {version}')
assert version == 'v0.1.5', f'Version mismatch: {version}'
"
```

#### 2. Runtime Validation
```python
# In config_loader.py
def validate_config():
    config = load_config()
    version = config['metadata']['stack_version']

    # Check version format
    if not re.match(r'v\d+\.\d+\.\d+(-[\w\.\-]+)?(\+[\w\.\-]+)?$', version):
        raise ValueError(f'Invalid version format: {version}')

    # Check against supported versions
    supported = ['v0.1.4-stable', 'v0.1.5']
    if version not in supported:
        logger.warning(f'Version {version} not in supported list: {supported}')
```

#### 3. Documentation Validation
```bash
# Check version consistency across docs
grep -r "v0\.1\.[45]" docs/ | sort | uniq -c
```

---

## 📋 Version Update Checklist

### Pre-Version Bump
- [ ] All features implemented and tested
- [ ] Integration tests passing
- [ ] Performance benchmarks completed
- [ ] Security review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

### Version Update Tasks
- [ ] Update `config.toml` stack_version
- [ ] Update `config_loader.py` validation
- [ ] Update `Dockerfile.*` labels
- [ ] Update `Makefile` comments
- [ ] Update documentation files
- [ ] Update FastAPI version
- [ ] Create release notes

### Post-Version Bump
- [ ] Tag release in git
- [ ] Push tags to repository
- [ ] Update deployment scripts
- [ ] Notify stakeholders
- [ ] Monitor post-release

---

## 🔗 Related Documentation

- **Release Process**: `docs/releases/README.md`
- **Changelog**: `docs/releases/CHANGELOG.md`
- **Build System**: `docs/howto/makefile-usage.md`
- **Configuration**: `app/XNAi_rag_app/config_loader.py`

---

## 📞 Contact

**Version Management Questions:**
- **Technical Lead**: Version format and validation
- **DevOps Team**: Build and deployment versioning
- **Documentation Team**: Release notes and changelog

---

**Policy Version:** 1.0
**Effective Date:** 2026-01-08
**Review Date:** 2026-07-08 (6 months)
**Maintained By:** Xoe-NovAi Release Management Team