---
status: active
last_updated: 2026-01-07
authors: ["Copilot"]
tags: [docker, build, troubleshooting, networking, dependencies]
---

# Docker Build Troubleshooting Runbook

**Last updated:** 2026-01-07

## Overview

This runbook documents solutions for common Docker build issues encountered in the Xoe-NovAi project, particularly focusing on network connectivity and dependency resolution problems.

## Issue: DNS Resolution Failures During apt-get update

### Problem Description
Docker builds fail with "Temporary failure resolving 'deb.debian.org'" during the `apt-get update` phase in both builder and runtime stages.

### Root Cause
- Network connectivity issues during Docker builds
- DNS resolution timeouts
- Container network isolation

### Solution: Retry Logic with Exponential Backoff

**Location:** `Dockerfile.api` (both builder and runtime stages)

**Implementation:**
```dockerfile
# Retry apt-get update with exponential backoff
for i in 1 2 3 4 5; do \
    if apt-get update 2>&1; then \
        echo "✓ apt-get update succeeded on attempt $i"; \
        break; \
    else \
        echo "⚠ apt-get update failed on attempt $i, retrying in $((i*5)) seconds..."; \
        sleep $((i*5)); \
    fi; \
    if [ $i -eq 5 ]; then \
        echo "❌ apt-get update failed after 5 attempts"; \
        exit 1; \
    fi; \
done
```

**Expected Result:**
- Builds complete successfully even with intermittent network issues
- Clear logging shows retry attempts and success/failure status
- Automatic recovery from temporary DNS failures

### Alternative Solutions Considered

1. **DNS Configuration**: Attempted to modify `/etc/resolv.conf` but failed because it's read-only in containers
2. **Docker DNS Options**: Could use `--dns` flag in docker build, but retry logic is more robust

## Issue: FastAPI Version Compatibility Conflicts

### Problem Description
API service built with FastAPI 0.120.4, but Chainlit UI requires FastAPI >=0.116.1,<0.117 for compatibility with version 2.8.3.

### Root Cause
- Version management system not properly syncing versions across requirements files
- Manual version overrides in requirements files bypassing version constraints

### Solution: Enforce Version Constraints

**Files Modified:**
- `requirements-api.txt`: Changed `fastapi==0.120.4` to `fastapi>=0.116.1,<0.117`
- Verified compatibility with `versions.toml` constraints

**Version Matrix:**
```
Component       | Version Required          | Status
---------------|--------------------------|--------
Chainlit        | 2.8.3                    | Fixed
FastAPI (API)   | >=0.116.1,<0.117         | ✅ Compatible
FastAPI (UI)    | >=0.116.1,<0.117         | ✅ Compatible
```

**Validation:**
- Docker build completes with FastAPI 0.116.2 (within compatible range)
- All validation checks pass in final image

## Issue: ImportLib Syntax Error in Validation Scripts

### Problem Description
Docker build fails with `AttributeError: module 'importlib' has no attribute 'util'` during image validation.

### Root Cause
Incorrect import syntax in Python validation commands within Dockerfile.

### Solution: Correct Import Syntax

**Before:**
```dockerfile
python3 -c 'import importlib; m = importlib.util.find_spec("llama_cpp"); ...'
```

**After:**
```dockerfile
python3 -c "import importlib.util; m = importlib.util.find_spec('llama_cpp'); ..."
```

## Issue: COPY Command Linting Errors

### Problem Description
Docker linting fails with "When using COPY with more than one source file, the destination must be a directory and end with a /".

### Root Cause
Incorrect COPY syntax for directory copying.

### Solution: Fix COPY Syntax

**Before:**
```dockerfile
COPY --from=builder /app/build-artifacts /app/build-logs || true
```

**After:**
```dockerfile
COPY --from=builder /app/build-artifacts /app/build-logs/
```

## Testing and Validation

### Build Verification
```bash
# Test API service build
docker build -f Dockerfile.api -t xnai-api:latest .

# Verify installed versions
docker run --rm xnai-api:latest python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

### Expected Output
```
✓ apt-get update succeeded on attempt 1
✓ FastAPI 0.116.2 ready
✓ LangChain ready
✓ FAISS ready
✓ llama-cpp-python present
✓ Redis client ready
```

## Prevention Measures

### Version Management
- Run `python3 versions/scripts/update_versions.py` regularly to sync versions
- Validate constraints with `python3 versions/scripts/update_versions.py` (includes validation)
- Review version changes in PRs for compatibility impacts

### Build Monitoring
- Monitor build logs for DNS failures
- Set up alerts for build failures
- Maintain fallback DNS servers in build environment

### Documentation Updates
- Update this runbook when new issues are discovered
- Include version compatibility matrices in relevant docs
- Document build environment requirements

## Related Documentation

- `docs/reference/architecture.md` - System architecture overview
- `versions/versions.toml` - Version management configuration
- `docs/policies/POLICIES.md` - Project policies and standards
- `docs/runbooks/build-logging.md` - Build logging procedures

## Emergency Procedures

If builds continue to fail:

1. **Check Network Connectivity:**
   ```bash
   docker run --rm alpine ping -c 3 deb.debian.org
   ```

2. **Use Alternative Base Images:**
   ```dockerfile
   FROM ubuntu:22.04  # Instead of python:3.12-slim
   ```

3. **Implement Offline Builds:**
   - Use wheelhouse for Python dependencies
   - Pre-download apt packages for air-gapped environments

## Version History

- **2026-01-07**: Initial document creation documenting DNS retry logic, FastAPI compatibility fixes, and validation script corrections
