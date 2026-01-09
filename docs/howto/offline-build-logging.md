---
status: active
last_updated: 2026-01-09
category: howto
---

# Offline Build System Logging Guide

**Purpose:** Comprehensive logging strategy for tracking ALL downloads during build and spin up.  
**For:** Ensuring complete offline build capability and download caching.

---

## Overview

The offline build system logs all downloads to ensure:
1. **Complete Caching:** All wheels, apt packages, and other downloads are cached
2. **Offline Verification:** Verify no network access during `--network=none` builds
3. **Download Tracking:** Track what was downloaded, from where, and when
4. **Cache Management:** Identify cached vs downloaded items

---

## Logging Locations

### Build Logs
- **Location:** `logs/build/`
- **Session Logs:** `logs/build/build_session_YYYYMMDD_HHMMSS.log`
- **Download JSON:** `logs/build/downloads_YYYYMMDD_HHMMSS.json`
- **APT Logs:** `logs/build/apt_downloads_YYYYMMDD_HHMMSS.log`
- **Wheel Logs:** `logs/build/wheel_downloads_YYYYMMDD_HHMMSS.log`

### Wheelhouse Logs
- **Location:** `logs/wheelhouse/`
- **Download Logs:** `logs/wheelhouse/download_YYYYMMDD_HHMMSS.log`
- **Manifest:** `wheelhouse/wheelhouse_manifest.json`

### Docker Build Logs
- **Location:** `logs/docker_build/`
- **Service Logs:** `logs/docker_build/build_${service}.log`
- **Build Report:** `logs/docker_build/build_report.md`

---

## What Gets Logged

### 1. Wheel Downloads (pip)
- **What:** All Python package wheels downloaded
- **Details:** Package name, version, URL, destination, size, cached status
- **When:** During `make wheelhouse` and Docker builds
- **Log File:** `wheel_downloads_*.log`, `downloads_*.json`

### 2. APT Package Downloads
- **What:** All Debian packages downloaded via apt-get
- **Details:** Package name, version, URL, size, cached status
- **When:** During Docker builds (apt-get install)
- **Log File:** `apt_downloads_*.log`, `downloads_*.json`

### 3. Docker Build Output
- **What:** Complete Docker build output
- **Details:** All build steps, downloads, errors
- **When:** During `make build`
- **Log File:** `build_${service}.log`

### 4. Cached Items
- **What:** Items retrieved from cache (not downloaded)
- **Details:** Same as downloads, but marked as cached
- **When:** When pip/apt uses cached items
- **Log File:** All logs, marked with `[CACHED]`

---

## Logging Strategy

### 1. Wheelhouse Download Logging

**Script:** `scripts/download_wheelhouse.sh`

**Features:**
- Logs all pip downloads to `logs/wheelhouse/download_*.log`
- Creates `wheelhouse_manifest.json` with download tracking
- Tracks success, errors, and skipped packages
- Logs download URLs and timestamps

**Usage:**
```bash
./scripts/download_wheelhouse.sh
# Logs automatically created in logs/wheelhouse/
```

### 2. Docker Build Logging

**Command:** `make build`

**Features:**
- Logs all Docker build output via BuildKit
- Per-service build logs with comprehensive tracking
- Build report generation with dependency analysis
- Tracks apt and pip downloads during build
- Integration with build tracking system

**Usage:**
```bash
make build
# Logs automatically created in logs/docker_build/
# Build analysis available via make build-analyze
```

### 3. Enhanced Build Logging

**Script:** `scripts/enhanced_build_logging.sh`

**Features:**
- Comprehensive download tracking
- JSON-formatted download logs
- Cache detection and logging
- Session summaries

**Usage:**
```bash
source scripts/enhanced_build_logging.sh
# Functions available: apt_get_with_logging, pip_with_logging, monitor_docker_build
```

---

## Dockerfile Logging

### APT Downloads

All Dockerfiles log apt downloads:

```dockerfile
RUN apt-get update 2>&1 | tee /app/apt-update.log && \
    apt-get install -y --no-install-recommends \
    build-essential cmake ... \
    2>&1 | tee /app/apt-build.log && \
    mkdir -p /app/apt-archives && \
    cp -r /var/cache/apt/archives /app/apt-archives || true
```

**Logs:**
- `/app/apt-update.log` - apt-get update output
- `/app/apt-build.log` - apt-get install output
- `/app/apt-archives/` - Cached .deb files

### PIP Downloads

All Dockerfiles log pip installs:

```dockerfile
ENV PIP_LOG=/app/pip-install.log
RUN pip install ... --log $PIP_LOG ...
```

**Logs:**
- `/app/pip-install.log` - Complete pip install output

---

## Offline Build Verification

### Verify No Network Access

```bash
# Build with --network=none
docker build --network=none -f Dockerfile.api -t xnai_api:latest .

# Check logs for network access attempts
grep -i "download\|fetch\|http" logs/docker_build/build_api.log
# Should show only cached items or errors
```

### Verify All Downloads Cached

```bash
# Check download log
cat logs/build/downloads_*.json | jq '.summary'

# Expected:
# - total_downloads: 0 (if fully cached)
# - total_cached: > 0 (cached items used)
```

---

## Cache Management

### APT Cache

**Location:** `/var/cache/apt/archives/` (in container)

**Preservation:**
- Copied to `/app/apt-archives/` during build
- Can be extracted and reused for offline builds

**Usage:**
```bash
# Extract apt cache from previous build
tar -xzf apt-archives.tgz -C /var/cache/apt/archives/

# Use in Dockerfile
COPY apt-archives /var/cache/apt/archives
```

### PIP Cache

**Location:** `~/.cache/pip/` (host) or `/root/.cache/pip/` (container)

**Preservation:**
- Automatically used by pip if available
- Can be copied into Docker build context

**Usage:**
```bash
# Copy pip cache into build context
cp -r ~/.cache/pip build_context/pip_cache

# Use in Dockerfile
COPY pip_cache /root/.cache/pip
```

### Wheelhouse

**Location:** `wheelhouse/` or `wheelhouse.tgz`

**Preservation:**
- Automatically included in Docker build context
- Extracted during Docker build

**Usage:**
```bash
# Create wheelhouse
make wheelhouse

# Build uses wheelhouse automatically
make build
```

---

## Log Analysis

### Download Summary

```bash
# View download summary
cat logs/build/downloads_*.json | jq '.summary'

# View all downloads
cat logs/build/downloads_*.json | jq '.downloads'

# View cached items
cat logs/build/downloads_*.json | jq '.cached'
```

### Find Uncached Downloads

```bash
# Find items that were downloaded (not cached)
cat logs/build/downloads_*.json | jq '.downloads.wheels[] | select(.cached == false)'

# Find items that should be cached next time
cat logs/build/downloads_*.json | jq '.downloads.wheels[] | select(.cached == false) | .source'
```

### Build Time Analysis

```bash
# View build duration
cat logs/build/downloads_*.json | jq '.duration_seconds'

# Compare cached vs downloaded
cat logs/build/downloads_*.json | jq '.summary'
```

---

## Best Practices

### 1. Always Use Logging

- Enable logging in all build scripts
- Use `tee` to capture output
- Save logs to `logs/` directory

### 2. Track All Downloads

- Log wheels, apt packages, and other downloads
- Mark cached items separately
- Track download sources (URLs)

### 3. Verify Offline Capability

- Test builds with `--network=none`
- Verify all downloads are cached
- Document any network requirements

### 4. Maintain Cache

- Preserve apt cache between builds
- Maintain wheelhouse up-to-date
- Archive caches for offline use

---

## Troubleshooting

### Missing Downloads in Logs

**Issue:** Downloads not appearing in logs

**Solutions:**
1. Check log file permissions
2. Verify logging functions are called
3. Check for errors in log files

### Cache Not Working

**Issue:** Downloads happening even with cache

**Solutions:**
1. Verify cache location is correct
2. Check cache permissions
3. Ensure cache is included in build context

### Offline Build Failing

**Issue:** Build fails with `--network=none`

**Solutions:**
1. Check download logs for missing items
2. Verify all dependencies in wheelhouse
3. Check apt cache is available
4. Review build logs for network access attempts

---

## Example Workflow

### Complete Offline Build

```bash
# 1. Create wheelhouse (with logging)
make wheelhouse
# Logs: logs/wheelhouse/download_*.log

# 2. Build Docker images (with logging)
make build
# Logs: logs/docker_build/build_*.log

# 3. Verify offline capability
docker build --network=none -f Dockerfile.api -t test .
# Should succeed if all dependencies cached

# 4. Review logs
cat logs/build/downloads_*.json | jq '.summary'
```

---

**Last Updated:** 2026-01-09  
**Maintained By:** Build System Team

