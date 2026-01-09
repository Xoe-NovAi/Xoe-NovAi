---
status: active
last_updated: 2026-01-09
category: runbook
---

# Build System Logging Runbook

**Purpose:** Operational guide for build system logging and offline build verification.  
**For:** Developers and operators managing the build process.

---

## Quick Start

### Standard Build (with logging)
```bash
# Build with comprehensive logging
make build

# Logs created in:
# - logs/wheelhouse/download_*.log
# - logs/docker_build/build_*.log
# - logs/build/downloads_*.json
```

### Offline Build Verification
```bash
# Verify offline capability
./scripts/verify_offline_build.sh

# Test offline build
docker build --network=none -f Dockerfile.api -t test .
```

---

## Log Locations

### Build Logs
- **Wheelhouse:** `logs/wheelhouse/download_YYYYMMDD_HHMMSS.log`
- **Docker Build:** `logs/docker_build/build_YYYYMMDD_HHMMSS.log`
- **Downloads:** `logs/build/downloads_YYYYMMDD_HHMMSS.json`
- **Session:** `logs/build/build_session_YYYYMMDD_HHMMSS.log`

### Docker Build Logs
- **Per Service:** `logs/docker_build/build_${service}.log`
- **Build Report:** `logs/docker_build/build_report.md`

---

## Logging Features

### Automatic Logging
- ✅ All pip downloads logged
- ✅ All apt downloads logged
- ✅ Cache hits logged
- ✅ Download sources tracked
- ✅ File sizes recorded
- ✅ Timestamps included

### Manual Logging
- Use `make build-tracking` for detailed build analysis
- Use `make build-analyze` for current state analysis
- Use `make build-report` for comprehensive build reports

---

## Verification Steps

### 1. Pre-Build Check
```bash
# Check wheelhouse exists
ls -la wheelhouse* 

# Check logs directory
ls -la logs/
```

### 2. Build with Logging
```bash
# Standard build (logs automatically)
make build

# Or manual build
./scripts/build_docker.sh
```

### 3. Verify Downloads
```bash
# View download summary
cat logs/build/downloads_*.json | jq '.summary'

# Check for uncached downloads
cat logs/build/downloads_*.json | jq '.downloads.wheels[] | select(.cached == false)'
```

### 4. Test Offline
```bash
# Run verification script
./scripts/verify_offline_build.sh

# Or manual test
docker build --network=none -f Dockerfile.api -t test .
```

---

## Troubleshooting

### Build Fails with Network Error
1. Check wheelhouse exists and has wheels
2. Verify all critical packages present
3. Check download logs for missing packages
4. Run `make wheelhouse` to update

### Downloads Not Logged
1. Check log file permissions
2. Verify logging scripts are executable
3. Check logs directory exists
4. Review build output for errors

### Offline Build Fails
1. Check wheelhouse completeness
2. Verify apt cache (if used)
3. Review build logs for network access
4. Ensure `--network=none` flag used

---

**Last Updated:** 2026-01-09

