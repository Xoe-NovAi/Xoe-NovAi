---
status: active
last_updated: 2026-01-08
category: howto
---

# Makefile Usage Guide

**Complete guide to using Xoe-NovAi's enhanced Makefile with voice-to-voice and build tracking features.**

---

## 🎯 Overview

The Makefile provides comprehensive build, testing, and deployment utilities with new voice-to-voice conversation and dependency tracking capabilities.

### Key Features Added (2026-01-08)
- **🎤 Voice-to-Voice Targets:** Test and deploy voice conversation features
- **📊 Build Tracking:** Comprehensive dependency analysis and duplicate detection
- **⚙️ Wheel Management:** Build, cache, and analyze Python wheels
- **📈 Enhanced Logging:** Multi-level logging for build processes

---

## 📋 Complete Target List

### Core Development Targets

```bash
make help              # Show all available targets with descriptions
make build             # Enhanced build with dependency tracking
make test              # Run tests with coverage
make up                # Start full stack
make down              # Stop full stack
make logs              # Show stack logs
make restart           # Restart all services
make cleanup           # Clean volumes and images (dangerous)
```

### Voice-to-Voice Targets

```bash
make voice-test        # Test voice interface functionality
make voice-build       # Build Docker with voice-to-voice support
make voice-up          # Start voice-enabled UI only
```

### Build Tracking & Analysis Targets

```bash
make build-tracking    # Run complete dependency analysis
make build-analyze     # Analyze current build state
make build-report      # Generate comprehensive build report
make check-duplicates  # Check for duplicate packages
```

### Stack-Cat Documentation Targets

```bash
make stack-cat                # Generate default stack documentation (alias)
make stack-cat-default        # Generate default stack documentation (all components)
make stack-cat-api            # Generate API backend documentation only
make stack-cat-rag            # Generate RAG subsystem documentation only
make stack-cat-frontend       # Generate UI frontend documentation only
make stack-cat-crawler        # Generate CrawlModule subsystem documentation only
make stack-cat-voice          # Generate voice interface documentation only
make stack-cat-all            # Generate documentation for all groups
make stack-cat-separate       # Generate separate markdown files for each source file
make stack-cat-deconcat       # De-concatenate markdown file into separate files (requires FILE=...)
```

### Wheel Management Targets

```bash
make wheelhouse        # Download dependencies to wheelhouse
make wheel-build       # Build wheels for offline caching
make wheel-analyze     # Analyze wheelhouse contents
make deps              # Install from wheelhouse (offline)
```

### Service Debugging Targets

```bash
make debug-rag         # Shell access to RAG API container
make debug-ui          # Shell access to Chainlit UI container
make debug-crawler     # Shell access to Crawler container
make debug-redis       # Shell access to Redis container
```

### Utility Targets

```bash
make validate          # Run configuration validation
make health            # Run health checks
make benchmark         # Run performance benchmarks
make curate            # Run curation example (Gutenberg)
make ingest            # Run library ingestion
```

---

## 🎤 Voice-to-Voice Usage

### Testing Voice Features

```bash
# Test voice interface imports and configuration
make voice-test

# Expected output:
# ✓ Voice interface imports successful
# ✓ Voice config: STT=faster_whisper, TTS=piper_onnx
```

### Building with Voice Support

```bash
# Build Docker image with voice-to-voice capabilities
make voice-build

# Start only the voice-enabled UI (not full stack)
make voice-up

# Access at: http://localhost:8001
# Click "🎤 Start Voice Chat" to begin voice conversation
```

### Voice Features in UI

Once started, the voice-enabled UI provides:

- **🎤 Start Voice Chat** - Begin voice-to-voice conversation
- **🎛️ Voice Settings** - Adjust speed, pitch, volume
- **🔊 Voice Commands** - "Stop voice chat", "Voice settings"
- **📝 Real-time Transcription** - See what you said
- **🔄 Seamless Conversation** - Speak naturally, get voice responses

---

## 📊 Build Tracking & Analysis

### Complete Build Analysis

```bash
# Run comprehensive dependency tracking
make build-tracking

# This executes:
# 1. Parse all requirements files
# 2. Analyze installation logs (if available)
# 3. Generate structured build report
# 4. Save results to build-report.json
```

### Current State Analysis

```bash
# Analyze current environment without full rebuild
make build-analyze

# Shows:
# - Dependencies found in requirements
# - Installation analysis (if logs exist)
# - Duplicate package detection
```

### Build Reports

```bash
# Generate detailed build report
make build-report

# Creates build-report.json with:
# - Total dependencies parsed
# - Packages installed
# - Downloads detected (should be 0 for wheelhouse)
# - Install method used
# - Performance metrics
```

### Duplicate Detection

```bash
# Check for packages installed multiple times
make check-duplicates

# Shows any packages with multiple versions
# Useful for troubleshooting conflicts
```

---

## 📚 Stack-Cat Documentation Generation

### Overview

Stack-Cat is a comprehensive documentation generator that creates structured documentation of your Xoe-NovAi codebase. It supports multiple output formats and component-specific documentation.

### Basic Usage

```bash
# Generate default stack documentation (all components)
make stack-cat

# Generate documentation for specific components
make stack-cat-api       # API backend only
make stack-cat-rag       # RAG subsystem only
make stack-cat-voice     # Voice interface only
make stack-cat-frontend  # UI frontend only
make stack-cat-crawler   # CrawlModule only

# Generate all component documentation
make stack-cat-all

# Generate separate markdown files for each source file
make stack-cat-separate
```

### Advanced Usage

```bash
# De-concatenate existing documentation (requires FILE variable)
make stack-cat-deconcat FILE=scripts/stack-cat/stack-cat-output/20251021_143022/stack-cat_20251021_143022.md

# View generated documentation
ls scripts/stack-cat/stack-cat-output/
# stack-cat_latest.md     - Symlink to latest markdown
# stack-cat_latest.html   - Symlink to latest HTML
# 20251021_143022/        - Timestamped snapshot directory
```

### Output Formats

Stack-Cat generates three types of documentation:

#### Markdown (.md)
- **Perfect for AI assistants** (Claude, ChatGPT, etc.)
- **Readable in any text editor**
- **Single comprehensive file**
- **Syntax-highlighted code blocks**

#### HTML (.html)
- **Interactive web interface**
- **Collapsible file sections**
- **Professional appearance**
- **Table of contents navigation**

#### JSON Manifest (.json)
- **Structured metadata**
- **Programmatic access**
- **File statistics and checksums**
- **API-friendly format**

### Documentation Groups

- **default**: Complete core stack (all components)
- **api**: Backend API services only
- **rag**: RAG (Retrieval-Augmented Generation) subsystem
- **frontend**: User interface components
- **crawler**: CrawlModule subsystem
- **voice**: Voice interface and TTS/STT components (v0.1.5)

### Stack Validation

Stack-Cat validates your codebase against Xoe-NovAi v0.1.5 standards:

```bash
# Validation checks:
# ✓ All mandatory design patterns (Pattern 1-5)
# ✓ Voice interface components present
# ✓ Circuit Breaker (Pattern 5) implementation
# ✓ Config version matches v0.1.5
# ✓ Required files and dependencies
```

### Integration with Development Workflow

```bash
# Before major changes - document current state
make stack-cat

# After implementing voice features
make stack-cat-voice

# For debugging specific components
make stack-cat-api

# Generate comprehensive documentation for release
make stack-cat-all
```

### Output Location

Generated documentation is saved to:
```
scripts/stack-cat/stack-cat-output/
├── stack-cat_latest.md              # Latest markdown (symlink)
├── stack-cat_latest.html            # Latest HTML (symlink)
├── stack-manifest_latest.json       # Latest JSON (symlink)
├── separate-md_latest/              # Latest separate files (symlink)
└── 20251021_143022/                 # Timestamped snapshot
    ├── stack-cat_20251021_143022.md
    ├── stack-cat_20251021_143022.html
    ├── stack-manifest_20251021_143022.json
    └── separate-md/
```

### ⚠️ Data Safety & Cleanup

**Important:** `make stack-cat-clean` includes safety measures to prevent accidental data loss:

```bash
make stack-cat-clean

# Shows warning:
# ⚠️  WARNING: This will permanently delete ALL historical Stack-Cat documentation snapshots!
# Output directory: scripts/stack-cat/stack-cat-output/
# Are you sure you want to permanently delete all Stack-Cat output? (type 'yes' to confirm):
```

**What gets cleaned:**
- ✅ Active documentation output (`stack-cat-output/`)
- ✅ Generated markdown, HTML, and JSON files
- ✅ Timestamped snapshot directories
- ✅ Symlinks to latest files

**What is preserved:**
- ✅ Archived historical data (`stack-cat-archive/` - older snapshots)
- ✅ Configuration files (`groups.json`, `whitelist.json`)
- ✅ The script itself (`stack-cat.sh`)
- ✅ User guide and documentation

### Automated Archiving

Stack-Cat includes automated maintenance to keep your workspace organized:

```bash
make stack-cat-archive

# Automatically moves files older than 7 days from:
# scripts/stack-cat/stack-cat-output/ → scripts/stack-cat/stack-cat-archive/

# Example output:
# Archiving: stack-cat-output/20251021_143022/stack-cat_20251021_143022.md
# ✓ Archived 5 files older than 1 week
# Archive location: scripts/stack-cat/stack-cat-archive/
```

**Archiving Process:**
- ✅ Finds files older than 7 days using `find -mtime +7`
- ✅ Preserves directory structure in archive
- ✅ Removes empty directories from active output
- ✅ No confirmation required (safe, reversible operation)

**Recommended Usage:**
```bash
# Weekly maintenance - archive old docs and clean up
make stack-cat-archive

# Generate new documentation
make stack-cat

# Emergency cleanup (with confirmation)
make stack-cat-clean
```

---

## ⚙️ Wheel Management

### Building Wheelhouse

```bash
# Download all dependencies as wheels
make wheelhouse

# Creates wheelhouse/ directory with:
# - All Python packages as .whl files
# - Offline installation capability
# - Faster subsequent builds
```

### Advanced Wheel Building

```bash
# Build wheels for all requirement files
make wheel-build

# Builds wheels for:
# - requirements-api.txt
# - requirements-chainlit.txt
# - requirements-crawl.txt
# - requirements-curation_worker.txt

# Creates compressed wheelhouse.tgz for distribution
```

### Wheel Analysis

```bash
# Analyze wheelhouse contents
make wheel-analyze

# Shows:
# - Total number of wheels
# - Total compressed size
# - Sample wheel filenames
```

### Offline Installation

```bash
# Install from pre-built wheelhouse (no internet required)
make deps

# Uses: pip install --no-index --find-links=wheelhouse
# Much faster than PyPI downloads
```

---

## 🔄 Build Process with Tracking

### Enhanced Build Workflow

```bash
# 1. Prepare wheelhouse (offline cache)
make wheelhouse

# 2. Build with comprehensive tracking
make build

# 3. Analyze build results
make build-analyze

# 4. Generate build report
make build-report
```

### Build Artifacts Created

The enhanced build process creates:

```
build-logs/
├── pip-install.log          # Detailed pip operations
├── dependency-tracking.log  # Build phase logging
├── wheel-build.log          # Wheel building operations
├── dependency-manifest.json # Structured dependency data
└── build-report.json        # Executive summary
```

### Accessing Build Logs

```bash
# From running container
docker run --rm xoe-novai-chainlit cat /app/build-logs/build-report.json

# Or inspect build artifacts in wheelhouse
make wheel-analyze
```

---

## 🚀 Development Workflow

### Local Development Setup

```bash
# 1. Create wheelhouse for fast installs
make wheelhouse

# 2. Install dependencies offline
make deps

# 3. Test voice functionality
make voice-test

# 4. Run tests
make test

# 5. Start development stack
make up

# 6. View logs
make logs
```

### Production Deployment

```bash
# 1. Build with tracking
make build

# 2. Analyze build quality
make build-analyze

# 3. Start production stack
make up

# 4. Run health checks
make health
```

### Troubleshooting Builds

```bash
# Check for build issues
make build-analyze

# Look for duplicate packages
make check-duplicates

# Analyze wheelhouse
make wheel-analyze

# Debug specific services
make debug-ui      # Chainlit UI shell
make debug-rag     # RAG API shell
make debug-redis   # Redis shell
```

---

## 📈 Performance Monitoring

### Build Performance Tracking

The build system tracks:

- **Build Duration:** Total time for each phase
- **Download Activity:** Number of network requests
- **Cache Hit Rate:** Percentage of cached vs downloaded packages
- **Wheel Efficiency:** Size and compression ratios
- **Duplicate Detection:** Packages installed multiple times

### Example Build Report

```json
{
  "build_summary": {
    "total_dependencies": 25,
    "wheels_built": 0,
    "packages_installed": 85,
    "downloads_detected": 0,
    "install_method": "wheelhouse"
  },
  "performance_metrics": {
    "build_duration_seconds": 45,
    "cache_hit_rate": "100%",
    "optimization_score": 95
  }
}
```

---

## 🔧 Advanced Usage

### Custom Build Arguments

```bash
# Build with specific wheelhouse settings
docker build --build-arg USE_WHEELHOUSE=true --build-arg BUILD_WHEELS=false -t xoe-novai .

# Override environment variables
DOCKER_BUILDKIT=1 make build
```

### Selective Service Management

```bash
# Start only specific services
docker compose up chainlit redis

# Debug specific containers
make debug-ui     # Chainlit shell
make debug-rag    # FastAPI shell
make debug-redis  # Redis shell
```

### Build Optimization Tips

1. **Always use wheelhouse** for faster builds:
   ```bash
   make wheelhouse && make deps
   ```

2. **Analyze before deploying:**
   ```bash
   make build && make build-analyze
   ```

3. **Monitor build performance:**
   ```bash
   make build-report
   ```

---

## 🎯 Target Categories

### 🔧 Development
- `make test` - Run test suite
- `make validate` - Configuration validation
- `make health` - Service health checks
- `make benchmark` - Performance testing

### 🐳 Docker
- `make build` - Enhanced build with tracking
- `make up/down` - Stack lifecycle
- `make logs` - Service logging
- `make debug-*` - Container access

### 🎤 Voice
- `make voice-test` - Interface validation
- `make voice-build` - Voice-enabled build
- `make voice-up` - Voice UI only

### 📦 Dependencies
- `make wheelhouse` - Download wheels
- `make wheel-build` - Build offline cache
- `make wheel-analyze` - Wheel inspection
- `make deps` - Offline installation

### � Documentation
- `make stack-cat` - Generate default documentation
- `make stack-cat-api` - API backend docs only
- `make stack-cat-rag` - RAG subsystem docs only
- `make stack-cat-voice` - Voice interface docs only
- `make stack-cat-all` - All component docs
- `make stack-cat-separate` - Separate files per source
- `make stack-cat-clean` - Clean documentation output

### �📊 Analysis
- `make build-tracking` - Complete analysis
- `make build-analyze` - Current state
- `make build-report` - Structured reports
- `make check-duplicates` - Conflict detection

---

## ❓ Troubleshooting

### Common Issues

#### Voice Tests Failing
```bash
# Check if voice dependencies are installed
make deps

# Test voice interface specifically
python3 -c "from app.XNAi_rag_app.voice_interface import VoiceInterface; print('Voice OK')"
```

#### Build Analysis Shows Errors
```bash
# Check build logs
docker run --rm xoe-novai-chainlit cat /app/build-logs/dependency-tracking.log

# Re-run analysis
make build-analyze
```

#### Wheelhouse Issues
```bash
# Rebuild wheelhouse
rm -rf wheelhouse && make wheelhouse

# Check wheelhouse contents
make wheel-analyze
```

#### Container Access Issues
```bash
# Ensure containers are running
make up

# Check container status
docker ps

# Access with proper user
make debug-ui  # Uses appuser
```

---

## 📚 Related Documentation

- **Voice Setup:** [`howto/voice-setup.md`](voice-setup.md)
- **Docker Deployment:** [`howto/docker-setup.md`](docker-setup.md)
- **Build System:** [`runbooks/build-tools.md`](runbooks/build-tools.md)
- **Dependency Tracking:** [`enhancements/enhancement-dependency-tracking.md`](../enhancements/enhancement-dependency-tracking.md)

---

## 🎉 Quick Start Summary

```bash
# Complete development setup
make wheelhouse  # Create offline cache
make deps        # Install dependencies
make voice-test  # Verify voice features
make test        # Run tests
make up          # Start stack
make logs        # Monitor services

# Voice conversation at: http://localhost:8001
```

The enhanced Makefile provides comprehensive build, testing, and deployment capabilities with full voice-to-voice support and detailed dependency tracking.

---

**Last Updated:** 2026-01-08
**Makefile Version:** v2.1 (Voice + Build Tracking + Stack-Cat)
**Targets Added:** 20 new targets (9 voice/analysis + 11 stack-cat documentation)
