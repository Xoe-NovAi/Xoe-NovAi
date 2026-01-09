---
status: active
last_updated: 2026-01-08
category: howto
tags:
  - deployment
  - offline
  - air-gapped
  - enterprise
---

# Offline Deployment Guide

**Xoe-NovAi v0.1.5** supports fully offline, air-gapped deployment with BuildKit caching and wheelhouse distribution.

---

## Quick Start

```bash
# Build with offline caching (first time - downloads required)
make build

# Subsequent builds - completely offline
make build
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Offline Build System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    BuildKit Cache    ┌─────────────────┐  │
│  │ PyPI        │ ──────────────────▶   │ /var/cache/apt  │  │
│  │ Wheelhouse  │                       │ /root/.cache/pip│  │
│  └─────────────┘                       └─────────────────┘  │
│                                             │               │
│                                             ▼               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Persistent Cache Between Builds            │    │
│  │   - apt packages (locked sharing)                    │    │
│  │   - pip wheels (shared access)                       │    │
│  │   - No re-downloads on rebuild                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Docker BuildKit Cache Configuration

### Cache Types

| Cache Type | Location | Sharing | Purpose |
|------------|----------|---------|---------|
| `type=cache` | `/var/cache/apt` | locked | APT packages |
| `type=cache` | `/root/.cache/pip` | shared | pip downloads |
| `type=cache` | `/build/wheelhouse` | shared | Pre-built wheels |

### Dockerfile Example

```dockerfile
# Build with persistent caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libgomp1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir -r requirements.txt
```

---

## Wheelhouse Distribution

### Create Wheelhouse

```bash
# Build all wheels for offline distribution
make wheel-build

# Output: wheelhouse/ directory with all .whl files
#        wheelhouse.tgz for easy transfer
```

### Install from Wheelhouse

```bash
# Extract wheelhouse (if using archive)
tar -xzf wheelhouse.tgz

# Install without network
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

---

## Model Caching

### Download Models Once

```bash
# Download LLM model
mkdir -p models
wget -P models https://huggingface.co/.../gemma-3-4b-it-UD-Q5_K_XL.gguf

# Download embedding model  
mkdir -p embeddings
wget -P embeddings https://huggingface.co/.../all-MiniLM-L12-v2.Q8_0.gguf
```

### Configure Model Paths

```toml
# config.toml
[models]
llm_path = "/models/local/all/gemma-3-4b-it-UD-Q5_K_XL.gguf"
embedding_path = "/embeddings/all-MiniLM-L12-v2.Q8_0.gguf"
```

---

## Air-Gapped Deployment Checklist

- [ ] Build wheelhouse on internet-connected machine
- [ ] Transfer wheelhouse.tgz to air-gapped environment
- [ ] Download models to local storage
- [ ] Configure `config.toml` with local paths
- [ ] Build Docker images with `DOCKER_BUILDKIT=1`
- [ ] Verify no external network calls

---

## Verify Offline Mode

```bash
# Check telemetry disables
python3 app/XNAi_rag_app/healthcheck.py -- telemetry

# Expected output:
# ✓ telemetry: Telemetry: 8/8 disables verified
```

---

## Performance Comparison

| Build Type | First Build | Subsequent Builds |
|------------|-------------|-------------------|
| Without caching | 5+ minutes | 5+ minutes |
| With BuildKit cache | 5+ minutes | **<30 seconds** |
| With wheelhouse | 5+ minutes | **<30 seconds** |

---

## Troubleshooting

### Cache Not Persisting

```bash
# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1

# Check cache mounts
docker build --progress=plain .
```

### Wheelhouse Not Found

```bash
# Verify wheelhouse directory
ls -la wheelhouse/

# Should contain .whl files
```

### Model Not Loading

```bash
# Verify model paths
cat config.toml | grep models

# Check file exists
ls -la /models/
```
