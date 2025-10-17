#!/bin/bash
# ============================================================================
# Xoe-NovAi Setup Script - Initialize New Stack (v0.1.1 PRODUCTION-READY)
# ============================================================================
# Purpose: Create a new Xoe-NovAi stack directory with all core files
# Usage: ./setup-new-stack.sh <target_dir>
# Guide Reference: Section 2 (Stack Initialization)
# Last Updated: 2025-10-11
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# ARGUMENT VALIDATION
# ============================================================================

if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Target directory required${NC}"
    echo ""
    echo "Usage: $0 <target_dir>"
    echo ""
    echo "Example:"
    echo "  $0 /opt/xnai-stack"
    echo "  $0 ~/projects/xnai"
    exit 1
fi

TARGET_DIR="$1"

# Resolve to absolute path
TARGET_DIR=$(cd "$(dirname "$TARGET_DIR")" 2>/dev/null && pwd)/$(basename "$TARGET_DIR") || TARGET_DIR="$1"

echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}Xoe-NovAi Phase 1 v0.1.0 - Stack Setup${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo "Target directory: $TARGET_DIR"
echo ""

# ============================================================================
# ARTIFACTS DIRECTORY DETECTION
# ============================================================================

# Try multiple possible artifact locations
ARTIFACTS_LOCATIONS=(
    "$HOME/XNAi-0.1.0/artifacts"
    "./artifacts"
    "../artifacts"
    "$(dirname "$0")/artifacts"
    "$(pwd)/artifacts"
)

ARTIFACTS_DIR=""

for location in "${ARTIFACTS_LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        ARTIFACTS_DIR="$location"
        echo -e "${GREEN}✓${NC} Found artifacts directory: $ARTIFACTS_DIR"
        break
    fi
done

if [ -z "$ARTIFACTS_DIR" ]; then
    echo -e "${RED}✗ Artifacts directory not found${NC}"
    echo ""
    echo "Searched locations:"
    for location in "${ARTIFACTS_LOCATIONS[@]}"; do
        echo "  - $location"
    done
    echo ""
    echo "Please ensure artifacts directory exists or specify path manually."
    exit 1
fi

# ============================================================================
# FILE LIST
# ============================================================================

# Core configuration files (Batch 1)
FILES_TO_COPY=(
    ".env"
    "config.toml"
    "docker-compose.yml"
    "validate-config.sh"
)

# Optional files (if present)
OPTIONAL_FILES=(
    ".dockerignore"
    ".gitignore"
    "Dockerfile.api"
    "Dockerfile.chainlit"
    "entrypoint-api.sh"
    "entrypoint-chainlit.sh"
    "entrypoint-redis.sh"
)

echo ""
echo "Files to copy: ${#FILES_TO_COPY[@]} core + ${#OPTIONAL_FILES[@]} optional"
echo ""

# ============================================================================
# TARGET DIRECTORY CREATION
# ============================================================================

if [ -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}⚠${NC} Target directory already exists: $TARGET_DIR"
    echo -n "Continue and overwrite files? (y/N): "
    read -r response
    
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
else
    echo "Creating target directory..."
    mkdir -p "$TARGET_DIR" || {
        echo -e "${RED}✗ Failed to create $TARGET_DIR${NC}"
        exit 1
    }
    echo -e "${GREEN}✓${NC} Created $TARGET_DIR"
fi

# Change to target directory
cd "$TARGET_DIR" || {
    echo -e "${RED}✗ Cannot change to $TARGET_DIR${NC}"
    exit 1
}

echo ""
echo "Working directory: $(pwd)"
echo ""

# ============================================================================
# COPY CORE FILES
# ============================================================================

echo "Copying core configuration files..."
echo ""

FILES_COPIED=0
FILES_FAILED=0

for file in "${FILES_TO_COPY[@]}"; do
    source_file="$ARTIFACTS_DIR/$file"
    
    if [ -f "$source_file" ]; then
        cp "$source_file" . || {
            echo -e "${RED}✗${NC} Failed to copy $file"
            ((FILES_FAILED++))
            continue
        }
        echo -e "${GREEN}✓${NC} Copied $file"
        ((FILES_COPIED++))
    else
        echo -e "${RED}✗${NC} $file not found in $ARTIFACTS_DIR"
        ((FILES_FAILED++))
    fi
done

echo ""

# ============================================================================
# COPY OPTIONAL FILES
# ============================================================================

echo "Copying optional files (if present)..."
echo ""

for file in "${OPTIONAL_FILES[@]}"; do
    source_file="$ARTIFACTS_DIR/$file"
    
    if [ -f "$source_file" ]; then
        cp "$source_file" . || {
            echo -e "${YELLOW}⚠${NC} Failed to copy optional file $file"
            continue
        }
        echo -e "${GREEN}✓${NC} Copied $file"
        ((FILES_COPIED++))
    else
        echo -e "${YELLOW}⚠${NC} Optional file $file not found (skipping)"
    fi
done

echo ""

# ============================================================================
# SET FILE PERMISSIONS
# ============================================================================

echo "Setting file permissions..."
echo ""

# Secure .env
if [ -f ".env" ]; then
    chmod 600 .env
    echo -e "${GREEN}✓${NC} Set .env permissions to 600 (read/write owner only)"
fi

# Make scripts executable
for script in validate-config.sh entrypoint-*.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo -e "${GREEN}✓${NC} Made $script executable"
    fi
done

echo ""

# ============================================================================
# CREATE .env.template
# ============================================================================

if [ -f ".env" ]; then
    echo "Creating .env.template..."
    
    # Create template with placeholder password
    sed 's/REDIS_PASSWORD=X7n9mQ4wP2kL8vRt/REDIS_PASSWORD=CHANGE_ME_16_CHARS/' .env > .env.template || {
        echo -e "${YELLOW}⚠${NC} Failed to create .env.template"
    }
    
    if [ -f ".env.template" ]; then
        echo -e "${GREEN}✓${NC} Created .env.template with placeholders"
    fi
    
    echo ""
fi

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

echo "Creating directory structure..."
echo ""

DIRECTORIES=(
    "models"
    "embeddings"
    "data/redis"
    "data/faiss_index"
    "data/faiss_index.bak"
    "data/prometheus-multiproc"
    "app/XNAi_rag_app"
    "app/XNAi_rag_app/logs"
    "app/XNAi_rag_app/library"
    "backups"
)

for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir" || {
        echo -e "${YELLOW}⚠${NC} Failed to create directory $dir"
        continue
    }
    echo -e "${GREEN}✓${NC} Created directory $dir"
done

echo ""

# ============================================================================
# CREATE README
# ============================================================================

echo "Creating README.md..."
echo ""

cat > README.md << 'EOF'
# Xoe-NovAi Phase 1 v0.1.1 - Stack Directory

This directory contains a complete Xoe-NovAi stack instance.

## Quick Start

1. **Download Models**
   ```bash
   # Place models in respective directories:
   # - ./models/gemma-3-4b-it-UD-Q5_K_XL.gguf
   # - ./embeddings/all-MiniLM-L12-v2.Q8_0.gguf
   ```

2. **Configure Environment**
   ```bash
   # Edit .env and customize:
   nano .env
   
   # Critical settings:
   # - REDIS_PASSWORD (change from default)
   # - APP_UID/APP_GID (match your user)
   ```

3. **Validate Configuration**
   ```bash
   bash validate-config.sh
   ```

4. **Deploy Stack**
   ```bash
   docker compose up -d
   ```

5. **Monitor Health**
   ```bash
   docker compose ps
   docker compose logs -f
   ```

## Directory Structure

```
.
├── .env                    # Environment variables (SECURE THIS)
├── config.toml             # Application configuration
├── docker-compose.yml      # Service orchestration
├── validate-config.sh      # Configuration validation
├── models/                 # LLM models (2.8GB)
├── embeddings/             # Embedding models (45MB)
├── data/                   # Persistent data
│   ├── redis/              # Redis data
│   ├── faiss_index/        # Primary FAISS index
│   ├── faiss_index.bak/    # FAISS backup
│   └── prometheus-multiproc/ # Metrics data
├── app/XNAi_rag_app/       # Application code
│   ├── logs/               # Application logs
│   └── library/            # Document library for RAG
└── backups/                # Backup storage

```

## Endpoints

- **Chainlit UI**: http://localhost:8001
- **RAG API**: http://localhost:8000
- **Prometheus Metrics**: http://localhost:8002/metrics

## Performance Targets

- Token Rate: 15-25 tok/s
- Memory: <6.0GB
- API Latency: <1000ms (p95)
- Startup Time: <90s

## Troubleshooting

```bash
# View logs
docker compose logs rag

# Check health
docker exec xnai_rag_api python3 /app/XNAi_rag_app/healthcheck.py

# Restart services
docker compose restart

# Full rebuild
docker compose down
docker compose up -d --build
```

## Documentation

See the complete Phase 1 guide for detailed information:
- Architecture overview
- Configuration reference
- Performance tuning
- Security hardening
- Troubleshooting guide

## Support

Project: Xoe-NovAi Phase 1 v0.1.1
Guide Version: v0.1.1
Last Updated: 2025-10-11
EOF

echo -e "${GREEN}✓${NC} Created README.md"
echo ""

# ============================================================================
# VALIDATION
# ============================================================================

echo "Validating setup..."
echo ""

# Check critical files exist
MISSING_FILES=0

for file in "${FILES_TO_COPY[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗${NC} Critical file missing: $file"
        ((MISSING_FILES++))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ Setup incomplete: $MISSING_FILES critical files missing${NC}"
    exit 1
fi

# Run validation if script exists
if [ -f "validate-config.sh" ]; then
    echo "Running configuration validation..."
    echo ""
    
    if bash validate-config.sh; then
        echo ""
        echo -e "${GREEN}✓ Configuration validation passed${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠ Configuration validation failed (expected - customize .env first)${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} validate-config.sh not found, skipping validation"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo -e "${GREEN}✓${NC} Stack directory created: $TARGET_DIR"
echo -e "${GREEN}✓${NC} Files copied: $FILES_COPIED"
echo -e "${GREEN}✓${NC} Directory structure created"
echo -e "${GREEN}✓${NC} README.md generated"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Download Models"
echo "   Place the following files:"
echo "   - ./models/gemma-3-4b-it-UD-Q5_K_XL.gguf (2.8GB)"
echo "   - ./embeddings/all-MiniLM-L12-v2.Q8_0.gguf (45MB)"
echo ""
echo "2. Customize Configuration"
echo "   ${BLUE}nano .env${NC}"
echo "   Critical settings:"
echo "   - REDIS_PASSWORD (change from default!)"
echo "   - APP_UID/APP_GID (match your user: $(id -u):$(id -g))"
echo ""
echo "3. Validate Configuration"
echo "   ${BLUE}bash validate-config.sh${NC}"
echo ""
echo "4. Deploy Stack"
echo "   ${BLUE}docker-compose up -d${NC}"
echo ""
echo "5. Monitor Health"
echo "   ${BLUE}docker-compose ps${NC}"
echo "   ${BLUE}docker-compose logs -f${NC}"
echo ""
echo -e "${YELLOW}Security Reminders:${NC}"
echo "  - Change REDIS_PASSWORD before production deployment"
echo "  - Secure .env file (permissions: 600)"
echo "  - Review all configuration settings in config.toml"
echo ""
echo -e "${GREEN}Stack initialized successfully!${NC}"
echo ""