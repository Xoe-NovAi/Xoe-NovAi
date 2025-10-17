#!/bin/bash
# ============================================================================
# Xoe-NovAi Phase 1 v0.1.2 rev_1.4 - Complete Stack Structure Creator
# ============================================================================
# Purpose: Create complete directory structure for rev_1.4 stack
# Usage: bash create-stack-structure.sh [target_directory]
# Last Updated: 2025-10-13
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Target directory
TARGET_DIR="${1:-./xnai-stack}"

echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}Xoe-NovAi Phase 1 v0.1.2 rev_1.4 - Stack Structure Creator${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo "Creating structure in: $TARGET_DIR"
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo -e "${GREEN}[1/5]${NC} Creating directory structure..."

# Root level
mkdir -p models
mkdir -p embeddings
mkdir -p data/redis
mkdir -p data/faiss_index
mkdir -p data/faiss_index.bak
mkdir -p data/prometheus-multiproc
mkdir -p backups

# Application
mkdir -p app/XNAi_rag_app/logs

# Scripts
mkdir -p scripts

# Tests
mkdir -p tests

# Library (curated content - NEW rev_1.4)
mkdir -p library/psychology
mkdir -p library/physics
mkdir -p library/classical-works
mkdir -p library/esoteric
mkdir -p library/technical-manuals

# Knowledge (Phase 2 agents - NEW rev_1.4)
mkdir -p knowledge/curator
mkdir -p knowledge/coding-expert
mkdir -p knowledge/linguist

echo -e "${GREEN}✓${NC} Directory structure created"
echo ""

echo -e "${GREEN}[2/5]${NC} Creating placeholder files..."

# .gitkeep for empty directories
touch app/XNAi_rag_app/logs/.gitkeep
touch data/redis/.gitkeep
touch data/faiss_index/.gitkeep
touch backups/.gitkeep
touch library/psychology/.gitkeep
touch library/physics/.gitkeep
touch library/classical-works/.gitkeep
touch library/esoteric/.gitkeep
touch library/technical-manuals/.gitkeep
touch knowledge/curator/.gitkeep
touch knowledge/coding-expert/.gitkeep
touch knowledge/linguist/.gitkeep

# __init__.py for tests package
touch tests/__init__.py

echo -e "${GREEN}✓${NC} Placeholder files created"
echo ""

echo -e "${GREEN}[3/5]${NC} Creating .gitignore..."

cat << EOF > .gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
venv/
*.egg-info/
build/
dist/
*.egg

# Docker
docker-compose.override.yml
*.log

# Data (runtime generated)
data/*
!data/.gitkeep
library/*
!library/.gitkeep
knowledge/*
!knowledge/.gitkeep
backups/*
!backups/.gitkeep
models/*
!models/.gitkeep
embeddings/*
!embeddings/.gitkeep

# Environment
.env
.env.local
EOF

echo -e "${GREEN}✓${NC} .gitignore created"
echo ""

echo -e "${GREEN}[4/5]${NC} Setting permissions (non-root UID/GID 1001)..."

chown -R 1001:1001 app data backups library knowledge
chmod -R 755 app data backups library knowledge
chmod 600 .env.template

echo -e "${GREEN}✓${NC} Permissions set"
echo ""

echo -e "${GREEN}[5/5]${NC} Validation..."

# Check total directories
total_dirs=$(find . -type d | wc -l)
if [ $total_dirs -lt 12 ]; then
    echo -e "${RED}✗${NC} Directory count mismatch (expected >=12, got $total_dirs)"
    exit 1
fi

echo -e "${GREEN}✓${NC} Validation passed"

echo ""
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}Structure Creation Complete!${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo ""
echo "Created in: $(pwd)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Copy code files into this structure"
echo "2. Download models to models/ and embeddings/"
echo "3. Customize .env from .env.template"
echo "4. Run docker compose up -d"
echo "5. Validate with docker compose ps"