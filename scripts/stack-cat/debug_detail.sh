#!/usr/bin/env bash
# Detailed whitelist debugging
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/arcana-novai/Documents/XNAi-v0_1_3-clean"
WHITELIST_JSON="$SCRIPT_DIR/whitelist.json"

echo "ROOT_DIR: $ROOT_DIR"
echo "WHITELIST_JSON: $WHITELIST_JSON"
echo ""

# Load arrays exactly as the script does
declare -a ALLOWED_ROOTS
declare -a ALLOWED_DIRS
declare -a EXCLUDED_EXTS

echo "Loading whitelist..."
mapfile -t ALLOWED_ROOTS < <(jq -r '.allowed_roots[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
mapfile -t ALLOWED_DIRS < <(jq -r '.allowed_dirs[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
mapfile -t EXCLUDED_EXTS < <(jq -r '.excluded_extensions[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")

echo "ALLOWED_ROOTS (${#ALLOWED_ROOTS[@]}):"
printf '  [%s]\n' "${ALLOWED_ROOTS[@]}"
echo ""

echo "ALLOWED_DIRS (${#ALLOWED_DIRS[@]}):"
printf '  [%s]\n' "${ALLOWED_DIRS[@]}"
echo ""

echo "EXCLUDED_EXTS (${#EXCLUDED_EXTS[@]}):"
printf '  [%s]\n' "${EXCLUDED_EXTS[@]}"
echo ""

# Test the exact matching logic with real files
echo "=== Testing files ==="

test_file() {
    local file="$1"
    local rel_path="${file#$ROOT_DIR/}"
    
    echo ""
    echo "File: $file"
    echo "Relative path: [$rel_path]"
    
    # Check excluded extensions
    local allowed=0
    for ext in "${EXCLUDED_EXTS[@]}"; do
        if [[ "$file" == *"$ext" ]]; then
            echo "  ✗ BLOCKED: excluded extension [$ext]"
            return 1
        fi
    done
    
    # Check allowed roots (exact match)
    for root in "${ALLOWED_ROOTS[@]}"; do
        if [[ "$rel_path" == "$root" ]]; then
            echo "  ✓ ALLOWED: matches allowed_root [$root]"
            return 0
        fi
    done
    
    # Check allowed dirs (prefix match)
    for dir in "${ALLOWED_DIRS[@]}"; do
        dir_clean="${dir%/}"
        echo "    Checking dir [$dir_clean] against [$rel_path]"
        if [[ "$rel_path" == "$dir_clean"/* ]] || [[ "$rel_path" == "$dir_clean" ]]; then
            echo "  ✓ ALLOWED: matches allowed_dir [$dir_clean]"
            return 0
        fi
    done
    
    echo "  ✗ BLOCKED: not in whitelist"
    return 1
}

# Test actual files
test_file "$ROOT_DIR/docker-compose.yml"
test_file "$ROOT_DIR/Dockerfile.api"
test_file "$ROOT_DIR/app/XNAi_rag_app/__init__.py"
test_file "$ROOT_DIR/app/XNAi_rag_app/main.py"