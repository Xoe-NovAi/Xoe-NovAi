#!/usr/bin/env bash
# Quick debug script to test the stack-cat setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(realpath "$SCRIPT_DIR/../..")}"
GROUPS_JSON="${GROUPS_JSON:-$SCRIPT_DIR/groups.json}"
WHITELIST_JSON="${WHITELIST_JSON:-$SCRIPT_DIR/whitelist.json}"

echo "=========================================="
echo "Stack-Cat Debug"
echo "=========================================="
echo ""
echo "ROOT_DIR: $ROOT_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "GROUPS_JSON: $GROUPS_JSON"
echo "WHITELIST_JSON: $WHITELIST_JSON"
echo ""

# Check whitelist file
echo "1. Whitelist file check:"
if [[ -f "$WHITELIST_JSON" ]]; then
    echo "   ✓ File exists"
    echo "   Contents:"
    cat "$WHITELIST_JSON" | jq . || echo "   ERROR: Invalid JSON"
else
    echo "   ✗ File NOT found at: $WHITELIST_JSON"
    echo "   Looking for whitelist.json in:"
    find "$SCRIPT_DIR" -name "whitelist.json" -o -name "*.json" | head -10
fi
echo ""

# Check if files can be found without whitelist
echo "2. Testing pattern expansion (no whitelist):"
echo "   Dockerfile* pattern:"
find "$ROOT_DIR" -maxdepth 1 -name "Dockerfile*" -type f 2>/dev/null | head -5

echo "   docker-compose.yml pattern:"
find "$ROOT_DIR" -maxdepth 1 -name "docker-compose.yml" -type f 2>/dev/null | head -5

echo "   app/**/*.py pattern:"
find "$ROOT_DIR/app" -name "*.py" -type f 2>/dev/null | head -5
echo ""

# Test whitelist filtering logic
echo "3. Testing whitelist logic:"
echo "   Loading whitelist arrays..."

if [[ -f "$WHITELIST_JSON" ]]; then
    mapfile -t ALLOWED_ROOTS < <(jq -r '.allowed_roots[]?' "$WHITELIST_JSON" 2>/dev/null)
    mapfile -t ALLOWED_DIRS < <(jq -r '.allowed_dirs[]?' "$WHITELIST_JSON" 2>/dev/null)
    mapfile -t EXCLUDED_EXTS < <(jq -r '.excluded_extensions[]?' "$WHITELIST_JSON" 2>/dev/null)
    
    echo "   Allowed roots (${#ALLOWED_ROOTS[@]}):"
    printf '     - %s\n' "${ALLOWED_ROOTS[@]}"
    echo "   Allowed dirs (${#ALLOWED_DIRS[@]}):"
    printf '     - %s\n' "${ALLOWED_DIRS[@]}"
    echo "   Excluded extensions (${#EXCLUDED_EXTS[@]}):"
    printf '     - %s\n' "${EXCLUDED_EXTS[@]}"
    echo ""
    
    # Test a sample file
    echo "4. Testing sample file: $ROOT_DIR/docker-compose.yml"
    local file="$ROOT_DIR/docker-compose.yml"
    local rel_path="${file#$ROOT_DIR/}"
    echo "   Relative path: $rel_path"
    
    # Check excluded extensions
    local allowed=0
    for ext in "${EXCLUDED_EXTS[@]}"; do
        if [[ "$file" == *"$ext" ]]; then
            echo "   ✗ Excluded by extension: $ext"
            allowed=1
        fi
    done
    
    if [[ $allowed -eq 0 ]]; then
        # Check if it's an allowed root file
        for allowed_root in "${ALLOWED_ROOTS[@]}"; do
            if [[ "$rel_path" == "$allowed_root" ]]; then
                echo "   ✓ Matches allowed_root: $allowed_root"
                allowed=1
                break
            fi
        done
    fi
    
    if [[ $allowed -eq 0 ]]; then
        # Check if it's in an allowed directory
        for allowed_dir in "${ALLOWED_DIRS[@]}"; do
            if [[ "$rel_path" == "$allowed_dir"* ]]; then
                echo "   ✓ Matches allowed_dir: $allowed_dir"
                allowed=1
                break
            fi
        done
    fi
    
    if [[ $allowed -eq 0 ]]; then
        echo "   ✗ NOT ALLOWED - does not match any whitelist criteria"
    fi
fi