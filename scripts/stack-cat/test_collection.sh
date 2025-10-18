#!/usr/bin/env bash
# Test file collection with all steps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/arcana-novai/Documents/XNAi-v0_1_3-clean"
GROUPS_JSON="$SCRIPT_DIR/groups.json"
WHITELIST_JSON="$SCRIPT_DIR/whitelist.json"

declare -a ALLOWED_ROOTS
declare -a ALLOWED_DIRS
declare -a EXCLUDED_EXTS

load_whitelist() {
    mapfile -t ALLOWED_ROOTS < <(jq -r '.allowed_roots[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
    mapfile -t ALLOWED_DIRS < <(jq -r '.allowed_dirs[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
    mapfile -t EXCLUDED_EXTS < <(jq -r '.excluded_extensions[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
}

is_file_allowed() {
    local file="$1"
    local rel_path="${file#$ROOT_DIR/}"
    
    for ext in "${EXCLUDED_EXTS[@]}"; do
        if [[ "$file" == *"$ext" ]]; then
            return 1
        fi
    done
    
    for allowed in "${ALLOWED_ROOTS[@]}"; do
        if [[ "$rel_path" == "$allowed" ]]; then
            return 0
        fi
    done
    
    for allowed_dir in "${ALLOWED_DIRS[@]}"; do
        allowed_dir="${allowed_dir%/}"
        if [[ "$rel_path" == "$allowed_dir"/* ]] || [[ "$rel_path" == "$allowed_dir" ]]; then
            return 0
        fi
    done
    
    return 1
}

expand_pattern() {
    local pattern="$1"
    
    echo "[EXPAND] Pattern: $pattern" >&2
    
    if [[ "$pattern" == *"**"* ]]; then
        local dir_part="${pattern%%/**}"
        local file_part="${pattern##**/}"
        
        echo "[EXPAND]   Type: recursive (**)" >&2
        echo "[EXPAND]   Dir: $dir_part, File: $file_part" >&2
        
        if [[ -d "$ROOT_DIR/$dir_part" ]]; then
            find "$ROOT_DIR/$dir_part" -type f -name "$file_part" 2>/dev/null || true
        fi
    elif [[ "$pattern" == *"*"* ]]; then
        echo "[EXPAND]   Type: wildcard (*)" >&2
        find "$ROOT_DIR" -maxdepth 1 -type f -name "${pattern##*/}" 2>/dev/null || true
    else
        echo "[EXPAND]   Type: exact" >&2
        local base_path="$ROOT_DIR/$pattern"
        if [[ -f "$base_path" ]]; then
            echo "$base_path"
        fi
    fi
}

get_group_patterns() {
    local group="$1"
    jq -r --arg g "$group" '.[$g].files[]?' "$GROUPS_JSON" 2>/dev/null
}

echo "=== Step 1: Load Whitelist ==="
load_whitelist
echo "Loaded: ${#ALLOWED_ROOTS[@]} roots, ${#ALLOWED_DIRS[@]} dirs"
echo ""

echo "=== Step 2: Get Group Patterns ==="
mapfile -t patterns < <(get_group_patterns "default")
echo "Found ${#patterns[@]} patterns:"
printf '  - %s\n' "${patterns[@]}"
echo ""

echo "=== Step 3: Expand and Filter Patterns ==="
declare -a files=()

for pattern in "${patterns[@]}"; do
    echo ""
    echo "[PATTERN] $pattern"
    
    expanded_count=0
    allowed_count=0
    
    while IFS= read -r file; do
        if [[ -n "$file" ]] && [[ -f "$file" ]]; then
            ((expanded_count++))
            
            if is_file_allowed "$file"; then
                ((allowed_count++))
                files+=("$file")
                echo "  [OK] ${file#$ROOT_DIR/}"
            else
                echo "  [BLOCKED] ${file#$ROOT_DIR/}"
            fi
        fi
    done < <(expand_pattern "$pattern")
    
    echo "  → Expanded: $expanded_count, Allowed: $allowed_count"
done

echo ""
echo "=== Step 4: Summary ==="
echo "Total files collected: ${#files[@]}"
if [[ ${#files[@]} -gt 0 ]]; then
    echo "Files:"
    printf '  - %s\n' "${files[@]}"
fi