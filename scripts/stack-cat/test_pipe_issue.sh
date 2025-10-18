#!/usr/bin/env bash
# Test if the pipe is losing data
set -euo pipefail

ROOT_DIR="/home/arcana-novai/Documents/XNAi-v0_1_3-clean"

expand_pattern() {
    local pattern="$1"
    
    if [[ "$pattern" == *"*"* ]]; then
        find "$ROOT_DIR" -maxdepth 1 -type f -name "${pattern##*/}" 2>/dev/null || true
    fi
}

echo "=== Test 1: Direct call to expand_pattern ==="
expand_pattern "Dockerfile*"

echo ""
echo "=== Test 2: Store in variable ==="
result=$(expand_pattern "Dockerfile*")
echo "Result: $result"
echo "Line count: $(echo "$result" | wc -l)"

echo ""
echo "=== Test 3: Loop through results ==="
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        echo "  → $file"
    fi
done < <(expand_pattern "Dockerfile*")

echo ""
echo "=== Test 4: The exact pattern from groups.json ==="
pattern="Dockerfile*"
echo "Pattern: $pattern"
echo "Files found:"
while IFS= read -r file; do
    if [[ -n "$file" ]] && [[ -f "$file" ]]; then
        echo "  ✓ $file"
    fi
done < <(expand_pattern "$pattern")