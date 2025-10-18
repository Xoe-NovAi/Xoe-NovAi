#!/usr/bin/env bash
# diagnose_stack_cat.sh - Diagnostic tool for stack-cat issues
# Guide Ref: Section 10 (Testing & Validation)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="${ROOT_DIR:-$(realpath "$SCRIPT_DIR/../..")}"
readonly GROUPS_JSON="${GROUPS_JSON:-$SCRIPT_DIR/groups.json}"
readonly WHITELIST_JSON="${WHITELIST_JSON:-$SCRIPT_DIR/whitelist.json}"

echo "=========================================="
echo "Stack-Cat Diagnostic Tool"
echo "=========================================="
echo ""

# Check 1: Directory structure
echo "1. ROOT_DIR Check"
echo "   Location: $ROOT_DIR"
if [[ -d "$ROOT_DIR" ]]; then
    echo "   Status: ✓ EXISTS"
    echo "   Contents:"
    ls -1 "$ROOT_DIR" | head -20 | sed 's/^/     - /'
else
    echo "   Status: ✗ NOT FOUND"
    exit 1
fi
echo ""

# Check 2: groups.json
echo "2. groups.json Check"
echo "   Location: $GROUPS_JSON"
if [[ -f "$GROUPS_JSON" ]]; then
    echo "   Status: ✓ EXISTS"
    if jq empty "$GROUPS_JSON" 2>/dev/null; then
        echo "   Syntax: ✓ VALID JSON"
        echo "   Groups:"
        jq -r 'keys[]' "$GROUPS_JSON" | sed 's/^/     - /'
        echo ""
        echo "   'default' group patterns:"
        jq -r '.default.files[]?' "$GROUPS_JSON" | sed 's/^/     - /'
    else
        echo "   Syntax: ✗ INVALID JSON"
        exit 1
    fi
else
    echo "   Status: ✗ NOT FOUND"
    exit 1
fi
echo ""

# Check 3: whitelist.json
echo "3. whitelist.json Check"
echo "   Location: $WHITELIST_JSON"
if [[ -f "$WHITELIST_JSON" ]]; then
    echo "   Status: ✓ EXISTS"
    if jq empty "$WHITELIST_JSON" 2>/dev/null; then
        echo "   Syntax: ✓ VALID JSON"
        echo "   Allowed roots:"
        jq -r '.allowed_roots[]?' "$WHITELIST_JSON" 2>/dev/null | sed 's/^/     - /' || echo "     (none)"
        echo "   Allowed dirs:"
        jq -r '.allowed_dirs[]?' "$WHITELIST_JSON" 2>/dev/null | sed 's/^/     - /' || echo "     (none)"
        echo "   Excluded extensions:"
        jq -r '.excluded_extensions[]?' "$WHITELIST_JSON" 2>/dev/null | sed 's/^/     - /' || echo "     (none)"
    else
        echo "   Syntax: ✗ INVALID JSON"
    fi
else
    echo "   Status: ⚠ NOT FOUND (will use defaults)"
fi
echo ""

# Check 4: Test pattern expansion
echo "4. Pattern Expansion Test"
echo "   Testing pattern: Dockerfile*"
if find "$ROOT_DIR" -maxdepth 1 -type f -name "Dockerfile*" 2>/dev/null | grep -q .; then
    echo "   Status: ✓ FOUND FILES"
    find "$ROOT_DIR" -maxdepth 1 -type f -name "Dockerfile*" | sed 's/^/     - /'
else
    echo "   Status: ✗ NO FILES FOUND"
fi
echo ""

echo "   Testing pattern: docker-compose.yml"
if [[ -f "$ROOT_DIR/docker-compose.yml" ]]; then
    echo "   Status: ✓ FOUND"
    echo "     - $ROOT_DIR/docker-compose.yml"
else
    echo "   Status: ✗ NOT FOUND"
fi
echo ""

echo "   Testing pattern: app/**/*.py"
if find "$ROOT_DIR/app" -type f -name "*.py" 2>/dev/null | grep -q .; then
    echo "   Status: ✓ FOUND FILES"
    find "$ROOT_DIR/app" -type f -name "*.py" 2>/dev/null | head -10 | sed 's/^/     - /'
    local count
    count=$(find "$ROOT_DIR/app" -type f -name "*.py" 2>/dev/null | wc -l)
    echo "     ... (total: $count files)"
else
    echo "   Status: ✗ NO FILES FOUND"
    if [[ ! -d "$ROOT_DIR/app" ]]; then
        echo "   Note: app/ directory does not exist"
    fi
fi
echo ""

# Check 5: Actual directory structure
echo "5. Actual Directory Structure"
echo "   First 3 levels:"
tree -L 3 -d "$ROOT_DIR" 2>/dev/null | head -30 || {
    echo "   (tree not installed, using find)"
    find "$ROOT_DIR" -maxdepth 3 -type d | head -30 | sed 's/^/     /'
}
echo ""

# Check 6: Recommendations
echo "=========================================="
echo "Recommendations"
echo "=========================================="

# Check if patterns match structure
if [[ ! -d "$ROOT_DIR/app" ]]; then
    echo "⚠ Pattern 'app/**/*.py' won't match because app/ doesn't exist"
    echo "  → Update groups.json patterns to match your actual structure"
fi

if [[ ! -f "$ROOT_DIR/docker-compose.yml" ]]; then
    echo "⚠ Pattern 'docker-compose.yml' won't match because file doesn't exist"
    echo "  → Check if file is named differently or in subdirectory"
fi

# Check whitelist vs structure
if [[ -f "$WHITELIST_JSON" ]]; then
    local allowed_dirs
    mapfile -t allowed_dirs < <(jq -r '.allowed_dirs[]?' "$WHITELIST_JSON" 2>/dev/null || echo "")
    
    if [[ ${#allowed_dirs[@]} -gt 0 ]]; then
        echo ""
        echo "Whitelist directory validation:"
        for dir in "${allowed_dirs[@]}"; do
            if [[ -d "$ROOT_DIR/$dir" ]]; then
                echo "  ✓ $dir exists"
            else
                echo "  ✗ $dir does not exist"
            fi
        done
    fi
fi

echo ""
echo "=========================================="
echo "Suggested Actions"
echo "=========================================="
echo "1. Update groups.json patterns to match your actual directory structure"
echo "2. Check that whitelist.json isn't excluding needed directories"
echo "3. Verify file permissions allow reading"
echo "4. Try running with a simpler pattern first:"
echo "   jq '.test = {\"description\": \"Test\", \"files\": [\"README.md\"]}' groups.json > groups_test.json"
echo "   GROUPS_JSON=groups_test.json bash stack-cat_v012.sh test"
echo ""