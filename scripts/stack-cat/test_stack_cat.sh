#!/usr/bin/env bash
# test_stack_cat.sh - Test suite for stack-cat_v012_fixed.sh
# Guide Ref: Section 10 (Testing & Validation)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_CAT="$SCRIPT_DIR/stack-cat_v012_fixed.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# --- Test helpers ---
log_test() {
    echo -e "${YELLOW}[TEST]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    ((TESTS_FAILED++))
}

run_test() {
    ((TESTS_RUN++))
    local test_name="$1"
    log_test "$test_name"
}

# --- Setup test environment ---
setup_test_env() {
    log_test "Setting up test environment..."
    
    # Create temporary directory structure
    export TEST_DIR="$(mktemp -d)"
    export TEST_ROOT="$TEST_DIR/xnai-test"
    
    mkdir -p "$TEST_ROOT"/{app/XNAi_rag_app,configs,scripts,tests}
    
    # Create test files
    cat > "$TEST_ROOT/README.md" <<'EOF'
# Test README
This is a test file.
EOF
    
    cat > "$TEST_ROOT/Dockerfile" <<'EOF'
FROM python:3.12
RUN echo "test"
EOF
    
    cat > "$TEST_ROOT/docker-compose.yml" <<'EOF'
version: '3.9'
services:
  test: {}
EOF
    
    cat > "$TEST_ROOT/app/XNAi_rag_app/main.py" <<'EOF'
# main.py
def main():
    print("Hello XNAi")
EOF
    
    cat > "$TEST_ROOT/app/XNAi_rag_app/config.py" <<'EOF'
# config.py
CONFIG = {"test": True}
EOF
    
    cat > "$TEST_ROOT/configs/rag.yml" <<'EOF'
rag:
  enabled: true
EOF
    
    cat > "$TEST_ROOT/scripts/test.sh" <<'EOF'
#!/bin/bash
echo "test script"
EOF
    chmod +x "$TEST_ROOT/scripts/test.sh"
    
    # Create groups.json
    cat > "$TEST_DIR/groups.json" <<'EOF'
{
  "default": {
    "description": "Full core stack",
    "files": [
      "Dockerfile*",
      "docker-compose.yml",
      "README.md",
      "app/**/*.py",
      "configs/**/*.yml",
      "scripts/**/*.sh"
    ]
  },
  "api": {
    "description": "API only",
    "files": [
      "Dockerfile",
      "app/XNAi_rag_app/main.py"
    ]
  },
  "minimal": {
    "description": "Minimal files",
    "files": [
      "README.md"
    ]
  }
}
EOF
    
    # Create whitelist.json
    cat > "$TEST_DIR/whitelist.json" <<'EOF'
{
  "allowed_roots": [
    "README.md",
    "Dockerfile",
    "docker-compose.yml"
  ],
  "allowed_dirs": [
    "app/XNAi_rag_app",
    "configs",
    "scripts"
  ],
  "excluded_extensions": [
    ".log",
    ".tmp"
  ]
}
EOF
    
    log_pass "Test environment created at: $TEST_DIR"
}

cleanup_test_env() {
    if [[ -n "${TEST_DIR:-}" ]] && [[ -d "$TEST_DIR" ]]; then
        rm -rf "$TEST_DIR"
        log_test "Cleaned up test environment"
    fi
}

# --- Test cases ---
test_dry_run() {
    run_test "Test dry run mode"
    
    local output
    output=$(ROOT_DIR="$TEST_ROOT" \
             GROUPS_JSON="$TEST_DIR/groups.json" \
             OUTPUT_DIR="$TEST_DIR/output" \
             bash "$STACK_CAT" --dry-run default 2>&1)
    
    if echo "$output" | grep -q "DRY RUN" && \
       echo "$output" | grep -q "Would process:"; then
        log_pass "Dry run works correctly"
    else
        log_fail "Dry run output incorrect"
        echo "$output"
    fi
}

test_group_default() {
    run_test "Test default group"
    
    ROOT_DIR="$TEST_ROOT" \
    GROUPS_JSON="$TEST_DIR/groups.json" \
    WHITELIST_JSON="$TEST_DIR/whitelist.json" \
    OUTPUT_DIR="$TEST_DIR/output" \
    bash "$STACK_CAT" default 2>&1 | grep -q "Completed successfully"
    
    local result=$?
    
    if [[ $result -eq 0 ]]; then
        # Check output file exists
        local md_file
        md_file=$(find "$TEST_DIR/output" -name "stack-concat_default_*.md" | head -1)
        
        if [[ -f "$md_file" ]]; then
            # Verify content
            if grep -q "README.md" "$md_file" && \
               grep -q "main.py" "$md_file" && \
               grep -q "docker-compose.yml" "$md_file"; then
                log_pass "Default group processed correctly"
            else
                log_fail "Output missing expected files"
                echo "Content preview:"
                head -50 "$md_file"
            fi
        else
            log_fail "Output file not created"
        fi
    else
        log_fail "Default group failed"
    fi
}

test_group_api() {
    run_test "Test API group (subset)"
    
    ROOT_DIR="$TEST_ROOT" \
    GROUPS_JSON="$TEST_DIR/groups.json" \
    OUTPUT_DIR="$TEST_DIR/output-api" \
    bash "$STACK_CAT" api 2>&1 | grep -q "Completed successfully"
    
    if [[ $? -eq 0 ]]; then
        local md_file
        md_file=$(find "$TEST_DIR/output-api" -name "stack-concat_api_*.md" | head -1)
        
        if [[ -f "$md_file" ]]; then
            # API group should have fewer files
            local file_count
            file_count=$(grep -c "^## File:" "$md_file" || echo 0)
            
            if [[ $file_count -lt 5 ]]; then
                log_pass "API group processed correctly (subset of files)"
            else
                log_fail "API group has too many files: $file_count"
            fi
        else
            log_fail "API output file not created"
        fi
    else
        log_fail "API group failed"
    fi
}

test_recursive_patterns() {
    run_test "Test recursive pattern (app/**/*.py)"
    
    ROOT_DIR="$TEST_ROOT" \
    GROUPS_JSON="$TEST_DIR/groups.json" \
    OUTPUT_DIR="$TEST_DIR/output-recursive" \
    bash "$STACK_CAT" default 2>&1 | grep -q "Completed successfully"
    
    if [[ $? -eq 0 ]]; then
        local md_file
        md_file=$(find "$TEST_DIR/output-recursive" -name "stack-concat_*.md" | head -1)
        
        if grep -q "main.py" "$md_file" && grep -q "config.py" "$md_file"; then
            log_pass "Recursive patterns work correctly"
        else
            log_fail "Recursive patterns missing expected files"
        fi
    else
        log_fail "Recursive pattern test failed"
    fi
}

test_whitelist() {
    run_test "Test whitelist filtering"
    
    # Create a .log file (should be excluded)
    echo "test log" > "$TEST_ROOT/test.log"
    
    ROOT_DIR="$TEST_ROOT" \
    GROUPS_JSON="$TEST_DIR/groups.json" \
    WHITELIST_JSON="$TEST_DIR/whitelist.json" \
    OUTPUT_DIR="$TEST_DIR/output-whitelist" \
    bash "$STACK_CAT" default 2>&1 | grep -q "Completed successfully"
    
    if [[ $? -eq 0 ]]; then
        local md_file
        md_file=$(find "$TEST_DIR/output-whitelist" -name "stack-concat_*.md" | head -1)
        
        if ! grep -q "test.log" "$md_file"; then
            log_pass "Whitelist excludes .log files correctly"
        else
            log_fail "Whitelist did not exclude .log file"
        fi
    else
        log_fail "Whitelist test failed"
    fi
}

test_invalid_group() {
    run_test "Test invalid group handling"
    
    local output
    output=$(ROOT_DIR="$TEST_ROOT" \
             GROUPS_JSON="$TEST_DIR/groups.json" \
             OUTPUT_DIR="$TEST_DIR/output-invalid" \
             bash "$STACK_CAT" nonexistent 2>&1 || true)
    
    if echo "$output" | grep -q "Group 'nonexistent' not found"; then
        log_pass "Invalid group handled correctly"
    else
        log_fail "Invalid group error message incorrect"
        echo "$output"
    fi
}

test_verbose_mode() {
    run_test "Test verbose mode"
    
    local output
    output=$(ROOT_DIR="$TEST_ROOT" \
             GROUPS_JSON="$TEST_DIR/groups.json" \
             OUTPUT_DIR="$TEST_DIR/output-verbose" \
             bash "$STACK_CAT" --verbose --dry-run default 2>&1)
    
    if echo "$output" | grep -q "\[DEBUG\]"; then
        log_pass "Verbose mode works correctly"
    else
        log_fail "Verbose mode not showing debug output"
    fi
}

test_checksums() {
    run_test "Test checksum generation"
    
    ROOT_DIR="$TEST_ROOT" \
    GROUPS_JSON="$TEST_DIR/groups.json" \
    OUTPUT_DIR="$TEST_DIR/output-checksum" \
    bash "$STACK_CAT" --verify default 2>&1 | grep -q "Completed successfully"
    
    if [[ $? -eq 0 ]]; then
        local sha_file
        sha_file=$(find "$TEST_DIR/output-checksum" -name "*.sha256" | head -1)
        
        if [[ -f "$sha_file" ]]; then
            log_pass "Checksum file generated"
        else
            log_fail "Checksum file not created"
        fi
    else
        log_fail "Checksum test failed"
    fi
}

# --- Run all tests ---
main() {
    echo "=========================================="
    echo "XNAi Stack-Cat Test Suite"
    echo "=========================================="
    
    # Setup
    setup_test_env
    trap cleanup_test_env EXIT
    
    # Run tests
    test_dry_run
    test_group_default
    test_group_api
    test_recursive_patterns
    test_whitelist
    test_invalid_group
    test_verbose_mode
    test_checksums
    
    # Summary
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo "Tests Run:    $TESTS_RUN"
    echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
    echo "=========================================="
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

main "$@"
