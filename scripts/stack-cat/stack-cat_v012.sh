#!/usr/bin/env bash
# stack-cat_v018n.sh – XNAi Stack Concatenation (Enterprise)
set -euo pipefail

# --- Configuration ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="$(realpath "$BASE_DIR/../..")"  # Repo root
readonly OUTPUT_DIR="$BASE_DIR/stack-cat-output"
readonly SPLIT_DIR="${OUTPUT_DIR}/splits"
readonly INDIVIDUAL_DIR="${OUTPUT_DIR}/individual"
readonly SPLIT_SIZE="${SPLIT_SIZE:-5000}"  # Default split size in lines
readonly GENERATE_HTML="${GENERATE_HTML:-true}"
readonly VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-false}"

# --- Whitelists ---
readonly ALLOWED_ROOT_FILES=(
    "README.md" "LICENSE" "CHANGELOG.md" ".env.example"
    "Dockerfile" "docker-compose.yml" "requirements.txt" "stack-cat_v012.sh"
)

# --- Groups & patterns ---
readonly GROUPS_JSON="$BASE_DIR/groups.json"

# --- Logging ---
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$1] $2"; }

# --- Dry-run mode ---
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]] || [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    log "INFO" "=== DRY RUN MODE ==="
fi

# --- Load group patterns ---
get_group_patterns() {
    local group="$1"
    jq -r --arg g "$group" '.[$g][]?' "$GROUPS_JSON" 2>/dev/null || echo ""
}

# --- Collect files ---
collect_files() {
    local group="$1"
    local patterns
    mapfile -t patterns < <(get_group_patterns "$group")

    # Always include whitelisted root files if they exist
    local files=()
    for file in "${ALLOWED_ROOT_FILES[@]}"; do
        local full="$ROOT_DIR/$file"
        if [[ -f "$full" ]]; then
            files+=("$full")
        else
            log "WARN" "Whitelisted root file missing: $full"
        fi
    done

    # Collect group pattern files
    for pattern in "${patterns[@]}"; do
        while IFS= read -r f; do
            files+=("$f")
        done < <(find "$ROOT_DIR" -type f -path "$ROOT_DIR/$pattern" 2>/dev/null)
    done

    echo "${files[@]}"
}

# --- Main ---
GROUP="${1:-default}"
log "INFO" "Starting XNAi Stack Concatenation (Group: $GROUP)"

FILES=( $(collect_files "$GROUP") )

if [[ ${#FILES[@]} -eq 0 ]]; then
    log "ERROR" "No files matched the whitelist or patterns."
    [[ "$DRY_RUN" == "true" ]] && log "INFO" "Would process 0 files:" || exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log "INFO" "Would process ${#FILES[@]} files:"
    for f in "${FILES[@]}"; do
        echo " - $f"
    done
    log "INFO" "Dry run complete. No files written."
    exit 0
fi

# --- Prepare output dirs ---
mkdir -p "$SPLIT_DIR" "$INDIVIDUAL_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_MD="$OUTPUT_DIR/stack-concat.md"
> "$OUTPUT_MD"

# --- Concatenate ---
for f in "${FILES[@]}"; do
    echo "## $f" >> "$OUTPUT_MD"
    cat "$f" >> "$OUTPUT_MD"
    echo -e "\n" >> "$OUTPUT_MD"

    # Save individual file copy
    cp "$f" "$INDIVIDUAL_DIR/"
done

# --- Optionally generate HTML ---
if [[ "$GENERATE_HTML" == "true" ]]; then
    OUTPUT_HTML="${OUTPUT_MD%.md}.html"
    pandoc "$OUTPUT_MD" -s -o "$OUTPUT_HTML"
fi

log "INFO" "✅ Completed successfully"
log "INFO" "Master Markdown: $OUTPUT_MD"
[[ "$GENERATE_HTML" == "true" ]] && log "INFO" "Master HTML: $OUTPUT_HTML"
log "INFO" "Individual files: $INDIVIDUAL_DIR"
log "INFO" "Splits: $SPLIT_DIR"
