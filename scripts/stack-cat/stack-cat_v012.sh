#!/bin/bash
set -euo pipefail

# ============================================================================
# stack-cat v0.15 - FIXED VERSION
# ============================================================================

# Use relative paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/stack-cat-output"

# Create output directory relative to script
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Simple Configuration
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# ============================================================================
# Collect files using simple find
# ============================================================================

collect_files() {
    log "Collecting files from: ${PROJECT_ROOT}"
    
    local -a files=()
    
    # Find all relevant files
    while IFS= read -r -d '' file; do
        # Skip hidden directories and common exclusions
        if [[ "$file" =~ /\. ]] || \
           [[ "$file" =~ /__pycache__/ ]] || \
           [[ "$file" =~ /\.venv/ ]] || \
           [[ "$file" =~ /venv/ ]] || \
           [[ "$file" =~ /node_modules/ ]] || \
           [[ "$file" =~ /\.git/ ]] || \
           [[ "$file" =~ /\.pytest_cache/ ]] || \
           [[ "$file" =~ /stack-cat-output/ ]] || \
           [[ "$file" =~ \.pyc$ ]] || \
           [[ "$file" =~ \.log$ ]] || \
           [[ "$file" =~ \.tmp$ ]]; then
            continue
        fi
        
        files+=("$file")
    done < <(
        find "${PROJECT_ROOT}" -type f \( \
            -name "*.py" -o \
            -name "*.js" -o \
            -name "*.ts" -o \
            -name "*.jsx" -o \
            -name "*.tsx" -o \
            -name "*.sh" -o \
            -name "*.bash" -o \
            -name "*.yml" -o \
            -name "*.yaml" -o \
            -name "*.json" -o \
            -name "*.toml" -o \
            -name "*.md" -o \
            -name "*.txt" -o \
            -name "Dockerfile*" -o \
            -name "docker-compose.yml" \
        \) -print0 2>/dev/null
    )
    
    printf '%s\n' "${files[@]}"
}

get_file_type() {
    local file="$1"
    local filename=$(basename "$file")
    
    if [[ "$filename" =~ ^Dockerfile ]]; then
        echo "Dockerfile"
    elif [[ "$filename" == "docker-compose.yml" ]]; then
        echo "Docker Compose"
    elif [[ "$filename" == *.py ]]; then
        echo "Python"
    elif [[ "$filename" == *.sh ]] || [[ "$filename" == *.bash ]]; then
        echo "Shell Script"
    elif [[ "$filename" == *.js ]]; then
        echo "JavaScript"
    elif [[ "$filename" == *.ts ]]; then
        echo "TypeScript"
    elif [[ "$filename" == *.yml ]] || [[ "$filename" == *.yaml ]]; then
        echo "YAML"
    elif [[ "$filename" == *.json ]]; then
        echo "JSON"
    elif [[ "$filename" == *.toml ]]; then
        echo "TOML"
    elif [[ "$filename" == *.md ]]; then
        echo "Markdown"
    elif [[ "$filename" == *.txt ]]; then
        echo "Text"
    else
        echo "Text"
    fi
}

generate_output() {
    local -a files=("$@")
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/stack-cat_${timestamp}.txt"
    
    log "Generating: ${output_file}"
    log "Processing ${#files[@]} files..."
    
    # Write header
    cat > "${output_file}" <<EOF
################################################################################
# Stack-Cat Output
# Generated: $(date +'%Y-%m-%d %H:%M:%S')
# Project: ${PROJECT_ROOT}
# Files: ${#files[@]}
################################################################################

EOF
    
    # Process each file
    local count=0
    for file in "${files[@]}"; do
        local rel_path="${file#${PROJECT_ROOT}/}"
        local file_type=$(get_file_type "$file")
        
        # Write file header
        cat >> "${output_file}" <<EOF
################################################################################
# File: ${rel_path}
# Type: ${file_type}
################################################################################

EOF
        
        # Write file content
        if [[ -r "$file" ]]; then
            cat "$file" >> "${output_file}" 2>/dev/null || echo "# ERROR: Cannot read file" >> "${output_file}"
        else
            echo "# ERROR: Cannot read file" >> "${output_file}"
        fi
        
        # Add spacing
        echo "" >> "${output_file}"
        echo "" >> "${output_file}"
        
        ((count++))
        if ((count % 10 == 0)); then
            log "Processed ${count}/${#files[@]} files"
        fi
    done
    
    log "Complete: ${output_file}"
    
    # Create symlink to latest
    cd "${OUTPUT_DIR}"
    ln -sf "$(basename "${output_file}")" "stack-cat_latest.txt"
    cd - > /dev/null
    
    # Show file info
    local file_size=$(ls -lh "${output_file}" | awk '{print $5}')
    log "Output size: ${file_size}"
    log "Location: ${output_file}"
}

# ============================================================================
# Main
# ============================================================================

main() {
    log "Stack-Cat v0.15"
    log "Script location: ${SCRIPT_DIR}"
    log "Project root: ${PROJECT_ROOT}"
    log "Output directory: ${OUTPUT_DIR}"
    echo ""
    
    # Collect files
    log "Scanning project..."
    mapfile -t FILES < <(collect_files)
    
    log "Found ${#FILES[@]} files"
    
    if [[ ${#FILES[@]} -eq 0 ]]; then
        log "ERROR: No files found"
        log "Check that ${PROJECT_ROOT} contains code files"
        exit 1
    fi
    
    # Show sample of files
    log "Sample files:"
    printf '%s\n' "${FILES[@]}" | head -10 | while read -r f; do
        log "  - ${f#${PROJECT_ROOT}/}"
    done
    
    if [[ ${#FILES[@]} -gt 10 ]]; then
        log "  ... and $((${#FILES[@]} - 10)) more"
    fi
    echo ""
    
    # Generate output
    generate_output "${FILES[@]}"
    
    echo ""
    log "SUCCESS!"
    log "Output location: ${OUTPUT_DIR}"
    log "Latest file: ${OUTPUT_DIR}/stack-cat_latest.txt"
    echo ""
    log "To view the output:"
    log "  cat ${OUTPUT_DIR}/stack-cat_latest.txt"
    log "  or"
    log "  less ${OUTPUT_DIR}/stack-cat_latest.txt"
}

main "$@"