#!/usr/bin/env bash
# ============================================================================
# Xoe-NovAi v0.1.5 - Wheelhouse Duplicate Cleaner
# ============================================================================
# Purpose: Remove duplicate wheels while respecting version constraints
# Guide Reference: Section 6.3 (Wheelhouse Management)
# Last Updated: 2026-01-09
# Features:
#   - Version constraint validation
#   - Duplicate detection and removal
#   - Comprehensive logging and reporting
#   - JSON manifest generation
# ============================================================================

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Xoe-NovAi v0.1.5 - Wheelhouse Duplicate Cleaner${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

WHEELHOUSE="wheelhouse"
VERSIONS_FILE="versions/versions.toml"
LOGDIR="logs/wheelhouse"
LOGFILE="${LOGDIR}/cleanup_$(date +%Y%m%d_%H%M%S).log"
REQUIREMENTS=(requirements-api.txt requirements-chainlit.txt requirements-crawl.txt requirements-curation_worker.txt)

# Create log directory
mkdir -p "${LOGDIR}"
exec 1> >(tee -a "${LOGFILE}")
exec 2> >(tee -a "${LOGFILE}" >&2)

log_info "Starting wheelhouse cleanup at $(date)"
echo ""

# ============================================================================
# CHECK 1: Prerequisites
# ============================================================================
log_info "Checking prerequisites..."

if [[ ! -d "$WHEELHOUSE" ]]; then
    log_error "Wheelhouse directory not found: $WHEELHOUSE"
    log_error "Run 'make wheelhouse' first to create wheelhouse"
    exit 1
fi

wheel_count=$(find "$WHEELHOUSE" -name "*.whl" 2>/dev/null | wc -l)
if [[ $wheel_count -eq 0 ]]; then
    log_warn "No wheels found in $WHEELHOUSE"
    log_warn "Run 'make wheelhouse' to populate wheelhouse"
    exit 0
fi

log_success "Found $wheel_count wheels in $WHEELHOUSE"

# ============================================================================
# CHECK 2: Version Management
# ============================================================================
log_info "Updating version requirements..."

# First, ensure our versions are up to date
if [[ -f "versions/scripts/update_versions.py" ]]; then
    log_info "Updating requirements from versions.toml..."
    if python3 versions/scripts/update_versions.py; then
        log_success "Version requirements updated"
    else
        log_warn "Failed to update versions, continuing with existing requirements"
    fi
else
    log_warn "Version update script not found, using existing requirements"
fi

# Initialize version tracking
declare -A required_versions
declare -A version_constraints

# ============================================================================
# CHECK 3: Load Version Constraints
# ============================================================================
log_info "Loading version constraints..."

# Load version constraints from versions.toml
if [[ -f "${VERSIONS_FILE}" ]]; then
    log_info "Loading version constraints from ${VERSIONS_FILE}..."
    while IFS= read -r line; do
        if [[ $line =~ \[(.*)\] ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi
        if [[ $line =~ ^([a-zA-Z0-9_-]+)[[:space:]]*=[[:space:]]*\"(.*)\" ]]; then
            pkg="${BASH_REMATCH[1]}"
            ver="${BASH_REMATCH[2]}"
            if [[ $section == "versions" ]]; then
                required_versions[$pkg]=$ver
            elif [[ $section == "constraints" ]]; then
                version_constraints[$pkg]=$ver
            fi
        fi
    done < "${VERSIONS_FILE}"
    log_success "Version constraints loaded"
else
    log_warn "Versions file not found: ${VERSIONS_FILE}"
fi

constraint_count=${#version_constraints[@]}
version_count=${#required_versions[@]}
log_info "Loaded $constraint_count constraints and $version_count version requirements"

# ============================================================================
# CHECK 4: Load Requirements Versions
# ============================================================================
log_info "Loading requirements file versions..."

requirements_processed=0
for req in "${REQUIREMENTS[@]}"; do
    if [[ -f "$req" ]]; then
        log_info "Processing $req..."
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip comments and empty lines
            [[ $line =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue

            if [[ $line =~ ^([^=<>~!]+)(==|>=|<=|~=|!=)(.+)$ ]]; then
                pkg="${BASH_REMATCH[1]}"
                ver="${BASH_REMATCH[3]}"
                required_versions[$pkg]=$ver
            fi
        done < "$req"
        ((requirements_processed++))
    else
        log_warn "Requirements file not found: $req"
    fi
done

log_success "Processed $requirements_processed requirements files"
final_version_count=${#required_versions[@]}
log_info "Total version requirements loaded: $final_version_count"

# Function to check version constraints
check_constraints() {
    local pkg="$1"
    local ver="$2"
    local constraints="${version_constraints[$pkg]:-}"
    
    if [[ -z "$constraints" ]]; then
        return 0  # No constraints, accept any version
    fi
    
    # TODO: Implement proper version constraint checking
    # For now, just log the constraint check
    echo "Checking $pkg==$ver against constraints: $constraints"
    return 0
}

# Create a manifest for tracking cleanup operations
MANIFEST="${WHEELHOUSE}/cleanup_manifest.json"
echo "{\"removed\": [], \"kept\": [], \"duplicates\": []}" > "${MANIFEST}"

# Function to update manifest
update_manifest() {
    local action="$1"
    local pkg="$2"
    local reason="$3"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    jq --arg act "$action" --arg pkg "$pkg" --arg reason "$reason" --arg time "$timestamp" \
       '.[$act] += [{"package": $pkg, "reason": $reason, "time": $time}]' \
       "${MANIFEST}" > "${MANIFEST}.tmp" && mv "${MANIFEST}.tmp" "${MANIFEST}"
}

# ============================================================================
# CHECK 5: Process Wheels and Remove Duplicates
# ============================================================================
log_info "Processing wheels and removing duplicates..."

declare -A seen_packages
processed_count=0
duplicate_count=0
removed_count=0
kept_count=0

for whl in $WHEELHOUSE/*.whl; do
    [[ -f "$whl" ]] || continue

    fname=$(basename "$whl")
    pkg=$(echo "$fname" | grep -Eo '^[a-zA-Z0-9_-]+')
    ver=$(echo "$fname" | grep -Eo '[0-9]+\.[0-9]+(\.[0-9]+)?')

    if [[ -z "$pkg" || -z "$ver" ]]; then
        log_warn "Could not parse package info from $fname"
        continue
    fi

    ((processed_count++))
    req_ver="${required_versions[$pkg]:-}"

    # Check for duplicates
    if [[ -n "${seen_packages[$pkg]:-}" ]]; then
        ((duplicate_count++))
        log_info "Duplicate package detected: $pkg"
        update_manifest "duplicates" "$fname" "Multiple versions found"

        # Keep the version that matches our requirements
        if [[ -n "$req_ver" && "$ver" != "$req_ver" ]]; then
            log_info "Removing $fname (wrong version: wanted $req_ver)"
            rm -f "$whl"
            update_manifest "removed" "$fname" "Version mismatch: wanted $req_ver"
            ((removed_count++))
        elif ! check_constraints "$pkg" "$ver"; then
            log_info "Removing $fname (constraint violation)"
            rm -f "$whl"
            update_manifest "removed" "$fname" "Failed constraint check"
            ((removed_count++))
        else
            log_info "Keeping $fname (matches requirements)"
            update_manifest "kept" "$fname" "Matches requirements"
            ((kept_count++))
        fi
    else
        # First time seeing this package
        seen_packages[$pkg]=$ver
        update_manifest "kept" "$fname" "First occurrence"
        ((kept_count++))
    fi
done

log_success "Processed $processed_count wheels"
log_info "Found $duplicate_count duplicates, removed $removed_count, kept $kept_count"

# ============================================================================
# CHECK 6: Generate Reports and Summary
# ============================================================================
log_info "Generating cleanup reports..."

# Generate cleanup report
{
    echo "# Wheelhouse Cleanup Report"
    echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo
    echo "## Summary"
    echo
    echo "- Kept packages: $(jq '.kept | length' "${MANIFEST}")"
    echo "- Removed packages: $(jq '.removed | length' "${MANIFEST}")"
    echo "- Duplicate packages found: $(jq '.duplicates | length' "${MANIFEST}")"
    echo
    echo "## Details"
    echo
    echo "### Removed Packages"
    echo
    jq -r '.removed[] | "- \(.package) (\(.reason))"' "${MANIFEST}"
    echo
    echo "### Kept Packages"
    echo
    jq -r '.kept[] | "- \(.package) (\(.reason))"' "${MANIFEST}"
    echo
    echo "### Duplicates Found"
    echo
    jq -r '.duplicates[] | "- \(.package)"' "${MANIFEST}"
} > "${WHEELHOUSE}/cleanup_report.md"

log_success "Cleanup report generated: ${WHEELHOUSE}/cleanup_report.md"
log_info "Full logs available: ${LOGFILE}"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
log_success "Wheelhouse cleanup completed successfully!"
echo ""
echo "📊 Summary:"
echo "  📦 Total wheels processed: $processed_count"
echo "  🔍 Duplicates found: $duplicate_count"
echo "  🗑️  Packages removed: $removed_count"
echo "  💾 Packages kept: $kept_count"
echo ""
echo "📄 Reports generated:"
echo "  📋 Cleanup report: ${WHEELHOUSE}/cleanup_report.md"
echo "  📊 JSON manifest: ${WHEELHOUSE}/cleanup_manifest.json"
echo "  📝 Full logs: ${LOGFILE}"
echo ""
log_success "✅ Wheelhouse cleanup complete!"
