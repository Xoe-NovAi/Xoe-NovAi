#!/bin/bash
# clean_wheelhouse_duplicates.sh
# Enhanced wheelhouse cleaner that works with version management system
# Removes duplicate wheels while respecting version constraints
set -euo pipefail

WHEELHOUSE="wheelhouse"
VERSIONS_FILE="versions/versions.toml"
LOGDIR="logs/wheelhouse"
LOGFILE="${LOGDIR}/cleanup_$(date +%Y%m%d_%H%M%S).log"
REQUIREMENTS=(requirements-api.txt requirements-chainlit.txt requirements-crawl.txt requirements-curation_worker.txt)

# Create log directory
mkdir -p "${LOGDIR}"
exec 1> >(tee -a "${LOGFILE}")
exec 2> >(tee -a "${LOGFILE}" >&2)

echo "Starting wheelhouse cleanup at $(date)"

# First, ensure our versions are up to date
if [[ -f "versions/scripts/update_versions.py" ]]; then
    echo "Updating requirements from versions.toml..."
    python3 versions/scripts/update_versions.py
fi

# Initialize version tracking
declare -A required_versions
declare -A version_constraints

# Load version constraints from versions.toml
if [[ -f "${VERSIONS_FILE}" ]]; then
    echo "Loading version constraints from ${VERSIONS_FILE}..."
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
fi

# Also collect versions from requirements files
for req in "${REQUIREMENTS[@]}"; do
    if [[ -f "$req" ]]; then
        echo "Processing $req..."
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
    fi
done

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

# Process wheels
echo "Processing wheels in ${WHEELHOUSE}..."
declare -A seen_packages
for whl in $WHEELHOUSE/*.whl; do
    [[ -f "$whl" ]] || continue
    
    fname=$(basename "$whl")
    pkg=$(echo "$fname" | grep -Eo '^[a-zA-Z0-9_-]+')
    ver=$(echo "$fname" | grep -Eo '[0-9]+\.[0-9]+(\.[0-9]+)?')
    
    if [[ -z "$pkg" || -z "$ver" ]]; then
        echo "Warning: Could not parse package info from $fname"
        continue
    fi
    
    req_ver="${required_versions[$pkg]:-}"
    
    # Check for duplicates
    if [[ -n "${seen_packages[$pkg]:-}" ]]; then
        echo "Duplicate package detected: $pkg"
        update_manifest "duplicates" "$fname" "Multiple versions found"
        
        # Keep the version that matches our requirements
        if [[ -n "$req_ver" && "$ver" != "$req_ver" ]]; then
            echo "Removing $fname (wrong version)"
            rm -f "$whl"
            update_manifest "removed" "$fname" "Version mismatch: wanted $req_ver"
        elif ! check_constraints "$pkg" "$ver"; then
            echo "Removing $fname (constraint violation)"
            rm -f "$whl"
            update_manifest "removed" "$fname" "Failed constraint check"
        else
            echo "Keeping $fname (matches requirements)"
            update_manifest "kept" "$fname" "Matches requirements"
        fi
    else
        # First time seeing this package
        seen_packages[$pkg]=$ver
        update_manifest "kept" "$fname" "First occurrence"
    fi
done

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

echo "Wheelhouse cleanup complete. See ${WHEELHOUSE}/cleanup_report.md for details."
echo "Full logs available in: ${LOGFILE}"
