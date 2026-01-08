#!/usr/bin/env bash
# scripts/download_wheelhouse.sh
# Downloads wheels for your project's requirements files into wheelhouse/
# Usage: ./scripts/download_wheelhouse.sh [OUTDIR] [REQ_GLOB]
# Example: ./scripts/download_wheelhouse.sh wheelhouse "requirements-*.txt"
set -euo pipefail

# Setup logging
LOGDIR="logs/wheelhouse"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/download_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "${LOGFILE}")
exec 2> >(tee -a "${LOGFILE}" >&2)

echo "Starting wheelhouse download process at $(date)"

# First, update requirements from versions.toml
if [[ -f "versions/scripts/update_versions.py" ]]; then
    echo "Updating requirements files from versions.toml..."
    python3 versions/scripts/update_versions.py
fi

OUTDIR=${1:-wheelhouse}
REQ_GLOB=${2:-"requirements-*.txt"}

echo "Creating wheelhouse in: ${OUTDIR}"
rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"

# Create a manifest file to track all downloads
MANIFEST="${OUTDIR}/wheelhouse_manifest.json"
echo "{\"downloads\": [], \"errors\": [], \"skipped\": []}" > "${MANIFEST}"

# Function to log download attempts
log_download() {
    local pkg="$1"
    local status="$2"
    local error="${3:-}"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    if [[ $status == "success" ]]; then
        if jq --arg pkg "$pkg" --arg time "$timestamp" \
           '.downloads += [{"package": $pkg, "time": $time}]' \
           "${MANIFEST}" > "${MANIFEST}.tmp" 2>/dev/null && [ -f "${MANIFEST}.tmp" ]; then
            mv "${MANIFEST}.tmp" "${MANIFEST}"
        else
            echo "Warning: Failed to update manifest for $pkg (jq may not be installed)"
        fi
    elif [[ $status == "error" ]]; then
        if jq --arg pkg "$pkg" --arg time "$timestamp" --arg err "$error" \
           '.errors += [{"package": $pkg, "time": $time, "error": $err}]' \
           "${MANIFEST}" > "${MANIFEST}.tmp" 2>/dev/null && [ -f "${MANIFEST}.tmp" ]; then
            mv "${MANIFEST}.tmp" "${MANIFEST}"
        else
            echo "Warning: Failed to update manifest for $pkg (jq may not be installed)"
        fi
    else
        if jq --arg pkg "$pkg" --arg time "$timestamp" \
           '.skipped += [{"package": $pkg, "time": $time}]' \
           "${MANIFEST}" > "${MANIFEST}.tmp" 2>/dev/null && [ -f "${MANIFEST}.tmp" ]; then
            mv "${MANIFEST}.tmp" "${MANIFEST}"
        else
            echo "Warning: Failed to update manifest for $pkg (jq may not be installed)"
        fi
    fi
}

# Always include pip, setuptools, wheel and build dependencies
echo "[1/6] Downloading core build dependencies..."
for pkg in pip setuptools wheel scikit-build-core; do
    echo "Downloading $pkg..."
    if python3 -m pip download --only-binary=:all: "$pkg" -d "${OUTDIR}" 2>/tmp/pip_error; then
        log_download "$pkg" "success"
    else
        error=$(cat /tmp/pip_error)
        log_download "$pkg" "error" "$error"
        echo "Warning: Failed to download $pkg: $error"
    fi
done

# Download build requirements for llama-cpp-python
echo "[2/6] Downloading llama-cpp-python build requirements..."
for pkg in cmake ninja; do
    echo "Downloading $pkg..."
    if python3 -m pip download --only-binary=:all: "$pkg" -d "${OUTDIR}" 2>/tmp/pip_error; then
        log_download "$pkg" "success"
    else
        error=$(cat /tmp/pip_error)
        log_download "$pkg" "error" "$error"
        echo "Warning: Failed to download $pkg: $error"
    fi
done

# Create deps cache if it doesn't exist
DEPS_CACHE=".deps_cache"
if [[ ! -f "${DEPS_CACHE}" ]]; then
    echo "{}" > "${DEPS_CACHE}"
fi

# Function to check if package needs updating
needs_update() {
    local pkg="$1"
    local version="$2"
    local cached_version=$(jq -r ".[\"$pkg\"]" "${DEPS_CACHE}")
    
    if [[ "$cached_version" == "null" ]] || [[ "$cached_version" != "$version" ]]; then
        return 0  # true, needs update
    fi
    return 1  # false, no update needed
}

# Resolve each matching requirements file and download packages
echo "[3/6] Scanning for requirements files to download:"
shopt -s nullglob
REQ_FILES=( $REQ_GLOB )
shopt -u nullglob

if [ ${#REQ_FILES[@]} -eq 0 ]; then
    echo "No requirements files matched pattern '${REQ_GLOB}'. Exiting with error."
    exit 2
fi

# Parse requirements files for version changes
declare -A pkg_versions
for req in "${REQ_FILES[@]}"; do
    echo "  -> Analyzing ${req}..."
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        
        if [[ $line =~ ^([^=<>~!]+)(==|>=|<=|~=|!=)(.+)$ ]]; then
            pkg="${BASH_REMATCH[1]}"
            version="${BASH_REMATCH[3]}"
            pkg_versions[$pkg]=$version
        fi
    done < "$req"
done

# Download packages, tracking new and updated ones
echo "[4/6] Downloading packages..."
for pkg in "${!pkg_versions[@]}"; do
    version="${pkg_versions[$pkg]}"
    
    if needs_update "$pkg" "$version"; then
        echo "  -> Downloading $pkg==$version (new/updated package)"
        if python3 -m pip download "$pkg==$version" --no-deps -d "${OUTDIR}" 2>/tmp/pip_error; then
            log_download "$pkg" "success"
            # Update cache
            jq --arg pkg "$pkg" --arg ver "$version" \
               '.[$pkg] = $ver' "${DEPS_CACHE}" > "${DEPS_CACHE}.tmp" && \
               mv "${DEPS_CACHE}.tmp" "${DEPS_CACHE}"
        else
            error=$(cat /tmp/pip_error)
            log_download "$pkg" "error" "$error"
            echo "Warning: Failed to download $pkg==$version: $error"
        fi
    else
        echo "  -> Skipping $pkg==$version (already cached)"
        log_download "$pkg" "skipped"
    fi
done

# Now download dependencies
echo "[5/6] Downloading dependencies..."
for req in "${REQ_FILES[@]}"; do
    echo "  -> Resolving dependencies for ${req}..."
    # Download with dependencies (remove --no-deps to get transitive deps)
    python3 -m pip download -r "${req}" -d "${OUTDIR}" || {
        echo "Warning: pip download returned non-zero for ${req}. Some packages may require building from sdist or manual intervention."
    }
done

# Optional: attempt to build wheels for any sdists found (pip wheel)
echo "[6/6] Building wheels from sdists..."
# Fix Docker socket path if needed
if [ -z "${DOCKER_HOST:-}" ] && [ ! -S "/var/run/docker.sock" ] && [ -S "/home/arcana-novai/.docker/desktop/docker.sock" ]; then
    export DOCKER_HOST="unix:///home/arcana-novai/.docker/desktop/docker.sock"
elif [ -z "${DOCKER_HOST:-}" ] && [ -S "/var/run/docker.sock" ]; then
    export DOCKER_HOST="unix:///var/run/docker.sock"
fi

# Temporarily disabled: Docker wheel building causes build freezes
# if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
#     echo "Docker detected — performing pip wheel build inside python:3.12-slim"
#     ... (Docker wheel building code removed to prevent freezes)
# else
    echo "Skipping Docker wheel building (disabled to prevent build freezes)"
# fi

# Log any sdists that couldn't be built
echo "Note: Source distributions (.tar.gz, .zip) found but not built into wheels"
find "${OUTDIR}" -type f \( -name "*.tar.gz" -o -name "*.zip" \) -exec basename {} \; | while read -r sdist; do
    echo "  - $sdist (requires manual wheel building if needed)"
done

# New: Auto-clean duplicates post-download and build
echo "Cleaning duplicates..."
./scripts/clean_wheelhouse_duplicates.sh "${OUTDIR}"

# Generate detailed manifest
echo "Generating detailed wheelhouse manifest..."
{
    echo "# Wheelhouse Manifest"
    echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo
    echo "## Package Summary"
    echo
    
    # List all wheels with their versions
    for wheel in "${OUTDIR}"/*.whl; do
        [[ -f "$wheel" ]] || continue
        base=$(basename "$wheel")
        echo "- ${base%.whl}"
    done
    
    echo
    echo "## Download Statistics"
    echo
    if command -v jq >/dev/null 2>&1 && [ -f "${MANIFEST}" ]; then
        jq -r '.downloads | length | "Total downloads: \(.)"' "${MANIFEST}" 2>/dev/null || echo "Total downloads: N/A"
        jq -r '.errors | length | "Total errors: \(.)"' "${MANIFEST}" 2>/dev/null || echo "Total errors: N/A"
        jq -r '.skipped | length | "Total skipped: \(.)"' "${MANIFEST}" 2>/dev/null || echo "Total skipped: N/A"
    else
        echo "Total downloads: N/A (jq not available or manifest missing)"
        echo "Total errors: N/A"
        echo "Total skipped: N/A"
    fi
    
    echo
    echo "## Error Log"
    echo
    if [[ -f "${OUTDIR}/build_errors.log" ]]; then
        cat "${OUTDIR}/build_errors.log"
    else
        echo "No build errors reported"
    fi
} > "${OUTDIR}/MANIFEST.md"

# Create compressed archive with manifest
echo "Creating compressed archive..."
tar -czf "${OUTDIR}.tgz" -C "$(dirname "${OUTDIR}")" "$(basename "${OUTDIR}")"

# Print summary
echo
echo "Wheelhouse Summary:"
echo "------------------"
echo "- Total packages: $(ls -1 "${OUTDIR}"/*.whl 2>/dev/null | wc -l)"
echo "- Archive size: $(du -h "${OUTDIR}.tgz" | cut -f1)"
echo "- Manifest: ${OUTDIR}/MANIFEST.md"
echo "- Download log: ${LOGFILE}"
echo "- Error log: ${OUTDIR}/build_errors.log"
echo
echo "To use this wheelhouse:"
echo "1. Copy '${OUTDIR}' or '${OUTDIR}.tgz' into your Docker build context"
echo "2. Build with OFFLINE=true"
echo "3. Check MANIFEST.md for any issues that need attention"
