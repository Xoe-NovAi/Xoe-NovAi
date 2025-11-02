#!/usr/bin/env bash
# build_docker.sh - Enhanced Docker build script with version management
# Guide Reference: Section 6.3 (Docker Builds)
# Last Updated: 2025-11-02 (Added OFFLINE detection, integrity verify)
# Ryzen Opt: Implicit via ENV (N_THREADS=6); Telemetry: Disabled in Dockerfiles
set -euo pipefail

# Setup logging
LOGDIR="logs/docker_build"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/build_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "${LOGFILE}")
exec 2> >(tee -a "${LOGFILE}" >&2)

# Load .env for OFFLINE_BUILD (fixed grep for indented comments)
if [ -f .env ]; then
    # Filter lines with leading whitespace + # (comments)
    export $(grep -v '^[[:space:]]*#' .env | xargs -d '\n')
fi

# Set OFFLINE arg if enabled
OFFLINE_ARG=""
if [ "${OFFLINE_BUILD:-false}" = "true" ]; then
    OFFLINE_ARG="--build-arg OFFLINE=true"
    echo "Offline build mode enabled (OFFLINE_BUILD=true)"
fi

# Load versions from versions.toml
if [[ ! -f versions/versions.toml ]]; then
    echo "Error: versions/versions.toml not found"
    exit 1
fi

echo "Starting Docker build process at $(date)"
echo "Loading versions from versions.toml..."

# Extract versions using Python
VERSIONS=$(python3 -c '
import toml
import json
with open("versions/versions.toml") as f:
    versions = toml.load(f)
print(json.dumps(versions.get("versions", {})))
')

# Build common build arguments from versions
BUILD_ARGS=""
for key in $(echo "$VERSIONS" | jq -r 'keys[]'); do
    value=$(echo "$VERSIONS" | jq -r --arg key "$key" '.[$key]')
    BUILD_ARGS+=" --build-arg ${key^^}_VERSION=$value"
done

# Clean up old containers and images
echo "Cleaning up old containers and images..."
docker system prune -f --volumes

# Function to build a service
build_service() {
    local service=$1
    local dockerfile=$2
    echo "Building $service from $dockerfile..."
    
    # Create build context
    echo "Preparing build context..."
    local ctx="build_context_${service}"
    rm -rf "$ctx"
    mkdir -p "$ctx"
    
    # Copy required files to build context
    cp -r app "$ctx/"
    cp -r versions "$ctx/"
    cp -r requirements-*.txt "$ctx/"
    if [ -f wheelhouse.tgz ]; then
        cp wheelhouse.tgz "$ctx/"
    else
        cp -r wheelhouse "$ctx/"
    fi
    cp "$dockerfile" "$ctx/Dockerfile"
    
    # Build the image with proper context
    DOCKER_BUILDKIT=1 docker build \
        $BUILD_ARGS \
        $OFFLINE_ARG \
        --progress=plain \
        --tag "xnai-${service}:latest" \
        -f "$ctx/Dockerfile" \
        "$ctx" 2>&1 | tee "${LOGDIR}/build_${service}.log"
        
    # Clean up
    rm -rf "$ctx"
}

# Update requirements from versions.toml
echo "Updating requirements files..."
python3 versions/scripts/update_versions.py

# Ensure wheelhouse is up to date
echo "Updating wheelhouse..."
./scripts/download_wheelhouse.sh

# Clean wheelhouse duplicates
echo "Cleaning wheelhouse..."
./scripts/clean_wheelhouse_duplicates.sh

# New: Verify wheelhouse integrity
echo "Verifying wheelhouse integrity..."
python3 scripts/build_tools/dependency_tracker.py --verify || {
    echo "Error: Wheel integrity verification failed"
    exit 1
}

# Build each service
echo "Building services..."
build_service "api" "Dockerfile.api"
build_service "chainlit" "Dockerfile.chainlit"
build_service "crawl" "Dockerfile.crawl"
build_service "curation" "Dockerfile.curation_worker"

echo "Build process complete at $(date)"
echo "Build logs available in: ${LOGDIR}"

# Generate build report
{
    echo "# Docker Build Report"
    echo "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo
    echo "## Build Arguments"
    echo "$(echo "$BUILD_ARGS" | tr ' ' '\n' | sort)"
    echo
    echo "## Service Status"
    echo
    docker images "xnai-*" --format "- {{.Repository}}: {{.Size}}"
    echo
    echo "## Build Logs"
    echo
    for log in "${LOGDIR}"/build_*.log; do
        service=$(basename "$log" .log | cut -d_ -f2)
        echo "### $service"
        echo
        tail -n 20 "$log" | sed 's/^/    /'
        echo
    done
} > "${LOGDIR}/build_report.md"

echo "Build report available at: ${LOGDIR}/build_report.md"