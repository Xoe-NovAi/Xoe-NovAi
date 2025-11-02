#!/usr/bin/env bash
# Orchestrates Docker builds for all services, ensuring wheelhouse is present in context
set -euo pipefail
SERVICES=(api chainlit crawl curation_worker)
ROOT_DIR="$(pwd)"
for svc in "${SERVICES[@]}"; do
    CONTEXT="build_context_${svc}"
    mkdir -p "$CONTEXT"
    cp -r app "$CONTEXT/"
    cp requirements-${svc}.txt "$CONTEXT/"
    cp -r wheelhouse "$CONTEXT/"
    cp wheelhouse.tgz "$CONTEXT/" || true
    cp Dockerfile.${svc} "$CONTEXT/Dockerfile"
    echo "Building $svc with context $CONTEXT..."
    docker build -f "$CONTEXT/Dockerfile" "$CONTEXT" || {
        echo "ERROR: Build failed for $svc. Check logs.";
        exit 2;
    }
done
