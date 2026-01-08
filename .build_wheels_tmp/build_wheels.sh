#!/bin/bash
set -euo pipefail

# Setup error logging
ERRLOG="/wheels/build_errors.log"
touch "$ERRLOG"

# Install build dependencies
echo "Installing build dependencies..."
apt-get update > /dev/null
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    pkg-config \
    python3-dev \
    >/dev/null 2>&1 || echo "Warning: Some build dependencies failed to install"

# Process each sdist
echo "Processing sdists..."
for s in /build/*.tar.gz /build/*.zip; do
    [ -f "$s" ] || continue
    echo "Building wheel for $(basename $s)..."
    if python -m pip wheel --no-deps --wheel-dir=/wheels "$s" 2>>"$ERRLOG"; then
        echo "Successfully built wheel for $(basename $s)"
    else
        echo "Failed to build wheel for $(basename $s) (see build_errors.log)"
    fi
done
