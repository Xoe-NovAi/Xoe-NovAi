#!/usr/bin/env bash
# scripts/download_wheelhouse.sh
# Downloads wheels for your project's requirements files into wheelhouse/
# Usage: ./scripts/download_wheelhouse.sh [OUTDIR] [REQ_GLOB]
# Example: ./scripts/download_wheelhouse.sh wheelhouse "requirements-*.txt"
set -euo pipefail

OUTDIR=${1:-wheelhouse}
REQ_GLOB=${2:-"requirements-*.txt"}

echo "Creating wheelhouse in: ${OUTDIR}"
rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"

# Always include pip, setuptools, wheel and build dependencies
echo "[1/4] Downloading core build dependencies..."
python3 -m pip download --only-binary=:all: pip setuptools wheel scikit-build-core -d "${OUTDIR}" || {
  echo "Warning: core dependencies download returned non-zero; continuing."
}

# Download build requirements for llama-cpp-python
echo "[2/4] Downloading llama-cpp-python build requirements..."
python3 -m pip download --only-binary=:all: cmake ninja -d "${OUTDIR}" || {
  echo "Warning: llama-cpp build dependencies download returned non-zero; continuing."
}

# Resolve each matching requirements file and download packages
echo "[2/4] Scanning for requirements files to download:"
shopt -s nullglob
REQ_FILES=( $REQ_GLOB )
shopt -u nullglob

if [ ${#REQ_FILES[@]} -eq 0 ]; then
  echo "No requirements files matched pattern '${REQ_GLOB}'. Exiting with error."
  exit 2
fi

for req in "${REQ_FILES[@]}"; do
  echo "  -> Downloading from ${req}..."
  # Use --no-deps only if you intentionally manage transitive deps yourself.
  # Here we fetch transitive deps so offline install is self-contained.
  python3 -m pip download -r "${req}" -d "${OUTDIR}" || {
    echo "Warning: pip download returned non-zero for ${req}. Some packages may require building from sdist or manual intervention."
  }
done

# Optional: attempt to build wheels for any sdists found (pip wheel)
echo "[3/4] Attempting to build wheels for any sdists (best-effort)..."
# Create a temporary build container environment if 'docker' available to match linux glibc
if command -v docker >/dev/null 2>&1; then
  echo "Docker detected — performing pip wheel build inside python:3.12-slim to convert sdists -> wheels where possible."
  docker run --rm -v "$(pwd)/${OUTDIR}":/wheels -w /wheels python:3.12-slim bash -lc "\
    apt-get update > /dev/null && apt-get install -y --no-install-recommends build-essential cmake git libopenblas-dev pkg-config python3-dev >/dev/null 2>&1 || true; \
    set -x; \
    for s in *.tar.gz *.zip; do \
      [ -f \"\$s\" ] || continue; \
      python -m pip wheel --no-deps --wheel-dir=/wheels \"\$s\" || true; \
    done; \
    true"
else
  echo "Docker not available on this host — skipping automatic wheel builds for sdists. You can build them later using a Debian/manylinux builder."
fi

# List summary and compress
echo "[4/4] Wheelhouse summary:"
ls -1 "${OUTDIR}" | sed -n '1,200p' || true
tar -czf "${OUTDIR}.tgz" -C "$(dirname "${OUTDIR}")" "$(basename "${OUTDIR}")"
echo "Created: ${OUTDIR} and ${OUTDIR}.tgz"

echo "Done. Copy '${OUTDIR}' or '${OUTDIR}.tgz' into your Docker build context and build with OFFLINE=true."
