#!/usr/bin/env bash
# Build the optional C++ device-decode binary (for scripts/run_device_decode.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/device_decode/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
cmake --build . -j"$(nproc 2>/dev/null || echo 4)"
echo "Built: ${BUILD_DIR}/df11_device_decode"
