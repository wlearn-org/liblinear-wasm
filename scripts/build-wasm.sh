#!/bin/bash
set -euo pipefail

# Build LIBLINEAR v2.50 as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/liblinear"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/linear.h" ]; then
  echo "ERROR: LIBLINEAR upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null 2>&1; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "$UPSTREAM_DIR" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

# LIBLINEAR is small: just a few source files, no CMake needed
EXPORTED_FUNCTIONS='["_wl_linear_get_last_error","_wl_linear_train","_wl_linear_predict","_wl_linear_predict_values","_wl_linear_predict_probability","_wl_linear_save_model","_wl_linear_load_model","_wl_linear_free_model","_wl_linear_free_buffer","_wl_linear_get_nr_class","_wl_linear_get_nr_feature","_wl_linear_get_labels","_wl_linear_get_bias","_malloc","_free"]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","UTF8ToString"]'

# Find BLAS source files
BLAS_SOURCES=""
if [ -d "${UPSTREAM_DIR}/blas" ]; then
  BLAS_SOURCES=$(find "${UPSTREAM_DIR}/blas" -name '*.c' | tr '\n' ' ')
fi

emcc \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  "${UPSTREAM_DIR}/linear.cpp" \
  "${UPSTREAM_DIR}/newton.cpp" \
  $BLAS_SOURCES \
  -I "${UPSTREAM_DIR}" \
  -I "${UPSTREAM_DIR}/blas" \
  -o "${OUTPUT_DIR}/linear.js" \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createLinear \
  -s FORCE_FILESYSTEM=1 \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: liblinear v2.50
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/linear.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
