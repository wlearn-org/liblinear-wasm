#!/bin/bash
set -euo pipefail

# Build browser-ready IIFE + ESM bundles using esbuild
# Generic: auto-infers exports from src/index.js module.exports
#
# Requires: WASM built with -sSINGLE_FILE=1 -sSINGLE_FILE_BINARY_ENCODE=0
# so the base64-encoded WASM survives bundler string re-encoding.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

# Read package name from package.json, derive global name
# e.g. @wlearn/liblinear -> wlearnLiblinear
PKG_NAME=$(node -e "
  const p = require('${PROJECT_DIR}/package.json')
  const name = p.name.replace(/^@/, '').replace(/[/-](\w)/g, (_, c) => c.toUpperCase())
  console.log(name)
")

# Auto-infer export names from module.exports in src/index.js
EXPORTS=$(node -e "
  const m = require('${PROJECT_DIR}/src/index.js')
  console.log(Object.keys(m).join(','))
")

echo "=== Building browser bundles ==="
echo "  Package: ${PKG_NAME}"
echo "  Exports: ${EXPORTS}"

mkdir -p "$DIST_DIR"

# Create empty stub module for Node built-ins referenced by Emscripten glue
if [ ! -f "${PROJECT_DIR}/scripts/empty.js" ]; then
  echo "module.exports = {}" > "${PROJECT_DIR}/scripts/empty.js"
fi

# Common esbuild flags
COMMON_FLAGS=(
  --bundle
  --platform=browser
  --minify
  --alias:node:fs=./scripts/empty.js
  --alias:node:crypto=./scripts/empty.js
  --alias:node:path=./scripts/empty.js
  --define:__dirname='""'
  --define:__filename='""'
)

# IIFE bundle (browser global, for <script> tags)
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${PKG_NAME}" \
  --outfile="${DIST_DIR}/${PKG_NAME}.iife.js"

# ESM bundle (IIFE with private global + appended named exports)
INTERNAL="__${PKG_NAME}"
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${INTERNAL}" \
  --outfile="${DIST_DIR}/${PKG_NAME}.esm.mjs"

# Append named ESM exports
IFS=',' read -ra KEYS <<< "$EXPORTS"
DESTRUCTURE=$(IFS=','; echo "${KEYS[*]}")
EXPORT_LINE=$(IFS=','; echo "${KEYS[*]}")
echo "var {${DESTRUCTURE}}=${INTERNAL};export{${EXPORT_LINE}};" >> "${DIST_DIR}/${PKG_NAME}.esm.mjs"

echo "=== Browser bundles built ==="
ls -lh "${DIST_DIR}/${PKG_NAME}.iife.js" "${DIST_DIR}/${PKG_NAME}.esm.mjs"
