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

# Read package name from package.json
# Browser global + bundle base name: @wlearn/liblinear -> liblinear
NAME=$(node -e "
  const p = require('${PROJECT_DIR}/package.json')
  console.log(p.name.split('/').pop())
")

# Auto-infer export names from the final module.exports object in src/index.js
# Do not require() the module here: prepack must work from a clean checkout
# without needing runtime deps like @wlearn/core to be installed.
EXPORTS=$(node -e "
  const fs = require('fs')
  const path = require('path')
  const src = fs.readFileSync(path.join('${PROJECT_DIR}', 'src', 'index.js'), 'utf8')
  const noComments = src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
  const match = noComments.match(/module\.exports\s*=\s*\{([\s\S]*?)\}/m)
  if (!match) throw new Error('Could not find module.exports object in src/index.js')
  const exportsText = match[1]
  const names = exportsText
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => s.split(':')[0].trim())
  console.log(names.join(','))
")

echo "=== Building browser bundles ==="
echo "  Package: ${NAME}"
echo "  Files: ${NAME}.js, ${NAME}.mjs"
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
  --alias:ws=./scripts/empty.js
  --define:__dirname='""'
  --define:__filename='""'
)

# IIFE bundle (browser global, for <script> tags)
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${NAME}" \
  --outfile="${DIST_DIR}/${NAME}.js"

# ESM bundle (IIFE with private global + appended named exports)
INTERNAL="__${NAME}"
npx esbuild "${PROJECT_DIR}/src/index.js" \
  "${COMMON_FLAGS[@]}" \
  --format=iife \
  --global-name="${INTERNAL}" \
  --outfile="${DIST_DIR}/${NAME}.mjs"

# Append named ESM exports
IFS=',' read -ra KEYS <<< "$EXPORTS"
DESTRUCTURE=$(IFS=','; echo "${KEYS[*]}")
EXPORT_LINE=$(IFS=','; echo "${KEYS[*]}")
echo "var {${DESTRUCTURE}}=${INTERNAL};export{${EXPORT_LINE}};" >> "${DIST_DIR}/${NAME}.mjs"

echo "=== Browser bundles built ==="
ls -lh "${DIST_DIR}/${NAME}.js" "${DIST_DIR}/${NAME}.mjs"
