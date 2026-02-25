#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GLUE="${1:-${PROJECT_DIR}/wasm/linear.cjs}"

if [ ! -f "$GLUE" ]; then
  echo "ERROR: glue file not found: $GLUE"
  exit 1
fi

REQUIRED_SYMBOLS=(
  wl_linear_get_last_error
  wl_linear_train
  wl_linear_predict
  wl_linear_predict_values
  wl_linear_predict_probability
  wl_linear_save_model
  wl_linear_load_model
  wl_linear_free_model
  wl_linear_free_buffer
  wl_linear_get_nr_class
  wl_linear_get_nr_feature
  wl_linear_get_labels
  wl_linear_get_bias
)

missing=0
for fn in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "_${fn}" "$GLUE"; then
    echo "MISSING: ${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} symbol(s) missing"
  exit 1
fi

echo "All ${#REQUIRED_SYMBOLS[@]} exports verified"
