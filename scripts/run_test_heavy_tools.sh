#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

TESTS=(
  "$ROOT_DIR/tests/test_grounding_dino_tool.py"
  "$ROOT_DIR/tests/test_depth_tool.py"
  "$ROOT_DIR/tests/test_ocr_tool.py"
  "$ROOT_DIR/tests/test_sam3_segment.py"
)

for test_path in "${TESTS[@]}"; do
  if [[ ! -f "$test_path" ]]; then
    echo "Missing test: $test_path" >&2
    exit 1
  fi
done

pass_count=0
fail_count=0
total_count=${#TESTS[@]}

echo "Running $total_count heavy tool tests."

for idx in "${!TESTS[@]}"; do
  test_path="${TESTS[$idx]}"
  test_name="$(basename "$test_path")"
  progress=$((idx + 1))
  output_file="$(mktemp)"
  echo "RUN  [$progress/$total_count]: $test_name"
  if python "$test_path" >"$output_file" 2>&1; then
    echo "PASS [$progress/$total_count]: $test_name"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL [$progress/$total_count]: $test_name"
    cat "$output_file"
    fail_count=$((fail_count + 1))
  fi
  rm -f "$output_file"
done

echo "Summary: $pass_count passed, $fail_count failed."

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
