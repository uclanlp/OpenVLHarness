#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"

TESTS=("$ROOT_DIR"/tests/test_*.py)
if [[ ${#TESTS[@]} -eq 0 || ! -e "${TESTS[0]}" ]]; then
  echo "No tests found in $ROOT_DIR/tests" >&2
  exit 1
fi

pass_count=0
fail_count=0

for test_path in "${TESTS[@]}"; do
  test_name="$(basename "$test_path")"
  output_file="$(mktemp)"
  if python "$test_path" >"$output_file" 2>&1; then
    echo "PASS: $test_name"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL: $test_name"
    cat "$output_file"
    fail_count=$((fail_count + 1))
  fi
  rm -f "$output_file"
done

echo "Summary: $pass_count passed, $fail_count failed."

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
