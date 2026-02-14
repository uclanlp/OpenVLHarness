#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR"

OUTPUT_DIR="$ROOT_DIR/tests/output"
BACKUP_DIR="$OUTPUT_DIR/backup"
mkdir -p "$BACKUP_DIR"

shopt -s nullglob
for path in "$OUTPUT_DIR"/test_*; do
  mv "$path" "$BACKUP_DIR"/
done
shopt -u nullglob

before_snapshot="$(mktemp)"
find "$OUTPUT_DIR" -path "$BACKUP_DIR" -prune -o -type f -print | sort >"$before_snapshot"

TESTS=("$ROOT_DIR"/tests/test_*.py)
if [[ ${#TESTS[@]} -eq 0 || ! -e "${TESTS[0]}" ]]; then
  echo "No tests found in $ROOT_DIR/tests" >&2
  exit 1
fi

TEST_JOBS="${TEST_JOBS:-1}"
if ! [[ "$TEST_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEST_JOBS must be a positive integer, got: $TEST_JOBS" >&2
  exit 1
fi

pass_count=0
fail_count=0
total_count=${#TESTS[@]}

if (( TEST_JOBS > total_count )); then
  TEST_JOBS=$total_count
fi

echo "Running $total_count tests with $TEST_JOBS parallel job(s)."

declare -a running_pids=()
declare -A pid_to_output=()
declare -A pid_to_name=()
declare -A pid_to_progress=()

collect_result() {
  local pid="$1"
  local test_name="${pid_to_name[$pid]}"
  local progress="${pid_to_progress[$pid]}"
  local output_file="${pid_to_output[$pid]}"

  if wait "$pid"; then
    echo "PASS [$progress/$total_count]: $test_name"
    pass_count=$((pass_count + 1))
  else
    echo "FAIL [$progress/$total_count]: $test_name"
    cat "$output_file"
    fail_count=$((fail_count + 1))
  fi

  rm -f "$output_file"
  unset 'pid_to_output[$pid]'
  unset 'pid_to_name[$pid]'
  unset 'pid_to_progress[$pid]'
}

reap_finished_jobs() {
  local pid
  local -a still_running=()
  for pid in "${running_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_running+=("$pid")
      continue
    fi
    collect_result "$pid"
  done
  running_pids=("${still_running[@]}")
}

for idx in "${!TESTS[@]}"; do
  test_path="${TESTS[$idx]}"
  test_name="$(basename "$test_path")"
  progress=$((idx + 1))
  output_file="$(mktemp)"
  echo "RUN  [$progress/$total_count]: $test_name"
  python "$test_path" >"$output_file" 2>&1 &
  pid=$!
  running_pids+=("$pid")
  pid_to_output["$pid"]="$output_file"
  pid_to_name["$pid"]="$test_name"
  pid_to_progress["$pid"]="$progress"

  while (( ${#running_pids[@]} >= TEST_JOBS )); do
    reap_finished_jobs
    if (( ${#running_pids[@]} >= TEST_JOBS )); then
      sleep 0.1
    fi
  done
done

while (( ${#running_pids[@]} > 0 )); do
  reap_finished_jobs
  if (( ${#running_pids[@]} > 0 )); then
    sleep 0.1
  fi
done

echo "Summary: $pass_count passed, $fail_count failed."

after_snapshot="$(mktemp)"
find "$OUTPUT_DIR" -path "$BACKUP_DIR" -prune -o -type f -print | sort >"$after_snapshot"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$OUTPUT_DIR/tests_$timestamp"
mkdir -p "$run_dir"

comm -13 "$before_snapshot" "$after_snapshot" | while IFS= read -r path; do
  rel_path="${path#"$OUTPUT_DIR"/}"
  dest_path="$run_dir/$rel_path"
  mkdir -p "$(dirname "$dest_path")"
  mv "$path" "$dest_path"
done

rm -f "$before_snapshot" "$after_snapshot"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
