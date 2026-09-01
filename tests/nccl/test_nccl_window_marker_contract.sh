#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
tests_build="${1:-$repo_root/build/nccl-tests-p0-sm70}"
nccl_build="${2:-$repo_root/build/nccl-v2.31.2-sm70-sys}"
tmp_dir=$(mktemp -d -t nccl-window-marker.XXXXXX)

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

test -x "$tests_build/all_gather_perf" || fail "missing all_gather_perf: $tests_build"
test -f "$nccl_build/lib/libnccl.so.2.31.2" || fail "missing NCCL library: $nccl_build"

marker_file="$tmp_dir/window.jsonl"
log_file="$tmp_dir/nccl.log"
if ! env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NCCL_TESTS_WINDOW_FILE="$marker_file" \
  NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0 \
  LD_LIBRARY_PATH="$nccl_build/lib:/usr/local/cuda/lib64" \
  "$tests_build/all_gather_perf" -b 64K -e 64K -f 2 -g 4 -n 2 -w 1 \
  >"$log_file" 2>&1; then
  cat "$log_file" >&2
  fail "instrumented all_gather_perf failed"
fi

python3 - "$marker_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]

assert records, records
begins = [record for record in records if record["event"] == "measured_begin"]
ends = [record for record in records if record["event"] == "measured_end"]
assert begins, records
assert ends, records
assert min(record["timestampNs"] for record in begins) < max(
    record["timestampNs"] for record in ends
), records
assert all(record["rank"] == 0 for record in begins + ends), records
PY

aggressor_marker="$tmp_dir/aggressor-window.jsonl"
aggressor_ready="$tmp_dir/aggressor.ready"
aggressor_stop="$tmp_dir/aggressor.stop"
aggressor_log="$tmp_dir/aggressor.log"
aggressor_pid=""

cleanup_aggressor() {
  if [[ -n "$aggressor_pid" ]] && kill -0 "$aggressor_pid" 2>/dev/null; then
    kill -TERM "$aggressor_pid" 2>/dev/null || true
    wait "$aggressor_pid" 2>/dev/null || true
  fi
}

cleanup_all() {
  cleanup_aggressor
  rm -rf "$tmp_dir"
}
trap cleanup_all EXIT

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NCCL_TESTS_WINDOW_FILE="$aggressor_marker" \
  NCCL_TESTS_READY_FILE="$aggressor_ready" \
  NCCL_TESTS_STOP_FILE="$aggressor_stop" \
  NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0 \
  LD_LIBRARY_PATH="$nccl_build/lib:/usr/local/cuda/lib64" \
  "$tests_build/all_gather_perf" -b 64K -e 64K -f 2 -g 4 -n 1 -w 0 -N 0 \
  >"$aggressor_log" 2>&1 &
aggressor_pid=$!

ready_seen=0
for _ in $(seq 1 200); do
  if [[ -f "$aggressor_ready" ]]; then
    ready_seen=1
    break
  fi
  if ! kill -0 "$aggressor_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
[[ "$ready_seen" == 1 ]] || {
  cat "$aggressor_log" >&2
  fail "infinite nccl-tests aggressor did not signal READY"
}

touch "$aggressor_stop"
for _ in $(seq 1 200); do
  if ! kill -0 "$aggressor_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if kill -0 "$aggressor_pid" 2>/dev/null; then
  cat "$aggressor_log" >&2
  fail "infinite nccl-tests aggressor did not honor STOP"
fi
if ! wait "$aggressor_pid"; then
  cat "$aggressor_log" >&2
  fail "infinite nccl-tests aggressor exited with failure"
fi
aggressor_pid=""

python3 - "$aggressor_marker" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]

events = [record["event"] for record in records]
assert "ready" in events, records
assert "measured_begin" in events, records
assert "measured_end" in events, records
PY

echo "PASS: nccl-tests measured-window markers"
