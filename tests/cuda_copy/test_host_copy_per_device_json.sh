#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
tmp_dir="$(mktemp -d -t cuda-copy-per-device-json.XXXXXX)"
background_pid=""

cleanup() {
  if [[ -n "${background_pid}" ]] && kill -0 "${background_pid}" 2>/dev/null; then
    kill -INT "${background_pid}" 2>/dev/null || true
    wait "${background_pid}" 2>/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${background_bin}" ]] || fail "missing executable: ${background_bin}"

ready_file="${tmp_dir}/ready"
stop_file="${tmp_dir}/stop"
json_file="${tmp_dir}/background.json"
log_file="${tmp_dir}/background.log"
"${background_bin}" --devList=0,1,2 --direction=d2h --size=1M \
  --readyFile="${ready_file}" --stopFile="${stop_file}" \
  --reportSec=1 --output="${json_file}" >"${log_file}" 2>&1 &
background_pid=$!

for _ in $(seq 1 100); do
  [[ -f "${ready_file}" ]] && break
  if ! kill -0 "${background_pid}" 2>/dev/null; then
    cat "${log_file}" >&2
    fail "background process exited before ready"
  fi
  sleep 0.1
done
[[ -f "${ready_file}" ]] || { cat "${log_file}" >&2; fail "background did not become ready"; }
touch "${stop_file}"
wait "${background_pid}" || fail "background process failed"
background_pid=""

python3 - "${json_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)

bytes_per_device = result["perDeviceBytes"]
gbps_per_device = result["perDeviceGBps"]
assert len(bytes_per_device) == 3, bytes_per_device
assert len(gbps_per_device) == 3, gbps_per_device
assert all(value > 0 for value in bytes_per_device), bytes_per_device
assert all(value > 0 for value in gbps_per_device), gbps_per_device
assert sum(bytes_per_device) == result["totalBytes"], (bytes_per_device, result)
PY

echo "PASS: host copy per-device JSON accounting"

