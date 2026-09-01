#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
host_bin="${repo_root}/build/cuda_copy/host_copy_background"
path_bin="${repo_root}/build/cuda_copy/copy_path_background"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

run_case() {
  local name="$1"
  local binary="$2"
  shift 2
  local ready_file="${temp_root}/${name}.ready"
  local stop_file="${temp_root}/${name}.stop"
  local output_file="${temp_root}/${name}.json"
  local log_file="${temp_root}/${name}.log"

  "${binary}" "$@" \
    --readyFile="${ready_file}" \
    --stopFile="${stop_file}" \
    --output="${output_file}" >"${log_file}" 2>&1 &
  local pid=$!
  for _ in $(seq 1 100); do
    [[ -e "${ready_file}" ]] && break
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      echo "background ${name} exited before ready" >&2
      cat "${log_file}" >&2
      return 1
    fi
    sleep 0.05
  done
  [[ -e "${ready_file}" ]] || {
    echo "background ${name} did not become ready" >&2
    kill -INT "${pid}" 2>/dev/null || true
    wait "${pid}" || true
    cat "${log_file}" >&2
    return 1
  }
  sleep 0.25
  touch "${stop_file}"
  kill -INT "${pid}" 2>/dev/null || true
  wait "${pid}"

  python3 - "${output_file}" "${name}" <<'PY'
import json
import sys

path, name = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)

devices = data.get("devices")
assert devices == [0, 1, 2], (name, devices)
bytes_values = data["perDeviceBytes"]
operations = data["perDeviceOperations"]
assert len(bytes_values) == len(operations) == len(devices), name
assert all(value > 0 for value in bytes_values), (name, bytes_values)
assert all(value > 0 for value in operations), (name, operations)
assert sum(bytes_values) == data["totalBytes"], name

for field in ("perDeviceWallActiveSec", "perDeviceWallActiveDuty",
              "perDeviceGpuActivitySec", "perDeviceGpuActivityDuty"):
    values = data[field]
    assert len(values) == len(devices), (name, field, values)

wall_duty = data["perDeviceWallActiveDuty"]
assert all(0.0 <= value <= 1.0 for value in wall_duty), (name, wall_duty)

for field in ("perDeviceOperationDurationMs", "perDeviceSubmitIntervalMs",
              "perDeviceIdleGapMs"):
    values = data[field]
    assert len(values) == len(devices), (name, field, values)
    for stats in values:
        for key in ("count", "p50", "p90", "p99", "max"):
            assert key in stats, (name, field, stats)
        assert stats["count"] > 0, (name, field, stats)
        assert stats["p50"] > 0, (name, field, stats)

print(f"validated temporal JSON: {name}")
PY
}

[[ -x "${host_bin}" ]] || { echo "missing ${host_bin}" >&2; exit 1; }
[[ -x "${path_bin}" ]] || { echo "missing ${path_bin}" >&2; exit 1; }

run_case host "${host_bin}" --devList=0,1,2 --direction=d2h --size=1M
run_case hbm "${path_bin}" --deviceList=0,1,2 --path=streaming-hbm-read \
  --size=1M --targetGBps=4.0 --dutyCycle=1.0

echo "PASS: temporal background metrics JSON contract"
