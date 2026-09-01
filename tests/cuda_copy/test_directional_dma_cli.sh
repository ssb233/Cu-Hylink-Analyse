#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_directional_dma_diagnostic.sh"
victim_bin="${repo_root}/build/cuda_copy/minimal_source_pair_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
temp_root="$(mktemp -d -t directional-dma-cli.XXXXXX)"
background_pid=""

cleanup() {
  if [[ -n "${background_pid}" ]] && kill -0 "${background_pid}" 2>/dev/null; then
    kill -INT "${background_pid}" 2>/dev/null || true
    wait "${background_pid}" 2>/dev/null || true
  fi
  rm -rf "${temp_root}"
}
trap cleanup EXIT

[[ -x "${runner}" ]] || { echo "missing ${runner}" >&2; exit 1; }
help_text="$(${runner} --help)"
grep -F -- '--victimSize=255M' <<<"${help_text}" >/dev/null
grep -F -- '--backgroundSizes=64K,128K,256K,512K,1M,2M,4M,8M,16M,255M' <<<"${help_text}" >/dev/null
grep -F -- '--dutyCycles=1.0' <<<"${help_text}" >/dev/null
grep -F -- '--directions=d2h,h2d' <<<"${help_text}" >/dev/null
grep -F -- '--backgroundSets=none,0,1,2,all' <<<"${help_text}" >/dev/null
grep -F -- '--topologies=shared,independent,independent+source-chain' <<<"${help_text}" >/dev/null

[[ -x "${victim_bin}" ]] || { echo "missing ${victim_bin}" >&2; exit 1; }
victim_json="${temp_root}/victim.json"
"${victim_bin}" --deviceList=0,1,2 --edgeOrder='0->1,0->2' \
  --streamMode=shared --streamDependency=none --repeats=2 --size=1M \
  --diagnostic=0 --output="${victim_json}" >/dev/null
python3 - "${victim_json}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
assert data["diagnostic"] is False
assert data["bytesPerMemcpy"] == 1024 * 1024
assert all(row["operationDurationMs"] is None for row in data["edgeResults"])
PY

output_root="${temp_root}/runner"
"${runner}" \
  --deviceList=0,1,2 \
  --backgroundSets=none,0 \
  --backgroundSizes=1M \
  --directions=d2h \
  --dutyCycles=1.0 \
  --victimSize=1M \
  --edgeOrders=forward,reverse \
  --topologies=shared,independent+source-chain \
  --runs=1 --repeats=2 \
  --outputRoot="${output_root}" >/dev/null

[[ -s "${output_root}/summary.csv" ]]
[[ "$(wc -l < "${output_root}/summary.csv")" -eq 9 ]]
[[ "$(rg -c ',pass,' "${output_root}/summary.csv")" -eq 8 ]]
rg -q 'backgroundSize,victimSize' "${output_root}/summary.csv"
rg -q '^1M,1M,none,NA,none,forward,shared,1,"0,1,2",' "${output_root}/summary.csv"
rg -q '^1M,1M,d2h,1\.0,0,forward,shared,1,"0,1,2",' "${output_root}/summary.csv"

[[ -x "${background_bin}" ]] || { echo "missing ${background_bin}" >&2; exit 1; }
ready_file="${temp_root}/ready"
stop_file="${temp_root}/stop"
background_json="${temp_root}/background.json"
"${background_bin}" --devList=0 --direction=d2h --size=1M --dutyCycle=0.5 \
  --readyFile="${ready_file}" --stopFile="${stop_file}" \
  --reportSec=1 --output="${background_json}" >"${temp_root}/background.log" 2>&1 &
background_pid=$!
for _ in $(seq 1 100); do
  [[ -e "${ready_file}" ]] && break
  if ! kill -0 "${background_pid}" 2>/dev/null; then
    cat "${temp_root}/background.log" >&2
    exit 1
  fi
  sleep 0.05
done
[[ -e "${ready_file}" ]] || { cat "${temp_root}/background.log" >&2; exit 1; }
sleep 0.4
touch "${stop_file}"
kill -INT "${background_pid}" 2>/dev/null || true
wait "${background_pid}"
background_pid=""
python3 - "${background_json}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
assert data["dutyCycleTarget"] == 0.5
assert data["perDeviceOperationDurationMs"][0]["count"] > 0
assert data["perDeviceIdleGapMs"][0]["count"] > 0
PY

echo "PASS: directional DMA Stage I CLI contract"
