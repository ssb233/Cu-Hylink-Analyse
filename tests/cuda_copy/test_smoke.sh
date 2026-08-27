#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
d2d_bin="${repo_root}/build/cuda_copy/d2d_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
tmp_dir="$(mktemp -d -t cuda-copy-smoke.XXXXXX)"
background_pid=""

cleanup() {
  if [[ -n "${background_pid}" ]] && kill -0 "${background_pid}" 2>/dev/null; then
    kill -INT "${background_pid}" 2>/dev/null || true
    wait "${background_pid}" 2>/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

[[ -x "${d2d_bin}" ]] || { echo "FAIL: missing ${d2d_bin}" >&2; exit 1; }
[[ -x "${background_bin}" ]] || { echo "FAIL: missing ${background_bin}" >&2; exit 1; }

one_way_log="${tmp_dir}/one-way.log"
"${d2d_bin}" --mode=unidirectional --srcDev=0 --dstDev=1 \
  --repeats=1 --size=1M --output="${tmp_dir}/one-way.json" >"${one_way_log}"
grep -F -- "mode=unidirectional" "${one_way_log}" >/dev/null
grep -F -- "warmup=10" "${one_way_log}" >/dev/null
grep -F -- "repeats=1" "${one_way_log}" >/dev/null
grep -F -- "bytes=1048576" "${one_way_log}" >/dev/null

two_way_log="${tmp_dir}/two-way.log"
"${d2d_bin}" --mode=bidirectional --srcDev=0 --dstDev=1 \
  --repeats=1 --size=1M --output="${tmp_dir}/two-way.json" >"${two_way_log}"
grep -F -- "mode=bidirectional" "${two_way_log}" >/dev/null
grep -F -- "directions=2" "${two_way_log}" >/dev/null

ready_file="${tmp_dir}/ready"
stop_file="${tmp_dir}/stop"
background_log="${tmp_dir}/background.log"
"${background_bin}" --devList=0,1 --direction=d2h --size=1M \
  --readyFile="${ready_file}" --stopFile="${stop_file}" \
  --reportSec=1 --output="${tmp_dir}/background.json" >"${background_log}" 2>&1 &
background_pid=$!

for _ in $(seq 1 100); do
  [[ -f "${ready_file}" ]] && break
  if ! kill -0 "${background_pid}" 2>/dev/null; then
    cat "${background_log}" >&2
    exit 1
  fi
  sleep 0.1
done
[[ -f "${ready_file}" ]] || { cat "${background_log}" >&2; exit 1; }
touch "${stop_file}"
wait "${background_pid}"
background_pid=""
grep -F -- "direction=d2h" "${background_log}" >/dev/null
grep -F -- "devices=0,1" "${background_log}" >/dev/null

echo "PASS: CUDA copy smoke tests"
