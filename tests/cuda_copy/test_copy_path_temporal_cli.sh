#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_copy_path_temporal_ablation.sh"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

"${runner}" --help >/dev/null
if "${runner}" --deviceList=0,1,2,3 --runs=1 --repeats=2 \
    --outputRoot="${temp_root}/invalid" >/dev/null 2>&1; then
  echo "runner accepted an invalid four-GPU list" >&2
  exit 1
fi

output_root="${temp_root}/smoke"
"${runner}" \
  --deviceList=0,1,2 \
  --backgroundPaths=none,original-d2h,streaming-hbm-read \
  --pressureLevels=current-pulsed,duty-0.1 \
  --topologies=single-two-copy \
  --runs=1 --repeats=2 --size=1M --backgroundSize=1M \
  --outputRoot="${output_root}" >/dev/null

[[ -s "${output_root}/environment.txt" ]]
[[ -s "${output_root}/summary.csv" ]]
rg -q 'single-two-copy means one source stream with two consecutive copies' \
  "${output_root}/environment.txt"
rg -q 'pressureLevel,backgroundPath,victimMode,topology' \
  "${output_root}/summary.csv"
[[ "$(wc -l < "${output_root}/summary.csv")" -eq 7 ]]
[[ "$(rg -c ',pass,' "${output_root}/summary.csv")" -eq 6 ]]
rg -q '^current-pulsed,none,original-p2p-ce,single-two-copy,1,' \
  "${output_root}/summary.csv"
rg -q ',4\.0,1\.0,(bandwidth-matched|duty-matched|control),' \
  "${output_root}/summary.csv"
rg -q '^duty-0.1,streaming-hbm-read,original-p2p-ce,single-two-copy,1,"0,1,2",1M,1M,0,0.1,duty-matched,' \
  "${output_root}/summary.csv"

trace_root="${temp_root}/trace"
"${runner}" \
  --deviceList=0,1,2 \
  --backgroundPaths=original-d2h \
  --pressureLevels=current-pulsed \
  --topologies=single-two-copy \
  --runs=1 --repeats=2 --size=1M --backgroundSize=1M --trace=1 \
  --outputRoot="${trace_root}" >/dev/null
trace_case="${trace_root}/pressure-current-pulsed/background-original-d2h/topology-single-two-copy/repetition-1"
[[ -s "${trace_case}/victim.memcpy.csv" ]]
[[ -s "${trace_case}/background.memcpy.csv" ]]
[[ "$(wc -l < "${trace_case}/victim.memcpy.csv")" -gt 1 ]]
[[ "$(wc -l < "${trace_case}/background.memcpy.csv")" -gt 1 ]]
rg -F -q ",${trace_case}/victim.memcpy.csv" "${trace_root}/summary.csv"

echo "PASS: temporal ablation runner CLI contract"
