#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
victim_bin="${repo_root}/build/cuda_copy/minimal_source_pair_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
trace_lib="${repo_root}/build/cuda_copy/libcupti_memcpy_channel_trace.so"

device_list="0,1,2"
background_sets="none,0"
background_sizes="64K,128K,256K,512K,1M,2M,4M,8M,16M,255M"
directions="d2h"
duty_cycles="1.0"
victim_size="255M"
edge_orders="forward,reverse"
topologies="shared,independent,independent+source-chain"
runs=3
repeats=300
diagnostic=0
trace_enabled=0
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_directional_dma_diagnostic.sh [options]

Run Stage I source-local P2P victim experiments with independently controlled host-copy
payload, direction, duty/cadence, background device set, edge order and topology.

Options:
  --deviceList=0,1,2
  --backgroundSets=none,0,1,2,all
  --backgroundSizes=64K,128K,256K,512K,1M,2M,4M,8M,16M,255M
  --directions=d2h,h2d
  --dutyCycles=1.0
  --victimSize=255M
  --edgeOrders=forward,reverse
  --topologies=shared,independent,independent+source-chain
  --runs=3                  repetitions per matrix point
  --repeats=300             measured copies per edge
  --diagnostic=0|1          record per-operation victim CUDA events
  --trace=0|1               preload the CUPTI memcpy tracer
  --outputRoot=<path>       fresh result directory
  --help

Warmup is fixed at 10. Stage I performance mode defaults to diagnostic=0, so only
batch start/stop events are used. A clean row is recorded once per size/topology/order
and is reused by the analyzer for all direction/duty/background comparisons.
EOF
}

stop_background() {
  background_rc=0
  if [[ -z "${background_pid}" ]]; then return 0; fi
  if [[ -n "${background_stop}" ]]; then touch "${background_stop}"; fi
  if kill -0 "${background_pid}" 2>/dev/null; then
    kill -INT "${background_pid}" 2>/dev/null || true
  fi
  wait "${background_pid}" || background_rc=$?
  background_pid=""
  background_stop=""
}

on_exit() { stop_background; }
on_signal() { stop_background; exit 130; }
trap on_exit EXIT
trap on_signal INT TERM

while [[ $# -gt 0 ]]; do
  argument="$1"
  case "${argument}" in
    --help|-h) usage; exit 0 ;;
    --deviceList=*) device_list="${argument#*=}"; shift ;;
    --deviceList)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      device_list="$2"; shift 2 ;;
    --backgroundSets=*) background_sets="${argument#*=}"; shift ;;
    --backgroundSets)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_sets="$2"; shift 2 ;;
    --backgroundSizes=*) background_sizes="${argument#*=}"; shift ;;
    --backgroundSizes|--backgroundSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_sizes="$2"; shift 2 ;;
    --directions=*) directions="${argument#*=}"; shift ;;
    --directions|--direction)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      directions="$2"; shift 2 ;;
    --dutyCycles=*) duty_cycles="${argument#*=}"; shift ;;
    --dutyCycles|--dutyCycle)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      duty_cycles="$2"; shift 2 ;;
    --victimSize=*) victim_size="${argument#*=}"; shift ;;
    --victimSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      victim_size="$2"; shift 2 ;;
    --edgeOrders=*) edge_orders="${argument#*=}"; shift ;;
    --edgeOrders)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      edge_orders="$2"; shift 2 ;;
    --topologies=*) topologies="${argument#*=}"; shift ;;
    --topologies)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      topologies="$2"; shift 2 ;;
    --runs=*) runs="${argument#*=}"; shift ;;
    --runs)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      runs="$2"; shift 2 ;;
    --repeats=*) repeats="${argument#*=}"; shift ;;
    --repeats)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      repeats="$2"; shift 2 ;;
    --diagnostic=*) diagnostic="${argument#*=}"; shift ;;
    --diagnostic)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      diagnostic="$2"; shift 2 ;;
    --trace=*) trace_enabled="${argument#*=}"; shift ;;
    --trace)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      trace_enabled="$2"; shift 2 ;;
    --outputRoot=*) output_root="${argument#*=}"; shift ;;
    --outputRoot)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      output_root="$2"; shift 2 ;;
    *) echo "ERROR: unknown option: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${runs}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --runs must be positive" >&2; exit 2; }
[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --repeats must be positive" >&2; exit 2; }
case "${diagnostic}" in 0|1) ;; *) echo "ERROR: --diagnostic must be 0 or 1" >&2; exit 2 ;; esac
case "${trace_enabled}" in 0|1) ;; *) echo "ERROR: --trace must be 0 or 1" >&2; exit 2 ;; esac

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 3 ]] || {
  echo "ERROR: Stage I requires exactly three devices" >&2; exit 2;
}
canonical_devices=""
for index in "${!device_array[@]}"; do
  value="${device_array[index]//[[:space:]]/}"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid device index: ${value}" >&2; exit 2; }
  [[ "${value}" == "${index}" ]] || { echo "ERROR: Stage I requires deviceList=0,1,2" >&2; exit 2; }
  [[ -z "${canonical_devices}" ]] || canonical_devices+=','
  canonical_devices+="${value}"
done
device_list="${canonical_devices}"

size_pattern='^[0-9]+([.][0-9]+)?([KMG]([iI]?[bB])?)?$'
[[ "${victim_size}" =~ ${size_pattern} ]] || { echo "ERROR: invalid victim size: ${victim_size}" >&2; exit 2; }

IFS=',' read -r -a background_array <<< "${background_sets}"
[[ "${#background_array[@]}" -gt 0 && -n "${background_array[0]}" ]] || {
  echo "ERROR: --backgroundSets cannot be empty" >&2; exit 2;
}
has_clean=0
for index in "${!background_array[@]}"; do
  value="${background_array[index]//[[:space:]]/}"
  case "${value}" in none|0|1|2|all) ;; *) echo "ERROR: unsupported background set: ${value}" >&2; exit 2 ;; esac
  [[ "${value}" == "none" ]] && has_clean=1
  background_array[index]="${value}"
done
[[ "${has_clean}" -eq 1 ]] || { echo "ERROR: --backgroundSets must include none" >&2; exit 2; }

IFS=',' read -r -a background_size_array <<< "${background_sizes}"
for index in "${!background_size_array[@]}"; do
  value="${background_size_array[index]//[[:space:]]/}"
  [[ "${value}" =~ ${size_pattern} ]] || { echo "ERROR: invalid background size: ${value}" >&2; exit 2; }
  background_size_array[index]="${value}"
done

IFS=',' read -r -a direction_array <<< "${directions}"
for index in "${!direction_array[@]}"; do
  value="${direction_array[index]//[[:space:]]/}"
  case "${value}" in d2h|h2d) ;; *) echo "ERROR: unsupported direction: ${value}" >&2; exit 2 ;; esac
  direction_array[index]="${value}"
done

IFS=',' read -r -a duty_array <<< "${duty_cycles}"
for index in "${!duty_array[@]}"; do
  value="${duty_array[index]//[[:space:]]/}"
  [[ "${value}" =~ ^(0\.[0-9]+|1(\.0*)?)$ ]] || {
    echo "ERROR: duty cycle must be in (0,1]: ${value}" >&2; exit 2;
  }
  duty_array[index]="${value}"
done

IFS=',' read -r -a edge_order_array <<< "${edge_orders}"
for index in "${!edge_order_array[@]}"; do
  value="${edge_order_array[index]//[[:space:]]/}"
  case "${value}" in forward|reverse) ;; *) echo "ERROR: unsupported edge order: ${value}" >&2; exit 2 ;; esac
  edge_order_array[index]="${value}"
done

IFS=',' read -r -a topology_array <<< "${topologies}"
for index in "${!topology_array[@]}"; do
  value="${topology_array[index]//[[:space:]]/}"
  case "${value}" in shared|independent|independent+source-chain) ;; *) echo "ERROR: unsupported topology: ${value}" >&2; exit 2 ;; esac
  topology_array[index]="${value}"
done

[[ -x "${victim_bin}" && -x "${background_bin}" ]] || {
  echo "ERROR: missing CUDA copy executable; run scripts/build_cuda_copy.sh" >&2; exit 1;
}
if [[ "${trace_enabled}" == "1" && ! -x "${trace_lib}" ]]; then
  echo "ERROR: missing CUPTI tracer: ${trace_lib}" >&2; exit 1
fi

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/directional-dma/${timestamp}-$$"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"
summary_file="${output_root}/summary.csv"
[[ ! -e "${summary_file}" ]] || { echo "ERROR: ${summary_file} already exists" >&2; exit 1; }

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'deviceList=%s\nbackgroundSets=%s\nbackgroundSizes=%s\ndirections=%s\ndutyCycles=%s\nvictimSize=%s\nedgeOrders=%s\ntopologies=%s\nruns=%s\nrepeats=%s\ndiagnostic=%s\ntrace=%s\n' \
    "${device_list}" "${background_sets}" "${background_sizes}" "${directions}" \
    "${duty_cycles}" "${victim_size}" "${edge_orders}" "${topologies}" \
    "${runs}" "${repeats}" "${diagnostic}" "${trace_enabled}"
  printf 'stage=I; victim is GPU0->GPU1 and GPU0->GPU2; background sizes are independent of victim size\n'
  printf 'warmup=10; performance mode uses batch start/stop events; diagnostic=1 adds per-operation CUDA events\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'backgroundSize,victimSize,direction,dutyCycle,backgroundSet,edgeOrder,topology,repetition,deviceList,victimAggregateGBps,backgroundAggregateGBps,targetDuty,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog,traceBackground,traceVictim\n' > "${summary_file}"

extract_metric() {
  local file="$1" metric="$2"
  [[ -f "${file}" ]] || return 0
  sed -n "s/.*${metric}=\([0-9.+eE-]*\).*/\1/p" "${file}" 2>/dev/null | tail -n 1
}

background_devices_for() {
  case "$1" in
    0|1|2) echo "$1" ;;
    all) echo '0,1,2' ;;
    none) echo '' ;;
  esac
}

topology_args_for() {
  case "$1" in
    shared) topology_args=(--streamMode=shared --streamDependency=none) ;;
    independent) topology_args=(--streamMode=independent --streamDependency=none) ;;
    independent+source-chain) topology_args=(--streamMode=independent --streamDependency=source-chain) ;;
  esac
}

trace_env_for() {
  local output="$1"
  trace_env=()
  if [[ "${trace_enabled}" == "1" ]]; then
    local preload="${trace_lib}"
    [[ -z "${LD_PRELOAD:-}" ]] || preload+=":${LD_PRELOAD}"
    trace_env=("COPYBENCH_CUPTI_MEMCPY_OUTPUT=${output}" "LD_PRELOAD=${preload}")
  fi
}

start_background() {
  local devices="$1" case_dir="$2" json="$3" log="$4" trace_output="$5"
  background_stop="${case_dir}/stop"
  background_pid=""
  background_rc=0
  rm -f "${case_dir}/ready" "${background_stop}"
  background_cmd=("${background_bin}" "--devList=${devices}" "--direction=${case_direction}"
    "--size=${case_background_size}" "--dutyCycle=${case_duty}"
    "--readyFile=${case_dir}/ready" "--stopFile=${background_stop}"
    --reportSec=1 "--output=${json}")
  trace_env_for "${trace_output}"
  printf '%q ' env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${background_cmd[@]}" > "${case_dir}/background.command"
  printf '\n' >> "${case_dir}/background.command"
  env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${background_cmd[@]}" >"${log}" 2>&1 &
  background_pid=$!
  for ((attempt = 0; attempt < 300; ++attempt)); do
    [[ -e "${case_dir}/ready" ]] && return 0
    if ! kill -0 "${background_pid}" 2>/dev/null; then break; fi
    sleep 0.1
  done
  echo "ERROR: background ${devices} did not become ready" >&2
  stop_background
  return 1
}

run_case() {
  local background_size_arg="$1" direction_arg="$2" duty_arg="$3"
  local background_set="$4" edge_order="$5" topology="$6" repetition="$7"
  case_background_size="${background_size_arg}"
  case_direction="${direction_arg}"
  case_duty="${duty_arg}"
  local background_devices edge_order_value
  background_devices="$(background_devices_for "${background_set}")"
  if [[ "${edge_order}" == "forward" ]]; then
    edge_order_value='0->1,0->2'
  else
    edge_order_value='0->2,0->1'
  fi
  topology_args_for "${topology}"
  local case_dir="${output_root}/backgroundSize-${background_size_arg}/direction-${direction_arg}/duty-${duty_arg}/background-${background_set}/edge-${edge_order}/topology-${topology}/repetition-${repetition}"
  local victim_json="${case_dir}/victim.json" background_json="${case_dir}/background.json"
  local victim_log="${case_dir}/victim.log" background_log="${case_dir}/background.log"
  local victim_trace="${case_dir}/victim.memcpy.csv" background_trace="${case_dir}/background.memcpy.csv"
  local victim_entry="NA" background_entry="NA" trace_victim_entry="NA" trace_background_entry="NA"
  local victim_rc=1 case_background_rc=0 victim_gbps="NA" background_gbps="NA" status="fail"
  local -a victim_cmd trace_env

  mkdir -p "${case_dir}"
  {
    printf 'stage=I backgroundSize=%s victimSize=%s direction=%s dutyCycle=%s backgroundSet=%s backgroundDevices=%s edgeOrder=%s topology=%s repetition=%s\n' \
      "${case_background_size}" "${victim_size}" "${case_direction}" "${case_duty}" \
      "${background_set}" "${background_devices:-none}" "${edge_order_value}" "${topology}" "${repetition}"
    printf 'repeats=%s warmup=10 source=0 destinations=1,2 diagnostic=%s\n' "${repeats}" "${diagnostic}"
  } > "${case_dir}/environment.txt"

  victim_cmd=("${victim_bin}" "--deviceList=${device_list}" "--edgeOrder=${edge_order_value}"
    "${topology_args[@]}" "--repeats=${repeats}" "--size=${victim_size}"
    "--diagnostic=${diagnostic}" "--output=${victim_json}")
  trace_env_for "${victim_trace}"
  printf '%q ' env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${victim_cmd[@]}" > "${case_dir}/victim.command"
  printf '\n' >> "${case_dir}/victim.command"

  if [[ "${background_set}" != "none" ]]; then
    start_background "${background_devices}" "${case_dir}" "${background_json}" "${background_log}" "${background_trace}"
  fi
  trace_env_for "${victim_trace}"
  if env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${victim_cmd[@]}" >"${victim_log}" 2>&1; then
    victim_rc=0
  else
    victim_rc=$?
  fi
  if [[ "${background_set}" != "none" ]]; then
    stop_background
    case_background_rc="${background_rc}"
  fi
  [[ -s "${victim_json}" ]] && victim_entry="${victim_json}"
  [[ -s "${background_json}" ]] && background_entry="${background_json}"
  [[ -s "${victim_trace}" ]] && trace_victim_entry="${victim_trace}"
  [[ -s "${background_trace}" ]] && trace_background_entry="${background_trace}"
  victim_gbps="$(extract_metric "${victim_log}" aggregateGBps)"; [[ -n "${victim_gbps}" ]] || victim_gbps="NA"
  background_gbps="$(extract_metric "${background_log}" aggregateGBps)"; [[ -n "${background_gbps}" ]] || background_gbps="NA"
  [[ "${background_set}" == "none" ]] && { case_background_rc=0; background_gbps="NA"; }
  if [[ "${victim_rc}" -eq 0 && "${case_background_rc}" -eq 0 && -s "${victim_json}" ]]; then status="pass"; fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_background_size}" "${victim_size}" "${case_direction}" "${case_duty}" \
    "${background_set}" "${edge_order}" "${topology}" "${repetition}" "${device_list}" \
    "${victim_gbps}" "${background_gbps}" "${case_duty}" "${victim_rc}" "${case_background_rc}" "${status}" \
    "${background_entry}" "${victim_entry}" "${victim_log}" "${background_log}" \
    "${trace_background_entry}" "${trace_victim_entry}" >> "${summary_file}"
  echo "${status^^}: backgroundSize=${case_background_size} direction=${case_direction} duty=${case_duty} background=${background_set} edge=${edge_order} topology=${topology} repetition=${repetition} victimGBps=${victim_gbps} backgroundGBps=${background_gbps}"
}

for case_background_size in "${background_size_array[@]}"; do
  for edge_order in "${edge_order_array[@]}"; do
    for topology in "${topology_array[@]}"; do
      for ((repetition = 1; repetition <= runs; ++repetition)); do
        run_case "${case_background_size}" none NA none "${edge_order}" "${topology}" "${repetition}"
        for direction in "${direction_array[@]}"; do
          for duty in "${duty_array[@]}"; do
            for background_set in "${background_array[@]}"; do
              [[ "${background_set}" == "none" ]] && continue
              run_case "${case_background_size}" "${direction}" "${duty}" "${background_set}" \
                "${edge_order}" "${topology}" "${repetition}"
            done
          done
        done
      done
    done
  done
done

echo "Completed Stage I directional DMA diagnostic: ${output_root}"
echo "Summary: ${summary_file}"
