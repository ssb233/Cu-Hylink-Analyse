#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_bin="${repo_root}/build/cuda_copy/minimal_source_pair_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
trace_lib="${repo_root}/build/cuda_copy/libcupti_memcpy_channel_trace.so"

device_list="0,1,2"
background_sets="none,0,1,2,all"
sizes="64K,128K,256K,512K,1M,2M,4M"
background_size=""
edge_orders="forward,reverse"
stream_modes="shared,independent"
stream_dependencies="none,source-chain"
runs=3
repeats=300
trace_enabled=0
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_minimal_source_diagnostic.sh [options]

Run the two-edge source-local diagnostic for GPU0->GPU1 and GPU0->GPU2.

Options:
  --deviceList=0,1,2
  --backgroundSets=none,0,1,2,all
  --sizes=64K,128K,256K,512K,1M,2M,4M
  --backgroundSize=<size>    host-copy payload; defaults to the victim size
  --edgeOrders=forward,reverse
  --streamModes=shared,independent
  --streamDependencies=none,source-chain
  --runs=3                  repetitions per matrix point
  --repeats=300             measured copies per edge
  --trace=0|1               preload the CUPTI memcpy tracer
  --outputRoot=<path>       fresh result directory
  --help

Warmup is fixed at 10. source-chain is only valid with independent streams.
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
    --sizes=*) sizes="${argument#*=}"; shift ;;
    --sizes)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      sizes="$2"; shift 2 ;;
    --backgroundSize=*) background_size="${argument#*=}"; shift ;;
    --backgroundSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_size="$2"; shift 2 ;;
    --edgeOrders=*) edge_orders="${argument#*=}"; shift ;;
    --edgeOrders)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      edge_orders="$2"; shift 2 ;;
    --streamModes=*) stream_modes="${argument#*=}"; shift ;;
    --streamModes)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      stream_modes="$2"; shift 2 ;;
    --streamDependencies=*) stream_dependencies="${argument#*=}"; shift ;;
    --streamDependencies)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      stream_dependencies="$2"; shift 2 ;;
    --runs=*) runs="${argument#*=}"; shift ;;
    --runs)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      runs="$2"; shift 2 ;;
    --repeats=*) repeats="${argument#*=}"; shift ;;
    --repeats)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      repeats="$2"; shift 2 ;;
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
case "${trace_enabled}" in 0|1) ;; *) echo "ERROR: --trace must be 0 or 1" >&2; exit 2 ;; esac

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 3 ]] || { echo "ERROR: H3 requires exactly three devices" >&2; exit 2; }
canonical_devices=""
for index in "${!device_array[@]}"; do
  value="${device_array[index]//[[:space:]]/}"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid device index: ${value}" >&2; exit 2; }
  [[ "${value}" == "${index}" ]] || { echo "ERROR: H3 requires deviceList=0,1,2" >&2; exit 2; }
  [[ -z "${canonical_devices}" ]] || canonical_devices+=','
  canonical_devices+="${value}"
done
device_list="${canonical_devices}"

IFS=',' read -r -a background_array <<< "${background_sets}"
for index in "${!background_array[@]}"; do
  value="${background_array[index]//[[:space:]]/}"
  case "${value}" in none|0|1|2|all) ;; *) echo "ERROR: unsupported background set: ${value}" >&2; exit 2 ;; esac
  background_array[index]="${value}"
done
IFS=',' read -r -a size_array <<< "${sizes}"
for index in "${!size_array[@]}"; do
  value="${size_array[index]//[[:space:]]/}"
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?([KMG]([iI]?[bB])?)?$ ]] || { echo "ERROR: invalid size: ${value}" >&2; exit 2; }
  size_array[index]="${value}"
done
if [[ -n "${background_size}" ]]; then
  [[ "${background_size}" =~ ^[0-9]+([.][0-9]+)?([KMG]([iI]?[bB])?)?$ ]] || {
    echo "ERROR: invalid background size: ${background_size}" >&2; exit 2;
  }
fi
IFS=',' read -r -a edge_order_array <<< "${edge_orders}"
for index in "${!edge_order_array[@]}"; do
  value="${edge_order_array[index]//[[:space:]]/}"
  case "${value}" in forward|reverse) ;; *) echo "ERROR: unsupported edge order: ${value}" >&2; exit 2 ;; esac
  edge_order_array[index]="${value}"
done
IFS=',' read -r -a stream_mode_array <<< "${stream_modes}"
for index in "${!stream_mode_array[@]}"; do
  value="${stream_mode_array[index]//[[:space:]]/}"
  case "${value}" in shared|independent) ;; *) echo "ERROR: unsupported stream mode: ${value}" >&2; exit 2 ;; esac
  stream_mode_array[index]="${value}"
done
IFS=',' read -r -a dependency_array <<< "${stream_dependencies}"
for index in "${!dependency_array[@]}"; do
  value="${dependency_array[index]//[[:space:]]/}"
  case "${value}" in none|source-chain) ;; *) echo "ERROR: unsupported stream dependency: ${value}" >&2; exit 2 ;; esac
  dependency_array[index]="${value}"
done

[[ -x "${source_bin}" && -x "${background_bin}" ]] || { echo "ERROR: missing CUDA executable; run scripts/build_cuda_copy.sh" >&2; exit 1; }
if [[ "${trace_enabled}" == "1" && ! -x "${trace_lib}" ]]; then
  echo "ERROR: missing CUPTI tracer: ${trace_lib}" >&2; exit 1
fi

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/minimal-source-diagnostic/${timestamp}-$$"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"
summary_file="${output_root}/summary.csv"
[[ ! -e "${summary_file}" ]] || { echo "ERROR: ${summary_file} already exists" >&2; exit 1; }

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'deviceList=%s\nbackgroundSets=%s\nsizes=%s\nbackgroundSize=%s\nedgeOrders=%s\nstreamModes=%s\nstreamDependencies=%s\nruns=%s\nrepeats=%s\ntrace=%s\n' \
    "${device_list}" "${background_sets}" "${sizes}" "${background_size:-per-victim-size}" "${edge_orders}" "${stream_modes}" "${stream_dependencies}" "${runs}" "${repeats}" "${trace_enabled}"
  printf 'H3 definition=source GPU0 to destination GPUs 1 and 2; shared is one stream and independent is one stream per edge\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'size,backgroundSet,streamMode,streamDependency,edgeOrder,repetition,deviceList,victimSize,backgroundSize,victimAggregateGBps,backgroundAggregateGBps,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog,traceBackground,traceVictim\n' > "${summary_file}"

extract_metric() {
  local file="$1" metric="$2"
  [[ -f "${file}" ]] || return 0
  sed -n "s/.*${metric}=\([0-9.+eE-]*\).*/\1/p" "${file}" 2>/dev/null | tail -n 1
}

edge_value_for() {
  case "$1" in
    forward) echo '0->1,0->2' ;;
    reverse) echo '0->2,0->1' ;;
  esac
}

background_devices_for() {
  case "$1" in
    0|1|2) echo "$1" ;;
    all) echo '0,1,2' ;;
    none) echo '' ;;
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
  local -a command
  command=("${background_bin}" "--devList=${devices}" --direction=d2h "--size=${case_background_size}" \
    "--readyFile=${case_dir}/ready" "--stopFile=${background_stop}" "--output=${json}")
  trace_env_for "${trace_output}"
  printf '%q ' env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${command[@]}" > "${case_dir}/background.command"
  printf '\n' >> "${case_dir}/background.command"
  env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${command[@]}" >"${log}" 2>&1 &
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
  local background_set="$1" stream_mode="$2" dependency="$3" edge_order="$4" repetition="$5"
  local background_devices edge_order_value
  background_devices="$(background_devices_for "${background_set}")"
  edge_order_value="$(edge_value_for "${edge_order}")"
  local case_dir="${output_root}/size-${case_size}/background-${background_set}/stream-${stream_mode}/dependency-${dependency}/edge-${edge_order}/repetition-${repetition}"
  local victim_json="${case_dir}/victim.json" background_json="${case_dir}/background.json"
  local victim_log="${case_dir}/victim.log" background_log="${case_dir}/background.log"
  local victim_trace="${case_dir}/victim.memcpy.csv" background_trace="${case_dir}/background.memcpy.csv"
  local victim_entry="NA" background_entry="NA" trace_victim_entry="NA" trace_background_entry="NA"
  local victim_rc=1 case_background_rc=0 victim_gbps="NA" background_gbps="NA" status="fail"
  local -a victim_cmd trace_env

  mkdir -p "${case_dir}"
  {
    printf 'size=%s backgroundSize=%s backgroundSet=%s backgroundDevices=%s streamMode=%s streamDependency=%s edgeOrder=%s repetition=%s\n' \
      "${case_size}" "${case_background_size}" "${background_set}" "${background_devices:-none}" "${stream_mode}" "${dependency}" "${edge_order_value}" "${repetition}"
    printf 'repeats=%s warmup=10 source=0 destinations=1,2 diagnostic=1\n' "${repeats}"
  } > "${case_dir}/environment.txt"

  victim_cmd=("${source_bin}" "--deviceList=${device_list}" "--edgeOrder=${edge_order_value}" \
    "--streamMode=${stream_mode}" "--streamDependency=${dependency}" "--repeats=${repeats}" \
    "--size=${case_size}" --diagnostic=1 "--output=${victim_json}")
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

  printf '%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_size}" "${background_set}" "${stream_mode}" "${dependency}" "${edge_order}" "${repetition}" "${device_list}" \
    "${case_size}" "${case_background_size}" "${victim_gbps}" "${background_gbps}" "${victim_rc}" "${case_background_rc}" "${status}" \
    "${background_entry}" "${victim_entry}" "${victim_log}" "${background_log}" "${trace_background_entry}" "${trace_victim_entry}" >> "${summary_file}"
  echo "${status^^}: size=${case_size} background=${background_set} stream=${stream_mode} dependency=${dependency} edge=${edge_order} repetition=${repetition} victimGBps=${victim_gbps}"
}

for case_size in "${size_array[@]}"; do
  case_background_size="${background_size:-${case_size}}"
  for background_set in "${background_array[@]}"; do
    for stream_mode in "${stream_mode_array[@]}"; do
      for dependency in "${dependency_array[@]}"; do
        if [[ "${stream_mode}" == "shared" && "${dependency}" == "source-chain" ]]; then
          continue
        fi
        for edge_order in "${edge_order_array[@]}"; do
          for ((repetition = 1; repetition <= runs; ++repetition)); do
            run_case "${background_set}" "${stream_mode}" "${dependency}" "${edge_order}" "${repetition}"
          done
        done
      done
    done
  done
done

echo "Completed minimal source diagnostic: ${output_root}"
echo "Summary: ${summary_file}"
