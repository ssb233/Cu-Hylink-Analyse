#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
d2d_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
replacement_bin="${repo_root}/build/cuda_copy/copy_path_background"
trace_lib="${repo_root}/build/cuda_copy/libcupti_memcpy_channel_trace.so"

device_list="0,1,2"
background_paths="none,original-d2h,local-d2d-ce,streaming-hbm-read,streaming-hbm-write,l2-resident-read"
pressure_levels="current-pulsed,duty-0.01,duty-0.1,duty-0.5,duty-1.0,saturated"
topologies="single-two-copy,edge-independent"
runs=3
repeats=300
size="255M"
background_size="255M"
target_gbps="4.0"
trace_enabled=0
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_copy_path_temporal_ablation.sh [options]

Run the three-GPU Stage H current-pulsed, duty-sweep, and saturated cases.
single-two-copy is one source stream with two consecutive allpairs copies.

Options:
  --deviceList=0,1,2
  --backgroundPaths=none,original-d2h,local-d2d-ce,streaming-hbm-read,streaming-hbm-write,l2-resident-read
  --pressureLevels=current-pulsed,duty-0.01,duty-0.1,duty-0.5,duty-1.0,saturated
  --topologies=single-two-copy,edge-independent
  --runs=3                  repetitions per matrix point
  --repeats=300             measured victim repetitions
  --size=255M               victim bytes per edge
  --backgroundSize=255M     background requested/working-set size
  --targetGBps=4.0          target per GPU for non-saturated levels
  --trace=0|1               preload CUPTI memcpy tracer
  --outputRoot=<path>       fresh result directory
  --help

Warmup is fixed at 10. Duty levels use targetGBps=0 so duty is independently
paced; saturated also maps to targetGBps=0 and dutyCycle=1.0.
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
    --backgroundPaths=*) background_paths="${argument#*=}"; shift ;;
    --backgroundPaths)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_paths="$2"; shift 2 ;;
    --pressureLevels=*) pressure_levels="${argument#*=}"; shift ;;
    --pressureLevels)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      pressure_levels="$2"; shift 2 ;;
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
    --size=*) size="${argument#*=}"; shift ;;
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      size="$2"; shift 2 ;;
    --backgroundSize=*) background_size="${argument#*=}"; shift ;;
    --backgroundSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_size="$2"; shift 2 ;;
    --targetGBps=*) target_gbps="${argument#*=}"; shift ;;
    --targetGBps)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      target_gbps="$2"; shift 2 ;;
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
[[ "${target_gbps}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "ERROR: --targetGBps must be a non-negative decimal" >&2; exit 2;
}
case "${trace_enabled}" in 0|1) ;; *) echo "ERROR: --trace must be 0 or 1" >&2; exit 2 ;; esac

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 3 ]] || {
  echo "ERROR: Stage H requires exactly three devices" >&2; exit 2;
}
canonical_devices=""
for index in "${!device_array[@]}"; do
  device_array[index]="${device_array[index]//[[:space:]]/}"
  [[ "${device_array[index]}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid device index: ${device_array[index]}" >&2; exit 2;
  }
  for ((previous = 0; previous < index; ++previous)); do
    [[ "${device_array[index]}" != "${device_array[previous]}" ]] || {
      echo "ERROR: duplicate device index: ${device_array[index]}" >&2; exit 2;
    }
  done
  [[ -z "${canonical_devices}" ]] || canonical_devices+=','
  canonical_devices+="${device_array[index]}"
done
device_list="${canonical_devices}"

IFS=',' read -r -a background_array <<< "${background_paths}"
[[ "${#background_array[@]}" -gt 0 && -n "${background_array[0]}" ]] || {
  echo "ERROR: --backgroundPaths cannot be empty" >&2; exit 2;
}
for index in "${!background_array[@]}"; do
  value="${background_array[index]//[[:space:]]/}"
  case "${value}" in
    none|original-d2h|local-d2d-ce|streaming-hbm-read|streaming-hbm-write|l2-resident-read) ;;
    *) echo "ERROR: unsupported background path: ${value}" >&2; exit 2 ;;
  esac
  background_array[index]="${value}"
done

IFS=',' read -r -a pressure_array <<< "${pressure_levels}"
[[ "${#pressure_array[@]}" -gt 0 && -n "${pressure_array[0]}" ]] || {
  echo "ERROR: --pressureLevels cannot be empty" >&2; exit 2;
}
for index in "${!pressure_array[@]}"; do
  value="${pressure_array[index]//[[:space:]]/}"
  case "${value}" in
    current-pulsed|duty-0.01|duty-0.1|duty-0.5|duty-1.0|saturated) ;;
    *) echo "ERROR: unsupported pressure level: ${value}" >&2; exit 2 ;;
  esac
  pressure_array[index]="${value}"
done

IFS=',' read -r -a topology_array <<< "${topologies}"
[[ "${#topology_array[@]}" -gt 0 && -n "${topology_array[0]}" ]] || {
  echo "ERROR: --topologies cannot be empty" >&2; exit 2;
}
for index in "${!topology_array[@]}"; do
  value="${topology_array[index]//[[:space:]]/}"
  case "${value}" in single-two-copy|edge-independent) ;;
    *) echo "ERROR: unsupported topology: ${value}" >&2; exit 2 ;;
  esac
  topology_array[index]="${value}"
done

[[ -x "${d2d_bin}" && -x "${background_bin}" && -x "${replacement_bin}" ]] || {
  echo "ERROR: missing CUDA copy executable; run scripts/build_cuda_copy.sh" >&2; exit 1;
}
if [[ "${trace_enabled}" == "1" && ! -x "${trace_lib}" ]]; then
  echo "ERROR: missing CUPTI tracer: ${trace_lib}" >&2; exit 1
fi

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/copy-path-temporal/${timestamp}-$$"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"
summary_file="${output_root}/summary.csv"
[[ ! -e "${summary_file}" ]] || { echo "ERROR: ${summary_file} already exists" >&2; exit 1; }

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'deviceList=%s\nbackgroundPaths=%s\npressureLevels=%s\ntopologies=%s\nruns=%s\nrepeats=%s\nvictimSize=%s\nbackgroundSize=%s\ntargetGBps=%s\ntrace=%s\n' \
    "${device_list}" "${background_paths}" "${pressure_levels}" "${topologies}" \
    "${runs}" "${repeats}" "${size}" "${background_size}" "${target_gbps}" "${trace_enabled}"
  printf 'three_gpu_topology_adaptation=single-two-copy means one source stream with two consecutive copies; edge-independent has one active stream per outgoing edge\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'pressureLevel,backgroundPath,victimMode,topology,repetition,deviceList,victimSize,backgroundSize,targetGBps,dutyCycle,bandwidthClass,victimAggregateGBps,backgroundAggregateGBps,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog,traceBackground,traceVictim\n' > "${summary_file}"

extract_metric() {
  local file="$1" metric="$2"
  [[ -f "${file}" ]] || return 0
  sed -n "s/.*${metric}=\([0-9.+eE-]*\).*/\1/p" "${file}" 2>/dev/null | tail -n 1
}

topology_args_for() {
  local topology="$1"
  case "${topology}" in
    single-two-copy) topology_args=(--streamMode=per-source --streamsPerSource=1) ;;
    edge-independent) topology_args=(--streamMode=per-edge --streamsPerSource=3) ;;
  esac
}

pressure_values_for() {
  local level="$1"
  case "${level}" in
    current-pulsed) case_target="${target_gbps}"; case_duty="1.0"; case_class="bandwidth-matched" ;;
    duty-0.01) case_target="0"; case_duty="0.01"; case_class="duty-matched" ;;
    duty-0.1) case_target="0"; case_duty="0.1"; case_class="duty-matched" ;;
    duty-0.5) case_target="0"; case_duty="0.5"; case_class="duty-matched" ;;
    duty-1.0) case_target="0"; case_duty="1.0"; case_class="duty-matched" ;;
    saturated) case_target="0"; case_duty="1.0"; case_class="saturated" ;;
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
  local path="$1" case_dir="$2" json="$3" log="$4" trace_output="$5"
  background_stop="${case_dir}/stop"
  background_pid=""
  background_rc=0
  rm -f "${case_dir}/ready" "${background_stop}"
  if [[ "${path}" == "original-d2h" ]]; then
    background_cmd=("${background_bin}" "--devList=${device_list}" --direction=d2h
      "--size=${background_size}" "--readyFile=${case_dir}/ready"
      "--stopFile=${background_stop}" "--output=${json}")
  else
    background_cmd=("${replacement_bin}" "--deviceList=${device_list}" "--path=${path}"
      "--size=${background_size}" "--targetGBps=${case_target}"
      "--dutyCycle=${case_duty}" "--readyFile=${case_dir}/ready"
      "--stopFile=${background_stop}" "--output=${json}")
  fi
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
  echo "ERROR: background ${path} did not become ready" >&2
  stop_background
  return 1
}

run_case() {
  local pressure="$1" background_path="$2" topology="$3" repetition="$4"
  pressure_values_for "${pressure}"
  [[ "${background_path}" == "none" ]] && case_class="control"
  [[ "${background_path}" == "original-d2h" ]] && case_class="original-d2h"
  local case_dir="${output_root}/pressure-${pressure}/background-${background_path}/topology-${topology}/repetition-${repetition}"
  local background_json="${case_dir}/background.json" victim_json="${case_dir}/victim.json"
  local background_log="${case_dir}/background.log" victim_log="${case_dir}/victim.log"
  local background_trace="${case_dir}/background.memcpy.csv" victim_trace="${case_dir}/victim.memcpy.csv"
  local background_entry="NA" victim_entry="NA" trace_background_entry="NA" trace_victim_entry="NA"
  local victim_rc=1 case_background_rc=0 victim_gbps="NA" background_gbps="NA" status="fail"
  local -a topology_args victim_cmd trace_env

  mkdir -p "${case_dir}"
  topology_args_for "${topology}"
  {
    printf 'pressureLevel=%s backgroundPath=%s victimMode=original-p2p-ce topology=%s repetition=%s\n' \
      "${pressure}" "${background_path}" "${topology}" "${repetition}"
    printf 'deviceList=%s victimSize=%s backgroundSize=%s targetGBps=%s dutyCycle=%s class=%s repeats=%s\n' \
      "${device_list}" "${size}" "${background_size}" "${case_target}" "${case_duty}" "${case_class}" "${repeats}"
  } > "${case_dir}/environment.txt"

  victim_cmd=("${d2d_bin}" --pattern=allpairs "--deviceList=${device_list}"
    --edgeOrder=source-major "${topology_args[@]}" "--size=${size}"
    "--repeats=${repeats}" "--output=${victim_json}")
  trace_env_for "${victim_trace}"
  printf '%q ' env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${victim_cmd[@]}" > "${case_dir}/victim.command"
  printf '\n' >> "${case_dir}/victim.command"

  if [[ "${background_path}" != "none" ]]; then
    start_background "${background_path}" "${case_dir}" "${background_json}" \
      "${background_log}" "${background_trace}"
  fi
  if [[ "${background_path}" == "none" || -n "${background_pid}" ]]; then
    trace_env_for "${victim_trace}"
    if env -u CUDA_DEVICE_MAX_CONNECTIONS "${trace_env[@]}" "${victim_cmd[@]}" >"${victim_log}" 2>&1; then
      victim_rc=0
    else
      victim_rc=$?
    fi
  fi
  if [[ "${background_path}" != "none" ]]; then
    stop_background
    case_background_rc="${background_rc}"
  fi
  [[ -s "${victim_json}" ]] && victim_entry="${victim_json}"
  [[ -s "${background_json}" ]] && background_entry="${background_json}"
  [[ -s "${victim_trace}" ]] && trace_victim_entry="${victim_trace}"
  [[ -s "${background_trace}" ]] && trace_background_entry="${background_trace}"
  victim_gbps="$(extract_metric "${victim_log}" aggregateGBps)"; [[ -n "${victim_gbps}" ]] || victim_gbps="NA"
  background_gbps="$(extract_metric "${background_log}" aggregateGBps)"; [[ -n "${background_gbps}" ]] || background_gbps="NA"
  [[ "${background_path}" == "none" ]] && { case_background_rc=0; background_gbps="NA"; }
  if [[ "${victim_rc}" -eq 0 && "${case_background_rc}" -eq 0 && -s "${victim_json}" ]]; then status="pass"; fi

  printf '%s,%s,original-p2p-ce,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${pressure}" "${background_path}" "${topology}" "${repetition}" "${device_list}" \
    "${size}" "${background_size}" "${case_target}" "${case_duty}" "${case_class}" \
    "${victim_gbps}" "${background_gbps}" "${victim_rc}" "${case_background_rc}" "${status}" \
    "${background_entry}" "${victim_entry}" "${victim_log}" "${background_log}" \
    "${trace_background_entry}" "${trace_victim_entry}" >> "${summary_file}"
  echo "${status^^}: pressure=${pressure} background=${background_path} topology=${topology} repetition=${repetition} victimGBps=${victim_gbps} backgroundGBps=${background_gbps}"
}

for pressure in "${pressure_array[@]}"; do
  for background_path in "${background_array[@]}"; do
    for topology in "${topology_array[@]}"; do
      for ((repetition = 1; repetition <= runs; ++repetition)); do
        run_case "${pressure}" "${background_path}" "${topology}" "${repetition}"
      done
    done
  done
done

echo "Completed temporal copy-path ablation: ${output_root}"
echo "Summary: ${summary_file}"
