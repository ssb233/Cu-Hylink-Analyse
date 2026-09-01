#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
d2d_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
kernel_victim_bin="${repo_root}/build/cuda_copy/p2p_kernel_bw"
original_background_bin="${repo_root}/build/cuda_copy/host_copy_background"
replacement_background_bin="${repo_root}/build/cuda_copy/copy_path_background"

device_list="0,1,2"
background_paths="none,original-d2h,local-d2d-ce,streaming-hbm-read,streaming-hbm-write,l2-resident-read"
victim_modes="original-p2p-ce"
topologies="single-two-copy,edge-independent"
runs=3
repeats=300
size="255M"
background_size="255M"
target_gbps=4.0
duty_cycle=1.0
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_copy_path_ablation.sh [options]

Run the minimum Stage G copy-path ablation on three V100 GPUs.

Options:
  --deviceList=0,1,2
  --backgroundPaths=none,original-d2h,local-d2d-ce,streaming-hbm-read,streaming-hbm-write,l2-resident-read
  --victimModes=original-p2p-ce,local-d2d-ce,peer-read,peer-write
  --topologies=single-two-copy,edge-independent
  --runs=3                  repetitions for every matrix point
  --repeats=300             measured repetitions
  --size=255M               victim bytes per edge
  --backgroundSize=255M     background working-set bytes
  --targetGBps=4.0          replacement background target per GPU
  --dutyCycle=1.0           replacement background active-time fraction
  --outputRoot=<path>       fresh result directory
  --help

Warmup is fixed at 10 iterations. Background replacement paths report their
actual bytes, operations, active time, and per-GPU rates in JSON.
EOF
}

stop_background() {
  background_rc=0
  if [[ -z "${background_pid}" ]]; then
    return 0
  fi
  if [[ -n "${background_stop}" ]]; then
    touch "${background_stop}"
  fi
  if kill -0 "${background_pid}" 2>/dev/null; then
    kill -INT "${background_pid}" 2>/dev/null || true
  fi
  wait "${background_pid}" || background_rc=$?
  background_pid=""
  background_stop=""
}

on_exit() {
  stop_background
}

on_signal() {
  stop_background
  exit 130
}

trap on_exit EXIT
trap on_signal INT TERM

while [[ $# -gt 0 ]]; do
  argument="$1"
  case "${argument}" in
    --help|-h)
      usage
      exit 0
      ;;
    --deviceList=*|--device-list=*)
      device_list="${argument#*=}"
      shift
      ;;
    --deviceList|--device-list)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      device_list="$2"
      shift 2
      ;;
    --backgroundPaths=*)
      background_paths="${argument#*=}"
      shift
      ;;
    --backgroundPaths)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_paths="$2"
      shift 2
      ;;
    --victimModes=*)
      victim_modes="${argument#*=}"
      shift
      ;;
    --victimModes)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      victim_modes="$2"
      shift 2
      ;;
    --topologies=*)
      topologies="${argument#*=}"
      shift
      ;;
    --topologies)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      topologies="$2"
      shift 2
      ;;
    --runs=*)
      runs="${argument#*=}"
      shift
      ;;
    --runs)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      runs="$2"
      shift 2
      ;;
    --repeats=*)
      repeats="${argument#*=}"
      shift
      ;;
    --repeats)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      repeats="$2"
      shift 2
      ;;
    --size=*)
      size="${argument#*=}"
      shift
      ;;
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      size="$2"
      shift 2
      ;;
    --backgroundSize=*)
      background_size="${argument#*=}"
      shift
      ;;
    --backgroundSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      background_size="$2"
      shift 2
      ;;
    --targetGBps=*)
      target_gbps="${argument#*=}"
      shift
      ;;
    --targetGBps)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      target_gbps="$2"
      shift 2
      ;;
    --dutyCycle=*)
      duty_cycle="${argument#*=}"
      shift
      ;;
    --dutyCycle)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      duty_cycle="$2"
      shift 2
      ;;
    --outputRoot=*)
      output_root="${argument#*=}"
      shift
      ;;
    --outputRoot)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      output_root="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown option: ${argument}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${runs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --runs must be a positive integer" >&2
  exit 2
}
[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --repeats must be a positive integer" >&2
  exit 2
}

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 3 ]] || {
  echo "ERROR: Stage G three-GPU adaptation requires exactly three devices" >&2
  exit 2
}
canonical_devices=""
for index in "${!device_array[@]}"; do
  device_array[index]="${device_array[index]//[[:space:]]/}"
  [[ "${device_array[index]}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid device index: ${device_array[index]}" >&2
    exit 2
  }
  for ((previous = 0; previous < index; ++previous)); do
    if [[ "${device_array[index]}" == "${device_array[previous]}" ]]; then
      echo "ERROR: duplicate device index: ${device_array[index]}" >&2
      exit 2
    fi
  done
  if [[ -n "${canonical_devices}" ]]; then canonical_devices+=","; fi
  canonical_devices+="${device_array[index]}"
done
device_list="${canonical_devices}"

IFS=',' read -r -a background_array <<< "${background_paths}"
[[ "${#background_array[@]}" -gt 0 && -n "${background_array[0]}" ]] || {
  echo "ERROR: --backgroundPaths cannot be empty" >&2
  exit 2
}
for index in "${!background_array[@]}"; do
  value="${background_array[index]//[[:space:]]/}"
  case "${value}" in
    none|original-d2h|local-d2d-ce|streaming-hbm-read|streaming-hbm-write|l2-resident-read) ;;
    *) echo "ERROR: unsupported background path: ${value}" >&2; exit 2 ;;
  esac
  background_array[index]="${value}"
done

IFS=',' read -r -a victim_array <<< "${victim_modes}"
[[ "${#victim_array[@]}" -gt 0 && -n "${victim_array[0]}" ]] || {
  echo "ERROR: --victimModes cannot be empty" >&2
  exit 2
}
for index in "${!victim_array[@]}"; do
  value="${victim_array[index]//[[:space:]]/}"
  case "${value}" in
    original-p2p-ce|local-d2d-ce|peer-read|peer-write) ;;
    *) echo "ERROR: unsupported victim mode: ${value}" >&2; exit 2 ;;
  esac
  victim_array[index]="${value}"
done

IFS=',' read -r -a topology_array <<< "${topologies}"
[[ "${#topology_array[@]}" -gt 0 && -n "${topology_array[0]}" ]] || {
  echo "ERROR: --topologies cannot be empty" >&2
  exit 2
}
for index in "${!topology_array[@]}"; do
  value="${topology_array[index]//[[:space:]]/}"
  case "${value}" in
    single-two-copy|edge-independent) ;;
    *) echo "ERROR: unsupported topology: ${value}" >&2; exit 2 ;;
  esac
  topology_array[index]="${value}"
done

[[ "${target_gbps}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "ERROR: --targetGBps must be a non-negative decimal" >&2
  exit 2
}
[[ "${duty_cycle}" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]] || {
  echo "ERROR: --dutyCycle must be in (0,1]" >&2
  exit 2
}
if [[ "${duty_cycle}" == "0" || "${duty_cycle}" == "0.0" ]]; then
  echo "ERROR: --dutyCycle must be in (0,1]" >&2
  exit 2
fi

[[ -x "${d2d_bin}" ]] || {
  echo "ERROR: missing executable: ${d2d_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${kernel_victim_bin}" ]] || {
  echo "ERROR: missing executable: ${kernel_victim_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${original_background_bin}" ]] || {
  echo "ERROR: missing executable: ${original_background_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${replacement_background_bin}" ]] || {
  echo "ERROR: missing executable: ${replacement_background_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/copy-path-ablation/${timestamp}-$$"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"
summary_file="${output_root}/summary.csv"
if [[ -e "${summary_file}" ]]; then
  echo "ERROR: output directory already contains ${summary_file}; choose a fresh --outputRoot" >&2
  exit 1
fi

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'NCCL_HOME=%s\n' "${NCCL_HOME:-unset}"
  printf 'deviceList=%s\nbackgroundPaths=%s\nvictimModes=%s\ntopologies=%s\nruns=%s\nrepeats=%s\nvictimSize=%s\nbackgroundSize=%s\ntargetGBps=%s\ndutyCycle=%s\n' \
    "${device_list}" "${background_paths}" "${victim_modes}" "${topologies}" \
    "${runs}" "${repeats}" "${size}" "${background_size}" "${target_gbps}" "${duty_cycle}"
  printf 'three_gpu_topology_adaptation=single-two-copy means one source stream with two consecutive allpairs copies; edge-independent has two active edge streams per source\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'backgroundPath,victimMode,topology,repetition,deviceList,victimSize,backgroundSize,targetGBps,dutyCycle,victimAggregateGBps,backgroundAggregateGBps,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog\n' > "${summary_file}"

extract_metric() {
  local file="$1"
  local metric="$2"
  if [[ ! -f "${file}" ]]; then return 0; fi
  sed -n "s/.*${metric}=\([0-9.+eE-]*\).*/\1/p" "${file}" 2>/dev/null | tail -n 1
}

topology_args_for() {
  local topology="$1"
  case "${topology}" in
    single-two-copy)
      topology_args=(--streamMode=per-source --streamsPerSource=1)
      ;;
    edge-independent)
      topology_args=(--streamMode=per-edge --streamsPerSource=3)
      ;;
  esac
}

start_background() {
  local path="$1"
  local case_dir="$2"
  local background_json="$3"
  local background_log="$4"
  local background_env_file="$5"
  background_stop="${case_dir}/stop"
  background_pid=""
  background_rc=0
  touch "${background_stop}"
  rm -f "${case_dir}/ready" "${background_stop}"
  if [[ "${path}" == "original-d2h" ]]; then
    background_cmd=(
      "${original_background_bin}"
      "--devList=${device_list}"
      --direction=d2h
      "--size=${background_size}"
      "--readyFile=${case_dir}/ready"
      "--stopFile=${background_stop}"
      "--output=${background_json}"
    )
  else
    background_cmd=(
      "${replacement_background_bin}"
      "--deviceList=${device_list}"
      "--path=${path}"
      "--size=${background_size}"
      "--targetGBps=${target_gbps}"
      "--dutyCycle=${duty_cycle}"
      "--readyFile=${case_dir}/ready"
      "--stopFile=${background_stop}"
      "--output=${background_json}"
    )
  fi
  printf '%q\n' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
    -u CUDA_DEVICE_MAX_CONNECTIONS "${background_cmd[@]}" > "${background_env_file}"
  env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS -u CUDA_DEVICE_MAX_CONNECTIONS \
    "${background_cmd[@]}" >"${background_log}" 2>&1 &
  background_pid=$!
  for ((attempt = 0; attempt < 300; ++attempt)); do
    if [[ -e "${case_dir}/ready" ]]; then return 0; fi
    if ! kill -0 "${background_pid}" 2>/dev/null; then break; fi
    sleep 0.1
  done
  echo "ERROR: background path ${path} did not become ready" >&2
  stop_background
  return 1
}

run_case() {
  local background_path="$1"
  local victim_mode="$2"
  local topology="$3"
  local repetition="$4"
  local case_dir="${output_root}/background-${background_path}/victim-${victim_mode}/topology-${topology}/repetition-${repetition}"
  local background_json="${case_dir}/background.json"
  local victim_json="${case_dir}/victim.json"
  local background_log="${case_dir}/background.log"
  local victim_log="${case_dir}/victim.log"
  local background_entry="NA"
  local victim_entry="NA"
  local background_log_entry="NA"
  local victim_rc=0
  local case_background_rc=0
  local victim_gbps="NA"
  local background_gbps="NA"
  local status="fail"
  local -a topology_args
  local -a victim_cmd

  mkdir -p "${case_dir}"
  topology_args_for "${topology}"
  {
    printf 'backgroundPath=%s victimMode=%s topology=%s repetition=%s\n' \
      "${background_path}" "${victim_mode}" "${topology}" "${repetition}"
    printf 'deviceList=%s victimSize=%s backgroundSize=%s targetGBps=%s dutyCycle=%s repeats=%s\n' \
      "${device_list}" "${size}" "${background_size}" "${target_gbps}" \
      "${duty_cycle}" "${repeats}"
  } > "${case_dir}/environment.txt"

  if [[ "${victim_mode}" == "original-p2p-ce" ]]; then
    victim_cmd=(
      "${d2d_bin}"
      --pattern=allpairs
      "--deviceList=${device_list}"
      --edgeOrder=source-major
      "${topology_args[@]}"
      "--size=${size}"
      "--repeats=${repeats}"
      "--output=${victim_json}"
    )
  else
    victim_cmd=(
      "${kernel_victim_bin}"
      "--deviceList=${device_list}"
      "--topology=${topology}"
      "--victimMode=${victim_mode}"
      "--size=${size}"
      "--repeats=${repeats}"
      "--output=${victim_json}"
    )
  fi
  printf 'victim_command=' > "${case_dir}/command.txt"
  printf '%q ' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
    -u CUDA_DEVICE_MAX_CONNECTIONS "${victim_cmd[@]}" >> "${case_dir}/command.txt"
  printf '\n' >> "${case_dir}/command.txt"
  printf '%q\n' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
    -u CUDA_DEVICE_MAX_CONNECTIONS "${victim_cmd[@]}" > "${case_dir}/victim.env"

  if [[ "${background_path}" != "none" ]]; then
    background_log_entry="${background_log}"
    if ! start_background "${background_path}" "${case_dir}" \
        "${background_json}" "${background_log}" "${case_dir}/background.env"; then
      case_background_rc="${background_rc}"
    fi
  fi
  if [[ -n "${background_pid}" || "${background_path}" == "none" ]]; then
    if env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS -u CUDA_DEVICE_MAX_CONNECTIONS \
        "${victim_cmd[@]}" >"${victim_log}" 2>&1; then
      victim_rc=0
    else
      victim_rc=$?
    fi
  else
    victim_rc=1
  fi
  if [[ "${background_path}" != "none" ]]; then
    stop_background
    case_background_rc="${background_rc}"
  fi
  if [[ -s "${victim_json}" ]]; then victim_entry="${victim_json}"; fi
  if [[ -s "${background_json}" ]]; then background_entry="${background_json}"; fi
  victim_gbps="$(extract_metric "${victim_log}" aggregateGBps)"
  background_gbps="$(extract_metric "${background_log}" aggregateGBps)"
  [[ -n "${victim_gbps}" ]] || victim_gbps="NA"
  [[ -n "${background_gbps}" ]] || background_gbps="NA"
  if [[ "${background_path}" == "none" ]]; then
    case_background_rc=0
    background_gbps="NA"
  fi
  if [[ "${victim_rc}" -eq 0 && "${case_background_rc}" -eq 0 &&
        -s "${victim_json}" ]]; then
    status="pass"
  fi

  printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${background_path}" "${victim_mode}" "${topology}" "${repetition}" \
    "${device_list}" "${size}" "${background_size}" "${target_gbps}" \
    "${duty_cycle}" "${victim_gbps}" "${background_gbps}" "${victim_rc}" \
    "${case_background_rc}" "${status}" "${background_entry}" \
    "${victim_entry}" "${victim_log}" "${background_log_entry}" >> "${summary_file}"
  echo "${status^^}: background=${background_path} victim=${victim_mode} topology=${topology} repetition=${repetition} victimGBps=${victim_gbps} backgroundGBps=${background_gbps}"
}

for background_path in "${background_array[@]}"; do
  for victim_mode in "${victim_array[@]}"; do
    for topology in "${topology_array[@]}"; do
      for ((repetition = 1; repetition <= runs; ++repetition)); do
        run_case "${background_path}" "${victim_mode}" "${topology}" "${repetition}"
      done
    done
  done
done

echo "Completed copy-path ablation: ${output_root}"
echo "Summary: ${summary_file}"
