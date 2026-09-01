#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
d2d_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
same_process_bin="${repo_root}/build/cuda_copy/d2d_with_background_bw"

device_list="0,1,2"
context_modes="two-process,same-process"
topologies="single-two-copy,edge-independent"
scenarios="none,d2h-all"
runs=3
repeats=300
size="255M"
background_size="255M"
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_context_boundary_diagnostic.sh [options]

Compare the two-process baseline with D2H workers in the same process and
primary CUDA contexts. The current three-GPU adaptation uses one source
stream with two consecutive copies and independent edge streams.

Options:
  --deviceList=0,1,2
  --contextModes=two-process,same-process
  --contextMode=two-process|same-process   singular alias
  --topologies=single-two-copy,edge-independent
  --scenarios=none,d2h-all
  --runs=3                  repetitions for every matrix point
  --repeats=300             measured D2D repetitions
  --size=255M               D2D size
  --backgroundSize=255M     D2H size
  --outputRoot=<path>       fresh result directory
  --help

default=two-process. Warmup is fixed at 10 iterations.
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
    --contextModes=*)
      context_modes="${argument#*=}"
      shift
      ;;
    --contextModes)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      context_modes="$2"
      shift 2
      ;;
    --contextMode=*)
      context_modes="${argument#*=}"
      shift
      ;;
    --contextMode)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      context_modes="$2"
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
    --scenarios=*)
      scenarios="${argument#*=}"
      shift
      ;;
    --scenarios)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      scenarios="$2"
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
      background_size="${argument#*=}"
      shift
      ;;
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      size="$2"
      background_size="$2"
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
  echo "ERROR: context-boundary diagnostic requires exactly three devices" >&2
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

IFS=',' read -r -a context_array <<< "${context_modes}"
[[ "${#context_array[@]}" -gt 0 && -n "${context_array[0]}" ]] || {
  echo "ERROR: --contextModes cannot be empty" >&2
  exit 2
}
for index in "${!context_array[@]}"; do
  value="${context_array[index]//[[:space:]]/}"
  case "${value}" in
    two-process|same-process) ;;
    *) echo "ERROR: unsupported context mode: ${value}" >&2; exit 2 ;;
  esac
  context_array[index]="${value}"
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

IFS=',' read -r -a scenario_array <<< "${scenarios}"
[[ "${#scenario_array[@]}" -gt 0 && -n "${scenario_array[0]}" ]] || {
  echo "ERROR: --scenarios cannot be empty" >&2
  exit 2
}
for index in "${!scenario_array[@]}"; do
  value="${scenario_array[index]//[[:space:]]/}"
  case "${value}" in
    none|d2h-all) ;;
    *) echo "ERROR: unsupported scenario: ${value}" >&2; exit 2 ;;
  esac
  scenario_array[index]="${value}"
done

[[ -x "${d2d_bin}" ]] || {
  echo "ERROR: missing executable: ${d2d_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${background_bin}" ]] || {
  echo "ERROR: missing executable: ${background_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${same_process_bin}" ]] || {
  echo "ERROR: missing executable: ${same_process_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/work-queue-channel/context-boundary/${timestamp}-$$"
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
  printf 'deviceList=%s\ncontextModes=%s\ntopologies=%s\nscenarios=%s\nruns=%s\nrepeats=%s\nd2dSize=%s\nbackgroundSize=%s\n' \
    "${device_list}" "${context_modes}" "${topologies}" "${scenarios}" \
    "${runs}" "${repeats}" "${size}" "${background_size}"
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

printf 'contextMode,topology,scenario,repetition,deviceList,d2dSize,backgroundSize,d2dAggregateGBps,backgroundAggregateGBps,d2dExit,backgroundExit,status,resultJson,d2dJson,backgroundJson,d2dLog,backgroundLog\n' > "${summary_file}"

extract_metric() {
  local file="$1"
  local metric="$2"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
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

start_two_process_background() {
  local case_dir="$1"
  local background_json="$2"
  local background_log="$3"
  local background_env_file="$4"
  background_stop="${case_dir}/stop"
  background_pid=""
  background_rc=0
  touch "${background_stop}"
  rm -f "${case_dir}/ready" "${background_stop}"
  background_cmd=(
    "${background_bin}"
    "--devList=${device_list}"
    --direction=d2h
    "--size=${background_size}"
    "--readyFile=${case_dir}/ready"
    "--stopFile=${background_stop}"
    "--output=${background_json}"
  )
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
  echo "ERROR: background did not become ready" >&2
  stop_background
  return 1
}

run_case() {
  local context_mode="$1"
  local topology="$2"
  local scenario="$3"
  local repetition="$4"
  local case_dir="${output_root}/context-${context_mode}/topology-${topology}/scenario-${scenario}/repetition-${repetition}"
  local d2d_json="${case_dir}/d2d.json"
  local background_json="${case_dir}/background.json"
  local result_json="${case_dir}/result.json"
  local d2d_log="${case_dir}/d2d.log"
  local background_log="${case_dir}/background.log"
  local d2d_rc=0
  local case_background_rc=0
  local d2d_gbps="NA"
  local background_gbps="NA"
  local status="fail"
  local result_entry="NA"
  local d2d_entry="NA"
  local background_entry="NA"
  local background_log_entry="NA"
  local -a topology_args
  local -a d2d_cmd
  local -a same_cmd

  mkdir -p "${case_dir}"
  topology_args_for "${topology}"
  {
    printf 'contextMode=%s topology=%s scenario=%s repetition=%s\n' \
      "${context_mode}" "${topology}" "${scenario}" "${repetition}"
    printf 'deviceList=%s d2dSize=%s backgroundSize=%s repeats=%s\n' \
      "${device_list}" "${size}" "${background_size}" "${repeats}"
  } > "${case_dir}/environment.txt"

  if [[ "${context_mode}" == "same-process" ]]; then
    local background_mode="none"
    if [[ "${scenario}" == "d2h-all" ]]; then background_mode="d2h"; fi
    same_cmd=(
      "${same_process_bin}"
      "--deviceList=${device_list}"
      "--topology=${topology}"
      "--background=${background_mode}"
      "--size=${size}"
      "--backgroundSize=${background_size}"
      "--repeats=${repeats}"
      "--output=${result_json}"
    )
    printf 'command=' > "${case_dir}/command.txt"
    printf '%q ' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
      -u CUDA_DEVICE_MAX_CONNECTIONS "${same_cmd[@]}" >> "${case_dir}/command.txt"
    printf '\n' >> "${case_dir}/command.txt"
    if env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS -u CUDA_DEVICE_MAX_CONNECTIONS \
        "${same_cmd[@]}" >"${d2d_log}" 2>&1; then
      d2d_rc=0
    else
      d2d_rc=$?
    fi
    if [[ -s "${result_json}" ]]; then
      result_entry="${result_json}"
      d2d_entry="${result_json}"
      background_entry="${result_json}"
    fi
    d2d_gbps="$(extract_metric "${d2d_log}" aggregateGBps)"
    background_gbps="$(extract_metric "${d2d_log}" backgroundAggregateGBps)"
    [[ -n "${d2d_gbps}" ]] || d2d_gbps="NA"
    [[ -n "${background_gbps}" ]] || background_gbps="NA"
    if [[ "${scenario}" == "none" ]]; then background_gbps="0.000"; fi
    if [[ "${d2d_rc}" -eq 0 && -s "${result_json}" ]]; then status="pass"; fi
  else
    d2d_cmd=(
      "${d2d_bin}"
      --pattern=allpairs
      "--deviceList=${device_list}"
      --edgeOrder=source-major
      "${topology_args[@]}"
      "--size=${size}"
      "--repeats=${repeats}"
      "--output=${d2d_json}"
    )
    printf 'command=' > "${case_dir}/command.txt"
    printf '%q ' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
      -u CUDA_DEVICE_MAX_CONNECTIONS "${d2d_cmd[@]}" >> "${case_dir}/command.txt"
    printf '\n' >> "${case_dir}/command.txt"
    printf '%q\n' env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS \
      -u CUDA_DEVICE_MAX_CONNECTIONS "${d2d_cmd[@]}" > "${case_dir}/d2d.env"

    if [[ "${scenario}" == "d2h-all" ]]; then
      background_log_entry="${background_log}"
      if ! start_two_process_background "${case_dir}" "${background_json}" \
          "${background_log}" "${case_dir}/background.env"; then
        case_background_rc="${background_rc}"
      fi
    fi
    if [[ -z "${background_pid}" && "${scenario}" == "d2h-all" ]]; then
      d2d_rc=1
    elif env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS -u CUDA_DEVICE_MAX_CONNECTIONS \
        "${d2d_cmd[@]}" >"${d2d_log}" 2>&1; then
      d2d_rc=0
    else
      d2d_rc=$?
    fi
    if [[ "${scenario}" == "d2h-all" ]]; then
      stop_background
      case_background_rc="${background_rc}"
    fi
    if [[ -s "${d2d_json}" ]]; then d2d_entry="${d2d_json}"; fi
    if [[ -s "${background_json}" ]]; then background_entry="${background_json}"; fi
    d2d_gbps="$(extract_metric "${d2d_log}" aggregateGBps)"
    background_gbps="$(extract_metric "${background_log}" aggregateGBps)"
    [[ -n "${d2d_gbps}" ]] || d2d_gbps="NA"
    [[ -n "${background_gbps}" ]] || background_gbps="NA"
    if [[ "${scenario}" == "none" ]]; then
      case_background_rc=0
      background_gbps="NA"
    fi
    if [[ "${d2d_rc}" -eq 0 && "${case_background_rc}" -eq 0 &&
          -s "${d2d_json}" ]]; then
      status="pass"
    fi
  fi

  printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${context_mode}" "${topology}" "${scenario}" "${repetition}" \
    "${device_list}" "${size}" "${background_size}" "${d2d_gbps}" \
    "${background_gbps}" "${d2d_rc}" "${case_background_rc}" "${status}" \
    "${result_entry}" "${d2d_entry}" "${background_entry}" "${d2d_log}" \
    "${background_log_entry}" >> "${summary_file}"
  echo "${status^^}: context=${context_mode} topology=${topology} scenario=${scenario} repetition=${repetition} d2dAggregateGBps=${d2d_gbps} backgroundAggregateGBps=${background_gbps}"
}

for context_mode in "${context_array[@]}"; do
  for topology in "${topology_array[@]}"; do
    for scenario in "${scenario_array[@]}"; do
      for ((repetition = 1; repetition <= runs; ++repetition)); do
        run_case "${context_mode}" "${topology}" "${scenario}" "${repetition}"
      done
    done
  done
done

echo "Completed context-boundary diagnostic: ${output_root}"
echo "Summary: ${summary_file}"
