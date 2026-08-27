#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"

device_list="0,1,2,3"
edge_order="source-major"
assignments="0,1,1:1,0,1:1,1,0"
repeats_list="20,300"
scenarios="none,d2h-all"
runs=3
d2d_size="255M"
background_size="255M"
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_stream_assignment_matrix.sh [options]

Run the two-stream one-vs-two copy assignment ablation for four-GPU allpairs
D2D traffic. Each assignment is indexed by the source's three source-major
edge positions; 0 and 1 are the two stream slots. Use ':' between assignments.

Options:
  --deviceList=0,1,2,3
  --edgeOrder=source-major
  --assignments=0,1,1:1,0,1:1,1,0
  --repeatsList=20,300
  --scenarios=none,d2h-all
  --runs=3                  repetitions for every matrix point
  --size=255M               set both D2D and background memcpy sizes
  --d2dSize=255M            one D2D memcpy size
  --backgroundSize=255M     one background memcpy size
  --outputRoot=<path>       fresh result directory
  --help

All cases use streamMode=per-source, streamsPerSource=2,
streamDependency=none, and edgePermutation=identity. D2D warmup is fixed at
10 iterations.
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
    --edgeOrder=*)
      edge_order="${argument#*=}"
      shift
      ;;
    --edgeOrder)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      edge_order="$2"
      shift 2
      ;;
    --assignments=*)
      assignments="${argument#*=}"
      shift
      ;;
    --assignments)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      assignments="$2"
      shift 2
      ;;
    --repeatsList=*)
      repeats_list="${argument#*=}"
      shift
      ;;
    --repeatsList)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      repeats_list="$2"
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
    --size=*)
      d2d_size="${argument#*=}"
      background_size="${argument#*=}"
      shift
      ;;
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      d2d_size="$2"
      background_size="$2"
      shift 2
      ;;
    --d2dSize=*)
      d2d_size="${argument#*=}"
      shift
      ;;
    --d2dSize)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      d2d_size="$2"
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

[[ -x "${multi_bin}" ]] || {
  echo "ERROR: missing executable: ${multi_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ -x "${background_bin}" ]] || {
  echo "ERROR: missing executable: ${background_bin}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
}
[[ "${runs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --runs must be a positive integer" >&2
  exit 2
}
[[ "${edge_order}" == "source-major" ]] || {
  echo "ERROR: stream assignment ablation requires --edgeOrder=source-major" >&2
  exit 2
}

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 4 ]] || {
  echo "ERROR: stream assignment ablation requires exactly four devices" >&2
  exit 2
}
for index in "${!device_array[@]}"; do
  device_array[index]="${device_array[index]//[[:space:]]/}"
  [[ "${device_array[index]}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid device index: ${device_array[index]}" >&2
    exit 2
  }
done

IFS=':' read -r -a assignment_array <<< "${assignments}"
IFS=',' read -r -a repeats_array <<< "${repeats_list}"
IFS=',' read -r -a scenario_array <<< "${scenarios}"
[[ "${#assignment_array[@]}" -gt 0 && -n "${assignment_array[0]}" ]] || {
  echo "ERROR: --assignments cannot be empty" >&2
  exit 2
}
[[ "${#repeats_array[@]}" -gt 0 && -n "${repeats_array[0]}" ]] || {
  echo "ERROR: --repeatsList cannot be empty" >&2
  exit 2
}
[[ "${#scenario_array[@]}" -gt 0 && -n "${scenario_array[0]}" ]] || {
  echo "ERROR: --scenarios cannot be empty" >&2
  exit 2
}

for index in "${!assignment_array[@]}"; do
  assignment_array[index]="${assignment_array[index]//[[:space:]]/}"
  case "${assignment_array[index]}" in
    0,1,1|1,0,1|1,1,0) ;;
    *)
      echo "ERROR: assignments must be 0,1,1, 1,0,1, or 1,1,0: ${assignment_array[index]}" >&2
      exit 2
      ;;
  esac
done
for index in "${!repeats_array[@]}"; do
  repeats_array[index]="${repeats_array[index]//[[:space:]]/}"
  [[ "${repeats_array[index]}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: repeats must be positive integers: ${repeats_array[index]}" >&2
    exit 2
  }
done
for index in "${!scenario_array[@]}"; do
  scenario_array[index]="${scenario_array[index]//[[:space:]]/}"
  case "${scenario_array[index]}" in
    none|d2h-all) ;;
    *)
      echo "ERROR: unsupported scenario: ${scenario_array[index]}" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/queue-phase-diagnostic/stream-assignment/${timestamp}-$$"
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
  printf 'device_list=%s\nedge_order=%s\nassignments=%s\nrepeats_list=%s\nscenarios=%s\nruns=%s\nd2dSize=%s\nbackgroundSize=%s\n' \
    "${device_list}" "${edge_order}" "${assignments}" "${repeats_list}" \
    "${scenarios}" "${runs}" "${d2d_size}" "${background_size}"
  printf 'stream_mode=per-source\nstreams_per_source=2\nstream_dependency=none\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvidia-smi nvlink -s ###\n'
  nvidia-smi nvlink -s 2>&1 || true
  printf '\n### nvcc --version ###\n'
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>&1 || true
  else
    printf 'nvcc=not-found\n'
  fi
} > "${output_root}/environment.txt"

printf 'streamAssignment,assignmentId,repeats,repetition,scenario,d2dSize,backgroundSize,d2dAggregateGBps,backgroundAggregateGBps,d2dExit,backgroundExit,status,d2dJson,backgroundJson,d2dLog,backgroundLog\n' > "${summary_file}"

extract_last_metric() {
  local file="$1"
  sed -n 's/.*aggregateGBps=\([0-9.+eE-]*\).*/\1/p' "${file}" 2>/dev/null | tail -n 1
}

run_case() {
  local assignment="$1"
  local repeats="$2"
  local repetition="$3"
  local scenario="$4"
  local direction="none"
  local assignment_id="${assignment//,/}"
  local case_dir="${output_root}/assignment-${assignment_id}/repeats-${repeats}/repetition-${repetition}/${scenario}"
  local ready_file="${case_dir}/ready"
  local stop_file="${case_dir}/stop"
  local d2d_json="${case_dir}/d2d.json"
  local d2d_log="${case_dir}/d2d.log"
  local background_json="${case_dir}/background.json"
  local background_log="${case_dir}/background.log"
  local d2d_rc=0
  local background_case_rc=0
  local d2d_gbps="NA"
  local background_gbps="NA"
  local background_log_entry="NA"
  local status="fail"
  local -a d2d_cmd
  local -a background_cmd

  case "${scenario}" in
    none) ;;
    d2h-all) direction="d2h" ;;
    *)
      echo "ERROR: unsupported scenario: ${scenario}" >&2
      return 2
      ;;
  esac

  mkdir -p "${case_dir}"
  rm -f "${ready_file}" "${stop_file}"
  d2d_cmd=(
    "${multi_bin}"
    --pattern=allpairs
    "--deviceList=${device_list}"
    --edgeOrder=source-major
    --streamMode=per-source
    --streamsPerSource=2
    --streamDependency=none
    "--streamAssignment=${assignment}"
    "--size=${d2d_size}"
    "--repeats=${repeats}"
    "--output=${d2d_json}"
  )
  {
    printf 'streamAssignment=%s assignmentId=%s repeats=%s repetition=%s scenario=%s\n' \
      "${assignment}" "${assignment_id}" "${repeats}" "${repetition}" "${scenario}"
    printf 'd2d_command='
    printf '%q ' "${d2d_cmd[@]}"
    printf '\n'
  } > "${case_dir}/command.txt"

  if [[ "${direction}" != "none" ]]; then
    background_log_entry="${background_log}"
    background_cmd=(
      "${background_bin}"
      "--devList=${device_list}"
      "--direction=${direction}"
      "--size=${background_size}"
      "--readyFile=${ready_file}"
      "--stopFile=${stop_file}"
      --reportSec=1
      "--output=${background_json}"
    )
    {
      printf 'background_command='
      printf '%q ' "${background_cmd[@]}"
      printf '\n'
    } >> "${case_dir}/command.txt"
    "${background_cmd[@]}" > "${background_log}" 2>&1 &
    background_pid=$!
    background_stop="${stop_file}"
    for _ in $(seq 1 300); do
      [[ -f "${ready_file}" ]] && break
      if ! kill -0 "${background_pid}" 2>/dev/null; then
        wait "${background_pid}" || true
        background_pid=""
        background_stop=""
        echo "ERROR: background process exited before ready: ${case_dir}" >&2
        return 1
      fi
      sleep 0.1
    done
    if [[ ! -f "${ready_file}" ]]; then
      echo "ERROR: background process did not become ready: ${case_dir}" >&2
      return 1
    fi
  fi

  if "${d2d_cmd[@]}" > "${d2d_log}" 2>&1; then
    d2d_rc=0
  else
    d2d_rc=$?
  fi
  if [[ "${direction}" != "none" ]]; then
    stop_background
    background_case_rc="${background_rc}"
  fi

  d2d_gbps="$(extract_last_metric "${d2d_log}" || true)"
  [[ -n "${d2d_gbps}" ]] || d2d_gbps="NA"
  if [[ "${direction}" != "none" ]]; then
    background_gbps="$(extract_last_metric "${background_log}" || true)"
    [[ -n "${background_gbps}" ]] || background_gbps="NA"
  fi

  if [[ "${d2d_rc}" -eq 0 && "${background_case_rc}" -eq 0 ]]; then
    status="pass"
  fi
  printf '"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${assignment}" "${assignment_id}" "${repeats}" "${repetition}" \
    "${scenario}" "${d2d_size}" "${background_size}" "${d2d_gbps}" "${background_gbps}" \
    "${d2d_rc}" "${background_case_rc}" "${status}" "${d2d_json}" \
    "${background_json}" "${d2d_log}" "${background_log_entry}" >> "${summary_file}"

  if [[ "${status}" != "pass" ]]; then
    echo "ERROR: failed case assignment=${assignment} repeats=${repeats} scenario=${scenario}; see ${case_dir}" >&2
    return 1
  fi
  echo "PASS: streamAssignment=${assignment} repeats=${repeats} scenario=${scenario} d2dAggregateGBps=${d2d_gbps} backgroundAggregateGBps=${background_gbps}"
}

for assignment in "${assignment_array[@]}"; do
  for repeats in "${repeats_array[@]}"; do
    for repetition in $(seq 1 "${runs}"); do
      for scenario in "${scenario_array[@]}"; do
        run_case "${assignment}" "${repeats}" "${repetition}" "${scenario}"
      done
    done
  done
done

echo "Completed two-stream assignment matrix: ${output_root}"
echo "Summary: ${summary_file}"
