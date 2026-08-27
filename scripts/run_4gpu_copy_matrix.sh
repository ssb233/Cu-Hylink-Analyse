#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"

device_list="0,1,2,3"
patterns="ring,allpairs"
directions="none,d2h,h2d"
size="255M"
repeats="20"
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_4gpu_copy_matrix.sh [options]

Run the four-GPU CUDA copy matrix. Each pattern is measured with no
background traffic, D2H background traffic, and H2D background traffic.

Options:
  --deviceList=0,1,2,3  visible GPU list
  --patterns=ring,allpairs
  --directions=none,d2h,h2d
  --size=255M           one memcpy size for D2D and background traffic
  --repeats=20          measured D2D repetitions
  --outputRoot=<path>   result directory (default: doc/results/4gpu/<UTC>)
  --help

D2D warmup is fixed at 10 iterations by d2d_multi_peer_bw.
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
    --patterns=*)
      patterns="${argument#*=}"
      shift
      ;;
    --patterns)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      patterns="$2"
      shift 2
      ;;
    --directions=*)
      directions="${argument#*=}"
      shift
      ;;
    --directions)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      directions="$2"
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
    --repeats=*)
      repeats="${argument#*=}"
      shift
      ;;
    --repeats)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      repeats="$2"
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

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/4gpu/${timestamp}-$$"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"

IFS=',' read -r -a pattern_array <<< "${patterns}"
IFS=',' read -r -a direction_array <<< "${directions}"
[[ "${#pattern_array[@]}" -gt 0 && -n "${pattern_array[0]}" ]] || {
  echo "ERROR: --patterns cannot be empty" >&2
  exit 2
}
[[ "${#direction_array[@]}" -gt 0 && -n "${direction_array[0]}" ]] || {
  echo "ERROR: --directions cannot be empty" >&2
  exit 2
}

for index in "${!pattern_array[@]}"; do
  pattern_array[index]="${pattern_array[index]//[[:space:]]/}"
  case "${pattern_array[index]}" in
    ring|allpairs) ;;
    *) echo "ERROR: unsupported pattern: ${pattern_array[index]}" >&2; exit 2 ;;
  esac
done
for index in "${!direction_array[@]}"; do
  direction_array[index]="${direction_array[index]//[[:space:]]/}"
  case "${direction_array[index]}" in
    none|d2h|h2d) ;;
    *) echo "ERROR: unsupported direction: ${direction_array[index]}" >&2; exit 2 ;;
  esac
done

environment_file="${output_root}/environment.txt"
{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'NCCL_HOME=%s\n' "${NCCL_HOME:-unset}"
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvidia-smi nvlink -s ###\n'
  nvidia-smi nvlink -s 2>&1 || true
  printf '\n### nvcc --version ###\n'
  if command -v nvcc >/dev/null 2>&1; then nvcc --version 2>&1 || true; else printf 'nvcc=not-found\n'; fi
} > "${environment_file}"

summary_file="${output_root}/summary.csv"
printf 'pattern,direction,size,repeats,d2dAggregateGBps,backgroundAggregateGBps,status,d2dLog,backgroundLog\n' > "${summary_file}"

extract_last_metric() {
  local file="$1"
  sed -n 's/.*aggregateGBps=\([0-9.+eE-]*\).*/\1/p' "${file}" 2>/dev/null | tail -n 1
}

run_case() {
  local pattern="$1"
  local direction="$2"
  local case_dir="${output_root}/${pattern}_${direction}"
  local d2d_log="${case_dir}/d2d.log"
  local d2d_json="${case_dir}/d2d.json"
  local background_log="${case_dir}/background.log"
  local background_json="${case_dir}/background.json"
  local ready_file="${case_dir}/ready"
  local stop_file="${case_dir}/stop"
  local d2d_rc=0
  local background_case_rc=0
  local d2d_gbps="NA"
  local background_gbps="NA"
  local background_log_entry="NA"
  local status="fail"
  local -a d2d_cmd
  local -a background_cmd

  mkdir -p "${case_dir}"
  rm -f "${ready_file}" "${stop_file}"

  d2d_cmd=(
    "${multi_bin}"
    "--pattern=${pattern}"
    "--deviceList=${device_list}"
    "--size=${size}"
    "--repeats=${repeats}"
    "--output=${d2d_json}"
  )
  {
    printf 'pattern=%s direction=%s\n' "${pattern}" "${direction}"
    printf 'd2d_command='
    printf '%q ' "${d2d_cmd[@]}"
    printf '\n'
  } > "${case_dir}/command.txt"

  if [[ "${direction}" != "none" ]]; then
    background_log_entry="${background_log}"
    [[ -x "${background_bin}" ]] || {
      echo "ERROR: missing executable: ${background_bin}" >&2
      return 1
    }
    background_cmd=(
      "${background_bin}"
      "--devList=${device_list}"
      "--direction=${direction}"
      "--size=${size}"
      "--readyFile=${ready_file}"
      "--stopFile=${stop_file}"
      "--reportSec=2"
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
        wait "${background_pid}" || background_case_rc=$?
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
    background_case_rc=${background_rc}
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
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${pattern}" "${direction}" "${size}" "${repeats}" \
    "${d2d_gbps}" "${background_gbps}" "${status}" \
    "${d2d_log}" "${background_log_entry}" >> "${summary_file}"

  if [[ "${status}" != "pass" ]]; then
    echo "ERROR: failed case ${pattern}/${direction}; see ${case_dir}" >&2
    return 1
  fi
  echo "PASS: ${pattern}/${direction} d2dAggregateGBps=${d2d_gbps} backgroundAggregateGBps=${background_gbps}"
}

for pattern in "${pattern_array[@]}"; do
  for direction in "${direction_array[@]}"; do
    run_case "${pattern}" "${direction}"
  done
done

echo "Completed four-GPU copy matrix: ${output_root}"
echo "Summary: ${summary_file}"
