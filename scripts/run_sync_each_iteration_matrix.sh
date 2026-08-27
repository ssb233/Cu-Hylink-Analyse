#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
output_root="${repo_root}/doc/results/gpu-contention/timing-repeat-diagnostic/sync-each-iteration"
runs=3
size="255M"
background_pid=""
background_stop=""
background_rc=0

usage() {
  cat <<'EOF'
Usage: run_sync_each_iteration_matrix.sh [options]

Run matched asynchronous-batch and per-iteration-synchronize allpairs tests.

Options:
  --runs=3                 repetitions for every matrix point
  --size=255M              one D2D/background memcpy size
  --outputRoot=<path>      fresh output directory
  --help

Matrix: repeats=20,500; scenarios=none,d2h-gpu0,d2h-all;
        modes=async,sync; D2D chunk timing is disabled.
Warmup is fixed at 10 iterations by d2d_multi_peer_bw.
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
  kill -INT "${background_pid}" 2>/dev/null || true
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
    --runs=*)
      runs="${argument#*=}"
      shift
      ;;
    --runs)
      [[ $# -ge 2 ]] || { echo "ERROR: --runs requires a value" >&2; exit 2; }
      runs="$2"
      shift 2
      ;;
    --size=*)
      size="${argument#*=}"
      shift
      ;;
    --size)
      [[ $# -ge 2 ]] || { echo "ERROR: --size requires a value" >&2; exit 2; }
      size="$2"
      shift 2
      ;;
    --outputRoot=*)
      output_root="${argument#*=}"
      shift
      ;;
    --outputRoot)
      [[ $# -ge 2 ]] || { echo "ERROR: --outputRoot requires a value" >&2; exit 2; }
      output_root="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
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
  exit 1
}
[[ "${runs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --runs must be a positive integer" >&2
  exit 2
}

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
  printf 'size=%s\nruns=%s\n' "${size}" "${runs}"
  printf '\n### nvidia-smi -L ###\n'
  nvidia-smi -L 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvidia-smi nvlink -s ###\n'
  nvidia-smi nvlink -s 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'mode,repeats,repetition,scenario,d2dAggregateGBps,backgroundAggregateGBps,d2dExit,backgroundExit,d2dJson,backgroundJson\n' > "${summary_file}"

extract_last_metric() {
  local file="$1"
  sed -n 's/.*aggregateGBps=\([0-9.+eE-]*\).*/\1/p' "${file}" 2>/dev/null | tail -n 1
}

run_case() {
  local mode="$1"
  local repeats="$2"
  local repetition="$3"
  local scenario="$4"
  local direction="none"
  local devices=""
  local sync_flag=0
  local case_dir=""
  local ready_file=""
  local stop_file=""
  local d2d_json=""
  local d2d_log=""
  local background_json=""
  local background_log=""
  local d2d_rc=0
  local d2d_gbps="NA"
  local background_gbps="NA"

  case "${scenario}" in
    none) ;;
    d2h-gpu0)
      direction="d2h"
      devices="0"
      ;;
    d2h-all)
      direction="d2h"
      devices="0,1,2,3"
      ;;
    *)
      echo "ERROR: unsupported scenario: ${scenario}" >&2
      return 2
      ;;
  esac
  if [[ "${mode}" == "sync" ]]; then
    sync_flag=1
  fi

  case_dir="${output_root}/repeats-${repeats}/repetition-${repetition}/${scenario}/${mode}"
  ready_file="${case_dir}/ready"
  stop_file="${case_dir}/stop"
  d2d_json="${case_dir}/d2d.json"
  d2d_log="${case_dir}/d2d.log"
  background_json="${case_dir}/background.json"
  background_log="${case_dir}/background.log"
  mkdir -p "${case_dir}"
  {
    printf 'mode=%s\nrepeats=%s\nrepetition=%s\nscenario=%s\ndirection=%s\ndevices=%s\n' \
      "${mode}" "${repeats}" "${repetition}" "${scenario}" "${direction}" "${devices}"
    printf 'd2d_command=%q ' "${multi_bin}" --pattern=allpairs --deviceList=0,1,2,3 \
      "--size=${size}" "--repeats=${repeats}" "--syncEachIteration=${sync_flag}" \
      "--output=${d2d_json}"
    printf '\n'
  } > "${case_dir}/command.txt"

  if [[ "${direction}" != "none" ]]; then
    rm -f "${ready_file}" "${stop_file}"
    "${background_bin}" "--devList=${devices}" "--direction=${direction}" \
      "--size=${size}" "--readyFile=${ready_file}" "--stopFile=${stop_file}" \
      --reportSec=1 "--output=${background_json}" > "${background_log}" 2>&1 &
    background_pid=$!
    background_stop="${stop_file}"
    for _ in $(seq 1 300); do
      [[ -f "${ready_file}" ]] && break
      if ! kill -0 "${background_pid}" 2>/dev/null; then
        wait "${background_pid}" || true
        background_pid=""
        background_stop=""
        echo "ERROR: background exited before ready: ${case_dir}" >&2
        return 1
      fi
      sleep 0.1
    done
    if [[ ! -f "${ready_file}" ]]; then
      echo "ERROR: background did not become ready: ${case_dir}" >&2
      return 1
    fi
  fi

  printf 'benchmark_start_epoch_ns=%s\n' "$(date +%s%N)" > "${case_dir}/timestamps.txt"
  if "${multi_bin}" --pattern=allpairs --deviceList=0,1,2,3 "--size=${size}" \
      "--repeats=${repeats}" "--syncEachIteration=${sync_flag}" \
      "--output=${d2d_json}" > "${d2d_log}" 2>&1; then
    d2d_rc=0
  else
    d2d_rc=$?
  fi
  printf 'benchmark_end_epoch_ns=%s\nd2d_exit=%s\n' "$(date +%s%N)" "${d2d_rc}" >> "${case_dir}/timestamps.txt"

  if [[ "${direction}" != "none" ]]; then
    stop_background
    printf 'background_exit=%s\n' "${background_rc}" >> "${case_dir}/timestamps.txt"
  else
    printf 'background_exit=0\n' >> "${case_dir}/timestamps.txt"
  fi

  d2d_gbps="$(extract_last_metric "${d2d_log}" || true)"
  [[ -n "${d2d_gbps}" ]] || d2d_gbps="NA"
  if [[ "${direction}" != "none" ]]; then
    background_gbps="$(extract_last_metric "${background_log}" || true)"
    [[ -n "${background_gbps}" ]] || background_gbps="NA"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${mode}" "${repeats}" "${repetition}" "${scenario}" "${d2d_gbps}" \
    "${background_gbps}" "${d2d_rc}" "${background_rc}" "${d2d_json}" "${background_json}" \
    >> "${summary_file}"
  if [[ "${d2d_rc}" -ne 0 || "${background_rc}" -ne 0 ]]; then
    echo "FAIL: mode=${mode} repeats=${repeats} repetition=${repetition} scenario=${scenario}" >&2
    return 1
  fi
  echo "PASS: mode=${mode} repeats=${repeats} repetition=${repetition} scenario=${scenario} d2d=${d2d_gbps} background=${background_gbps}"
}

for repeats in 20 500; do
  for repetition in $(seq 1 "${runs}"); do
    for scenario in none d2h-gpu0 d2h-all; do
      for mode in async sync; do
        run_case "${mode}" "${repeats}" "${repetition}" "${scenario}"
      done
    done
  done
done

echo "Completed sync-each-iteration matrix: ${output_root}"
echo "Summary: ${summary_file}"
