#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
trace_lib="${repo_root}/build/cuda_copy/libcupti_memcpy_channel_trace.so"

device_list="0,1,2"
connection_values="unset,1,2,4,8,16,32"
topologies="single-two-copy,edge-independent,edge-source-chain"
scenarios="none,d2h-all"
runs=3
repeats=300
size="255M"
background_size="255M"
trace_enabled=0
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_work_queue_channel_diagnostic.sh [options]

Run the three-GPU Stage F work-queue/HW-channel diagnostic. The three-GPU
allpairs adaptation uses one source stream with two consecutive copies as the
victim topology; the other topologies use one active stream per outgoing edge.

Options:
  --deviceList=0,1,2
  --connectionValues=unset,1,2,4,8,16,32
  --topologies=single-two-copy,edge-independent,edge-source-chain
  --scenarios=none,d2h-all
  --runs=3                  repetitions for every matrix point
  --repeats=300             D2D measured repetitions
  --size=255M               D2D and background memcpy size
  --backgroundSize=255M     background memcpy size override
  --trace=0|1               preload CUPTI memcpy channel tracer
  --outputRoot=<path>       fresh result directory
  --help

Warmup is fixed at 10 iterations by both CUDA benchmarks.
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
    --connectionValues=*)
      connection_values="${argument#*=}"
      shift
      ;;
    --connectionValues)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      connection_values="$2"
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
    --trace=*)
      trace_enabled="${argument#*=}"
      shift
      ;;
    --trace)
      [[ $# -ge 2 ]] || { echo "ERROR: ${argument} requires a value" >&2; exit 2; }
      trace_enabled="$2"
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
case "${trace_enabled}" in
  0|1) ;;
  *) echo "ERROR: --trace must be 0 or 1" >&2; exit 2 ;;
esac

IFS=',' read -r -a device_array <<< "${device_list}"
[[ "${#device_array[@]}" -eq 3 ]] || {
  echo "ERROR: Stage F three-GPU adaptation requires exactly three devices" >&2
  exit 2
}
canonical_devices=""
for index in "${!device_array[@]}"; do
  device_array[index]="${device_array[index]//[[:space:]]/}"
  [[ "${device_array[index]}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid device index: ${device_array[index]}" >&2
    exit 2
  }
  for previous in $(seq 0 $((index - 1))); do
    if [[ "${device_array[index]}" == "${device_array[previous]}" ]]; then
      echo "ERROR: duplicate device index: ${device_array[index]}" >&2
      exit 2
    fi
  done
  if [[ -n "${canonical_devices}" ]]; then canonical_devices+=","; fi
  canonical_devices+="${device_array[index]}"
done
device_list="${canonical_devices}"

IFS=',' read -r -a connection_array <<< "${connection_values}"
[[ "${#connection_array[@]}" -gt 0 && -n "${connection_array[0]}" ]] || {
  echo "ERROR: --connectionValues cannot be empty" >&2
  exit 2
}
for index in "${!connection_array[@]}"; do
  value="${connection_array[index]//[[:space:]]/}"
  case "${value}" in
    unset|1|2|4|8|16|32) ;;
    *) echo "ERROR: unsupported CUDA_DEVICE_MAX_CONNECTIONS value: ${value}" >&2; exit 2 ;;
  esac
  connection_array[index]="${value}"
done

IFS=',' read -r -a topology_array <<< "${topologies}"
[[ "${#topology_array[@]}" -gt 0 && -n "${topology_array[0]}" ]] || {
  echo "ERROR: --topologies cannot be empty" >&2
  exit 2
}
for index in "${!topology_array[@]}"; do
  value="${topology_array[index]//[[:space:]]/}"
  case "${value}" in
    single-two-copy|edge-independent|edge-source-chain) ;;
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
if [[ "${trace_enabled}" == "1" && ! -x "${trace_lib}" ]]; then
  echo "ERROR: missing CUPTI tracer: ${trace_lib}" >&2
  echo "Run scripts/build_cuda_copy.sh first." >&2
  exit 1
fi

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/work-queue-channel/${timestamp}-$$"
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
  printf 'deviceList=%s\nconnectionValues=%s\ntopologies=%s\nscenarios=%s\nruns=%s\nrepeats=%s\nd2dSize=%s\nbackgroundSize=%s\ntrace=%s\n' \
    "${device_list}" "${connection_values}" "${topologies}" "${scenarios}" \
    "${runs}" "${repeats}" "${size}" "${background_size}" "${trace_enabled}"
  printf 'three_gpu_topology_adaptation=single-two-copy means one source stream with two consecutive allpairs copies; edge-independent and edge-source-chain have two active edge streams per source\n'
  printf '\n### command_line ###\n'
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n\n### nvidia-smi ###\n'
  nvidia-smi 2>&1 || true
  printf '\n### nvidia-smi topo -m ###\n'
  nvidia-smi topo -m 2>&1 || true
  printf '\n### nvidia-smi nvlink -s ###\n'
  nvidia-smi nvlink -s 2>&1 || true
  printf '\n### nvcc --version ###\n'
  nvcc --version 2>&1 || true
} > "${output_root}/environment.txt"

printf 'topology,connectionValue,scenario,repetition,deviceList,d2dSize,backgroundSize,d2dAggregateGBps,backgroundAggregateGBps,d2dExit,backgroundExit,status,d2dJson,backgroundJson,d2dLog,backgroundLog,traceD2d,traceBackground\n' > "${summary_file}"

extract_last_metric() {
  local file="$1"
  sed -n 's/.*aggregateGBps=\([0-9.+eE-]*\).*/\1/p' "${file}" 2>/dev/null | tail -n 1
}

connection_env_prefix() {
  local value="$1"
  connection_env=(env -u CUDA_DEVICE_MAX_COPY_CONNECTIONS)
  if [[ "${value}" == "unset" ]]; then
    connection_env+=(-u CUDA_DEVICE_MAX_CONNECTIONS)
  else
    connection_env+=("CUDA_DEVICE_MAX_CONNECTIONS=${value}")
  fi
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
    edge-source-chain)
      topology_args=(--streamMode=per-edge --streamsPerSource=3 --streamDependency=source-chain)
      ;;
  esac
}

trace_env_for() {
  local output="$1"
  if [[ "${trace_enabled}" == "1" ]]; then
    local inherited_preload="${LD_PRELOAD:-}"
    local preload="${trace_lib}"
    if [[ -n "${inherited_preload}" ]]; then preload+=":${inherited_preload}"; fi
    trace_env=("COPYBENCH_CUPTI_MEMCPY_OUTPUT=${output}" "LD_PRELOAD=${preload}")
  else
    trace_env=()
  fi
}

run_case() {
  local topology="$1"
  local connection_value="$2"
  local scenario="$3"
  local repetition="$4"
  local case_dir="${output_root}/topology-${topology}/connection-${connection_value}/scenario-${scenario}/repetition-${repetition}"
  local ready_file="${case_dir}/ready"
  local stop_file="${case_dir}/stop"
  local d2d_json="${case_dir}/d2d.json"
  local d2d_log="${case_dir}/d2d.log"
  local background_json="${case_dir}/background.json"
  local background_log="${case_dir}/background.log"
  local d2d_env_file="${case_dir}/d2d.env"
  local background_env_file="${case_dir}/background.env"
  local trace_d2d="${case_dir}/d2d.memcpy.csv"
  local trace_background="${case_dir}/background.memcpy.csv"
  local d2d_rc=0
  local background_case_rc=0
  local d2d_gbps="NA"
  local background_gbps="NA"
  local background_log_entry="NA"
  local background_json_entry="NA"
  local trace_d2d_entry="NA"
  local trace_background_entry="NA"
  local status="fail"
  local direction="none"
  local -a connection_env
  local -a topology_args
  local -a trace_env
  local -a d2d_cmd
  local -a background_cmd

  topology_args_for "${topology}"
  connection_env_prefix "${connection_value}"
  mkdir -p "${case_dir}"
  rm -f "${ready_file}" "${stop_file}" "${trace_d2d}" "${trace_background}"

  d2d_cmd=(
    "${multi_bin}"
    --pattern=allpairs
    "--deviceList=${device_list}"
    --edgeOrder=source-major
    "${topology_args[@]}"
    "--size=${size}"
    "--repeats=${repeats}"
    "--output=${d2d_json}"
  )
  trace_env_for "${trace_d2d}"
  {
    printf 'topology=%s connectionValue=%s scenario=%s repetition=%s\n' \
      "${topology}" "${connection_value}" "${scenario}" "${repetition}"
    printf 'd2d_environment='; printf '%q ' "${connection_env[@]}"; printf '\n'
    printf 'd2d_command='; printf '%q ' "${d2d_cmd[@]}"; printf '\n'
  } > "${case_dir}/command.txt"
  printf '%q\n' "${connection_env[@]}" "${trace_env[@]}" > "${d2d_env_file}"

  case "${scenario}" in
    none) ;;
    d2h-all) direction="d2h" ;;
  esac

  if [[ "${direction}" != "none" ]]; then
    background_log_entry="${background_log}"
    background_json_entry="${background_json}"
    trace_env_for "${trace_background}"
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
      printf 'background_environment='; printf '%q ' "${connection_env[@]}"; printf '\n'
      printf 'background_command='; printf '%q ' "${background_cmd[@]}"; printf '\n'
    } >> "${case_dir}/command.txt"
    printf '%q\n' "${connection_env[@]}" "${trace_env[@]}" > "${background_env_file}"
    "${connection_env[@]}" "${trace_env[@]}" "${background_cmd[@]}" > "${background_log}" 2>&1 &
    background_pid=$!
    background_stop="${stop_file}"
    for wait_index in $(seq 1 300); do
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

  trace_env_for "${trace_d2d}"
  if "${connection_env[@]}" "${trace_env[@]}" "${d2d_cmd[@]}" > "${d2d_log}" 2>&1; then
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
  if [[ "${trace_enabled}" == "1" ]]; then
    trace_d2d_entry="${trace_d2d}"
    if [[ "${direction}" != "none" ]]; then trace_background_entry="${trace_background}"; fi
  fi

  if [[ "${d2d_rc}" -eq 0 && "${background_case_rc}" -eq 0 ]]; then
    status="pass"
  fi
  printf '%s,%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${topology}" "${connection_value}" "${scenario}" "${repetition}" \
    "${device_list}" "${size}" "${background_size}" "${d2d_gbps}" "${background_gbps}" \
    "${d2d_rc}" "${background_case_rc}" "${status}" "${d2d_json}" \
    "${background_json_entry}" "${d2d_log}" "${background_log_entry}" \
    "${trace_d2d_entry}" "${trace_background_entry}" >> "${summary_file}"

  if [[ "${status}" != "pass" ]]; then
    echo "ERROR: failed Stage F case topology=${topology} connection=${connection_value} scenario=${scenario}; see ${case_dir}" >&2
    return 1
  fi
  echo "PASS: topology=${topology} connection=${connection_value} scenario=${scenario} d2dAggregateGBps=${d2d_gbps} backgroundAggregateGBps=${background_gbps}"
}

for topology in "${topology_array[@]}"; do
  for connection_value in "${connection_array[@]}"; do
    for scenario in "${scenario_array[@]}"; do
      for repetition in $(seq 1 "${runs}"); do
        run_case "${topology}" "${connection_value}" "${scenario}" "${repetition}"
      done
    done
  done
done

echo "Completed three-GPU Stage F work-queue matrix: ${output_root}"
echo "Summary: ${summary_file}"
