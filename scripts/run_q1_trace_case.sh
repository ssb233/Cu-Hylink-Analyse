#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"

device_list="0,1,2,3"
edge_order="source-major"
edge_permutation=""
stream_mode=""
stream_dependency="none"
streams_per_source=""
stream_assignment=""
source_offsets_us=""
scenario="d2h-all"
repeats=300
size="255M"
output_root=""
background_pid=""
background_stop=""
background_rc=0
original_args=("$@")

usage() {
  cat <<'EOF'
Usage: run_q1_trace_case.sh [options]

Run one Q1 allpairs case. This helper is intended to be wrapped by
`nsys profile` so that the D2D process and optional background worker are
captured in the same trace.

Options:
  --streamMode=per-source|per-edge
  --streamDependency=none|source-chain
  --streamsPerSource=1|2|3  optional Q2 stream-count override
  --streamAssignment=0,1,1  optional two-stream copy assignment
  --sourceOffsetsUs=0,250,500,750  optional Q4 source delays
  --edgeOrder=source-major|destination-major
  --edgePermutation=0,1,2  optional Q3 per-source permutation
  --deviceList=0,1,2,3
  --scenario=d2h-all|none
  --repeats=300
  --size=255M
  --outputRoot=<path>       fresh case-result directory
  --help

D2D warmup is fixed at 10 iterations. The default scenario is all-GPU D2H.
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
    --streamMode=*)
      stream_mode="${argument#*=}"
      shift
      ;;
    --streamMode)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      stream_mode="$2"
      shift 2
      ;;
    --streamDependency=*)
      stream_dependency="${argument#*=}"
      shift
      ;;
    --streamDependency)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      stream_dependency="$2"
      shift 2
      ;;
    --streamsPerSource=*)
      streams_per_source="${argument#*=}"
      shift
      ;;
    --streamsPerSource)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      streams_per_source="$2"
      shift 2
      ;;
    --sourceOffsetsUs=*)
      source_offsets_us="${argument#*=}"
      shift
      ;;
    --sourceOffsetsUs)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      source_offsets_us="$2"
      shift 2
      ;;
    --streamAssignment=*)
      stream_assignment="${argument#*=}"
      shift
      ;;
    --streamAssignment)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      stream_assignment="$2"
      shift 2
      ;;
    --edgeOrder=*)
      edge_order="${argument#*=}"
      shift
      ;;
    --edgeOrder)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      edge_order="$2"
      shift 2
      ;;
    --edgePermutation=*)
      edge_permutation="${argument#*=}"
      shift
      ;;
    --edgePermutation)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      edge_permutation="$2"
      shift 2
      ;;
    --deviceList=*|--device-list=*)
      device_list="${argument#*=}"
      shift
      ;;
    --deviceList|--device-list)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      device_list="$2"
      shift 2
      ;;
    --scenario=*)
      scenario="${argument#*=}"
      shift
      ;;
    --scenario)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      scenario="$2"
      shift 2
      ;;
    --repeats=*)
      repeats="${argument#*=}"
      shift
      ;;
    --repeats)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      repeats="$2"
      shift 2
      ;;
    --size=*)
      size="${argument#*=}"
      shift
      ;;
    --size)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
      size="$2"
      shift 2
      ;;
    --outputRoot=*)
      output_root="${argument#*=}"
      shift
      ;;
    --outputRoot)
      [[ $# -ge 2 ]] || {
        echo "ERROR: ${argument} requires a value" >&2
        exit 2
      }
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

if [[ -n "${streams_per_source}" ]]; then
  case "${streams_per_source}" in
    1|2|3) ;;
    *)
      echo "ERROR: --streamsPerSource must be 1, 2, or 3" >&2
      exit 2
      ;;
  esac
fi
if [[ -z "${stream_mode}" && -n "${streams_per_source}" ]]; then
  stream_mode="per-source"
  if [[ "${streams_per_source}" == "3" ]]; then
    stream_mode="per-edge"
  fi
fi
[[ -n "${stream_mode}" ]] || {
  echo "ERROR: --streamMode or --streamsPerSource is required" >&2
  exit 2
}
case "${stream_mode}" in
  per-source|per-edge) ;;
  *)
    echo "ERROR: unsupported stream mode: ${stream_mode}" >&2
    exit 2
    ;;
esac
if [[ -z "${streams_per_source}" ]]; then
  streams_per_source=1
  if [[ "${stream_mode}" == "per-edge" ]]; then
    streams_per_source=3
  fi
elif [[ "${stream_mode}" == "per-source" &&
        "${streams_per_source}" == "3" ]]; then
  echo "ERROR: --streamMode conflicts with --streamsPerSource" >&2
  exit 2
elif [[ "${stream_mode}" == "per-edge" &&
        "${streams_per_source}" != "3" ]]; then
  echo "ERROR: --streamMode conflicts with --streamsPerSource" >&2
  exit 2
fi
case "${stream_dependency}" in
  none|source-chain) ;;
  *)
    echo "ERROR: unsupported stream dependency: ${stream_dependency}" >&2
    exit 2
    ;;
esac
case "${edge_order}" in
  source-major|destination-major) ;;
  *)
    echo "ERROR: unsupported edge order: ${edge_order}" >&2
    exit 2
    ;;
esac
if [[ -n "${edge_permutation}" ]]; then
  case "${edge_permutation}" in
    0,1,2|2,1,0|1,2,0|2,0,1|0,2,1|1,0,2) ;;
    *)
      echo "ERROR: unsupported edge permutation: ${edge_permutation}" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${source_offsets_us}" ]]; then
  [[ "${stream_mode}" == "per-source" &&
     "${streams_per_source}" == "1" ]] || {
    echo "ERROR: --sourceOffsetsUs requires one stream per source" >&2
    exit 2
  }
  IFS=',' read -r -a device_array <<< "${device_list}"
  IFS=',' read -r -a source_offset_array <<< "${source_offsets_us}"
  [[ "${#source_offset_array[@]}" -eq "${#device_array[@]}" ]] || {
    echo "ERROR: --sourceOffsetsUs length must equal the device count" >&2
    exit 2
  }
  canonical_offsets=""
  for offset in "${source_offset_array[@]}"; do
    offset="${offset//[[:space:]]/}"
    [[ "${offset}" =~ ^[0-9]+$ ]] || {
      echo "ERROR: --sourceOffsetsUs values must be non-negative integers" >&2
      exit 2
    }
    if [[ -n "${canonical_offsets}" ]]; then canonical_offsets+=","; fi
    canonical_offsets+="${offset}"
  done
  source_offsets_us="${canonical_offsets}"
fi
if [[ -n "${stream_assignment}" ]]; then
  [[ "${stream_mode}" == "per-source" &&
     "${streams_per_source}" == "2" ]] || {
    echo "ERROR: --streamAssignment requires per-source streamsPerSource=2" >&2
    exit 2
  }
  case "${stream_assignment//[[:space:]]/}" in
    0,1,1|1,0,1|1,1,0) ;;
    *)
      echo "ERROR: unsupported --streamAssignment: ${stream_assignment}" >&2
      exit 2
      ;;
  esac
  stream_assignment="${stream_assignment//[[:space:]]/}"
fi
case "${scenario}" in
  none|d2h-all) ;;
  *)
    echo "ERROR: unsupported scenario: ${scenario}" >&2
    exit 2
    ;;
esac
[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --repeats must be a positive integer" >&2
  exit 2
}
if [[ "${stream_mode}" == "per-source" &&
      "${stream_dependency}" == "source-chain" ]]; then
  echo "ERROR: source-chain requires --streamMode=per-edge" >&2
  exit 2
fi

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

if [[ -z "${output_root}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_root="${repo_root}/doc/results/gpu-contention/queue-phase-diagnostic/q1-source-chain/traces/${timestamp}/${stream_mode}-${streams_per_source}-${stream_dependency}-${scenario}"
fi
mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"
if [[ -e "${output_root}/command.txt" ]]; then
  echo "ERROR: output directory already contains command.txt; choose a fresh --outputRoot" >&2
  exit 1
fi

ready_file="${output_root}/ready"
stop_file="${output_root}/stop"
d2d_json="${output_root}/d2d.json"
d2d_log="${output_root}/d2d.log"
background_json="${output_root}/background.json"
background_log="${output_root}/background.log"
rm -f "${ready_file}" "${stop_file}"

d2d_cmd=(
  "${multi_bin}"
  --pattern=allpairs
  "--deviceList=${device_list}"
  "--edgeOrder=${edge_order}"
  "--streamMode=${stream_mode}"
  "--streamDependency=${stream_dependency}"
  "--streamsPerSource=${streams_per_source}"
  "--size=${size}"
  "--repeats=${repeats}"
  "--output=${d2d_json}"
)
if [[ -n "${edge_permutation}" ]]; then
  d2d_cmd+=("--edgePermutation=${edge_permutation}")
fi
if [[ -n "${source_offsets_us}" ]]; then
  d2d_cmd+=("--sourceOffsetsUs=${source_offsets_us}")
fi
if [[ -n "${stream_assignment}" ]]; then
  d2d_cmd+=("--streamAssignment=${stream_assignment}")
fi
{
  printf 'streamMode=%s streamsPerSource=%s streamDependency=%s edgeOrder=%s edgePermutation=%s streamAssignment=%s sourceOffsetsUs=%s scenario=%s repeats=%s size=%s\n' \
    "${stream_mode}" "${streams_per_source}" "${stream_dependency}" \
    "${edge_order}" "${edge_permutation:-identity}" \
    "${stream_assignment:-identity}" "${source_offsets_us:-identity}" \
    "${scenario}" \
    "${repeats}" "${size}"
  printf 'case_command='
  printf '%q ' "$0" "${original_args[@]}"
  printf '\n'
  printf 'd2d_command='
  printf '%q ' "${d2d_cmd[@]}"
  printf '\n'
} > "${output_root}/command.txt"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${repo_root}"
  printf 'git_revision='; git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'NCCL_HOME=%s\n' "${NCCL_HOME:-unset}"
  printf 'device_list=%s\nedge_order=%s\nstream_mode=%s\nstreams_per_source=%s\nstream_dependency=%s\nedge_permutation=%s\nstream_assignment=%s\nsource_offsets_us=%s\nscenario=%s\nrepeats=%s\nsize=%s\n' \
    "${device_list}" "${edge_order}" "${stream_mode}" "${streams_per_source}" \
    "${stream_dependency}" "${edge_permutation:-identity}" \
    "${stream_assignment:-identity}" "${source_offsets_us:-identity}" \
    "${scenario}" \
    "${repeats}" "${size}"
  printf '\n### nvidia-smi ###\n'
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

if [[ "${scenario}" == "d2h-all" ]]; then
  background_cmd=(
    "${background_bin}"
    "--devList=${device_list}"
    --direction=d2h
    "--size=${size}"
    "--readyFile=${ready_file}"
    "--stopFile=${stop_file}"
    --reportSec=1
    "--output=${background_json}"
  )
  {
    printf 'background_command='
    printf '%q ' "${background_cmd[@]}"
    printf '\n'
  } >> "${output_root}/command.txt"
  "${background_cmd[@]}" > "${background_log}" 2>&1 &
  background_pid=$!
  background_stop="${stop_file}"
  for _ in $(seq 1 300); do
    [[ -f "${ready_file}" ]] && break
    if ! kill -0 "${background_pid}" 2>/dev/null; then
      wait "${background_pid}" || true
      background_pid=""
      background_stop=""
      echo "ERROR: background process exited before ready" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ ! -f "${ready_file}" ]]; then
    echo "ERROR: background process did not become ready" >&2
    exit 1
  fi
fi

d2d_rc=0
if "${d2d_cmd[@]}" > "${d2d_log}" 2>&1; then
  d2d_rc=0
else
  d2d_rc=$?
fi

if [[ "${scenario}" == "d2h-all" ]]; then
  stop_background
  background_case_rc="${background_rc}"
else
  background_case_rc=0
fi

printf 'd2d_exit=%s\nbackground_exit=%s\n' "${d2d_rc}" "${background_case_rc}" > "${output_root}/status.txt"
if [[ "${d2d_rc}" -ne 0 || "${background_case_rc}" -ne 0 ]]; then
  echo "ERROR: Q1 trace case failed; see ${output_root}" >&2
  exit 1
fi
echo "PASS: Q1 trace case stream=${stream_mode} dependency=${stream_dependency} streamAssignment=${stream_assignment:-identity} sourceOffsetsUs=${source_offsets_us:-identity} scenario=${scenario} output=${output_root}"
