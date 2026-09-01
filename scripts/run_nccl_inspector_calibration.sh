#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$BASH_SOURCE")" && pwd -P)
repo_root=$(cd "$script_dir/.." && pwd -P)

devices=0,1,2
collectives=allgather,allreduce,reducescatter
sizes=64M,256M
warmup=20
iterations=100
repetitions=5
inspector_interval_us=1000
debug_level=INFO
debug_subsystems=INIT,GRAPH,TUNING,COLL,PROFILE
nccl_build="$repo_root/build/nccl-v2.31.2-sm70-sys"
tests_build="$repo_root/build/nccl-tests-v2.31.2-sm70"
inspector_plugin="$nccl_build/inspector/libnccl-profiler-inspector.so"
output_root=

usage() {
  cat <<'EOF'
Usage: run_nccl_inspector_calibration.sh [options]

Run clean NCCL cases with Inspector OFF and ON. The two modes use the same
third_party NCCL v2.31.2 sm70 .sys build and are kept in separate case dirs.

Options:
  --devices=0,1,2                 physical GPUs exposed through CUDA_VISIBLE_DEVICES
  --collectives=allgather,...     allgather, allreduce, reducescatter
  --sizes=64M,256M                NCCL test sizes
  --warmup=20                     warmup count
  --iterations=100                measured iteration count
  --repetitions=5                 independent repetitions per mode
  --inspectorIntervalUs=1000      positive periodic Inspector dump interval
  --debug=INFO                    NCCL_DEBUG value
  --debugSubsystems=...           NCCL_DEBUG_SUBSYS value
  --ncclBuild=<path>              NCCL build root
  --testsBuild=<path>             nccl-tests build root
  --inspectorPlugin=<path>        Inspector profiler plugin
  --outputRoot=<path>             fresh result root
  --help

This calibration has no D2H background. It measures Inspector overhead only;
the performance-truth runner remains scripts/run_nccl_regime_sweep.sh.
The current hardware adaptation requires exactly three GPUs.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  argument=$1
  case "$argument" in
    --help|-h)
      usage
      exit 0
      ;;
    --devices=*) devices=${argument#*=}; shift ;;
    --devices)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      devices=$2
      shift 2
      ;;
    --collectives=*) collectives=${argument#*=}; shift ;;
    --collectives)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      collectives=$2
      shift 2
      ;;
    --sizes=*) sizes=${argument#*=}; shift ;;
    --sizes)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      sizes=$2
      shift 2
      ;;
    --warmup=*) warmup=${argument#*=}; shift ;;
    --warmup)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      warmup=$2
      shift 2
      ;;
    --iterations=*) iterations=${argument#*=}; shift ;;
    --iterations)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      iterations=$2
      shift 2
      ;;
    --repetitions=*) repetitions=${argument#*=}; shift ;;
    --repetitions)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      repetitions=$2
      shift 2
      ;;
    --inspectorIntervalUs=*) inspector_interval_us=${argument#*=}; shift ;;
    --inspectorIntervalUs)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      inspector_interval_us=$2
      shift 2
      ;;
    --debug=*) debug_level=${argument#*=}; shift ;;
    --debug)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      debug_level=$2
      shift 2
      ;;
    --debugSubsystems=*) debug_subsystems=${argument#*=}; shift ;;
    --debugSubsystems)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      debug_subsystems=$2
      shift 2
      ;;
    --ncclBuild=*) nccl_build=${argument#*=}; shift ;;
    --ncclBuild)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      nccl_build=$2
      shift 2
      ;;
    --testsBuild=*) tests_build=${argument#*=}; shift ;;
    --testsBuild)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      tests_build=$2
      shift 2
      ;;
    --inspectorPlugin=*) inspector_plugin=${argument#*=}; shift ;;
    --inspectorPlugin)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      inspector_plugin=$2
      shift 2
      ;;
    --outputRoot=*) output_root=${argument#*=}; shift ;;
    --outputRoot)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      output_root=$2
      shift 2
      ;;
    *)
      usage >&2
      fail "unknown option: $argument"
      ;;
  esac
done

[[ "$warmup" =~ ^[0-9]+$ ]] || fail "--warmup must be a non-negative integer"
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || fail "--iterations must be a positive integer"
[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || fail "--repetitions must be a positive integer"
[[ "$inspector_interval_us" =~ ^[1-9][0-9]*$ ]] || \
  fail "--inspectorIntervalUs must be a positive integer"

IFS=, read -r device0 device1 device2 extra_device <<< "$devices"
[[ -n "$device0" && -n "$device1" && -n "$device2" && -z "$extra_device" ]] || \
  fail "this runner currently requires exactly three GPUs"
for device in "$device0" "$device1" "$device2"; do
  [[ "$device" =~ ^[0-9]+$ ]] || fail "invalid device index: $device"
done
[[ "$device0" != "$device1" && "$device0" != "$device2" && "$device1" != "$device2" ]] || \
  fail "duplicate device index in --devices"
devices="$device0,$device1,$device2"

[[ -n "$collectives" && -n "$sizes" ]] || fail "collectives and sizes cannot be empty"
old_ifs=$IFS
IFS=,
for collective in $collectives; do
  case "$collective" in
    allgather|allreduce|reducescatter) ;;
    *) fail "unsupported collective: $collective" ;;
  esac
done
IFS=$old_ifs

[[ -x "$tests_build/all_gather_perf" ]] || fail "all_gather_perf is missing: $tests_build"
[[ -x "$tests_build/all_reduce_perf" ]] || fail "all_reduce_perf is missing: $tests_build"
[[ -x "$tests_build/reduce_scatter_perf" ]] || fail "reduce_scatter_perf is missing: $tests_build"
[[ -d "$nccl_build/lib" ]] || fail "NCCL lib directory is missing: $nccl_build/lib"
[[ -f "$inspector_plugin" ]] || fail "Inspector plugin is missing: $inspector_plugin"

if [[ -z "$output_root" ]]; then
  run_id=$(date -u +%Y%m%dT%H%M%SZ)
  output_root="$repo_root/doc/results/nccl-contention/inspector/calibration-$run_id"
elif [[ "$output_root" != /* ]]; then
  output_root="$repo_root/$output_root"
fi
[[ ! -e "$output_root" ]] || fail "output root already exists: $output_root"
mkdir -p "$output_root"

local_devices=0,1,2
library_path="$nccl_build/lib:/usr/local/cuda/lib64"
printf '%s\n' \
  "runner=run_nccl_inspector_calibration.sh" \
  "devices=$devices" \
  "visible_local_devices=$local_devices" \
  "gpu_count=3 (current hardware adaptation)" \
  "collectives=$collectives" \
  "sizes=$sizes" \
  "warmup=$warmup" \
  "iterations=$iterations" \
  "repetitions=$repetitions" \
  "inspector_interval_us=$inspector_interval_us" \
  "NCCL_ALGO=RING" \
  "NCCL_PROTO=SIMPLE" \
  "NCCL_DEBUG=$debug_level" \
  "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
  "nccl_build=$nccl_build" \
  "tests_build=$tests_build" \
  "inspector_plugin=$inspector_plugin" \
  "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$output_root/run-metadata.txt"
env | sort > "$output_root/runner-environment.txt"
nvidia-smi -L > "$output_root/gpu-list.txt"
nvidia-smi topo -m > "$output_root/topology.txt"
printf 'collective,size,warmup,iterations,repetition,mode,case_dir,test_rc\n' \
  > "$output_root/case-index.csv"

collective_binary() {
  case "$1" in
    allgather) printf '%s\n' "$tests_build/all_gather_perf" ;;
    allreduce) printf '%s\n' "$tests_build/all_reduce_perf" ;;
    reducescatter) printf '%s\n' "$tests_build/reduce_scatter_perf" ;;
  esac
}

run_case() {
  local collective=$1
  local size=$2
  local repetition=$3
  local mode=$4
  local test_bin
  local case_dir="$output_root/$collective/size_$size/warmup_$warmup/iters_$iterations/rep_$repetition/$mode"
  local debug_file="$case_dir/nccl-debug.log"
  test_bin=$(collective_binary "$collective")
  mkdir -p "$case_dir"
  if [[ "$mode" == inspector-on ]]; then
    mkdir -p "$case_dir/inspector"
  fi

  local -a unset_args=(
    -u NCCL_PROFILER_PLUGIN
    -u NCCL_INSPECTOR_ENABLE
    -u NCCL_INSPECTOR_ENABLE_P2P
    -u NCCL_INSPECTOR_DUMP_THREAD_ENABLE
    -u NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
    -u NCCL_INSPECTOR_DUMP_VERBOSE
    -u NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING
    -u NCCL_INSPECTOR_DUMP_DIR
  )
  local -a env_args=(
    "LD_LIBRARY_PATH=$library_path"
    "CUDA_VISIBLE_DEVICES=$devices"
    "NCCL_ALGO=RING"
    "NCCL_PROTO=SIMPLE"
    "NCCL_DEBUG=$debug_level"
    "NCCL_DEBUG_SUBSYS=$debug_subsystems"
    "NCCL_DEBUG_FILE=$debug_file"
  )
  if [[ "$mode" == inspector-on ]]; then
    env_args+=(
      "NCCL_PROFILER_PLUGIN=$inspector_plugin"
      "NCCL_INSPECTOR_ENABLE=1"
      "NCCL_INSPECTOR_ENABLE_P2P=0"
      "NCCL_INSPECTOR_DUMP_THREAD_ENABLE=1"
      "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=$inspector_interval_us"
      "NCCL_INSPECTOR_DUMP_VERBOSE=1"
      "NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=1"
      "NCCL_INSPECTOR_DUMP_DIR=$case_dir/inspector"
    )
  fi

  {
    printf 'env'
    printf ' %q' "${unset_args[@]}" "${env_args[@]}" "$test_bin"
    printf ' -b %q -e %q -f 2 -g 3 -n %q -w %q\n' \
      "$size" "$size" "$iterations" "$warmup"
  } > "$case_dir/command.txt"
  {
    printf '%s\n' \
      "collective=$collective" \
      "size=$size" \
      "warmup=$warmup" \
      "iterations=$iterations" \
      "repetition=$repetition" \
      "mode=$mode" \
      "physical_devices=$devices" \
      "visible_local_devices=$local_devices" \
      "NCCL_ALGO=RING" \
      "NCCL_PROTO=SIMPLE" \
      "NCCL_DEBUG=$debug_level" \
      "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
      "NCCL_DEBUG_FILE=$debug_file" \
      "LD_LIBRARY_PATH=$library_path" \
      "inspector_plugin=$inspector_plugin"
    if [[ "$mode" == inspector-on ]]; then
      printf '%s\n' \
        "NCCL_INSPECTOR_ENABLE=1" \
        "NCCL_INSPECTOR_ENABLE_P2P=0" \
        "NCCL_INSPECTOR_DUMP_THREAD_ENABLE=1" \
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=$inspector_interval_us" \
        "NCCL_INSPECTOR_DUMP_VERBOSE=1" \
        "NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=1" \
        "NCCL_INSPECTOR_DUMP_DIR=$case_dir/inspector"
    else
      printf '%s\n' 'inspector=off'
    fi
  } > "$case_dir/environment.txt"
  env LD_LIBRARY_PATH="$library_path" ldd "$test_bin" > "$case_dir/ldd.txt"

  local test_rc=0
  if env "${unset_args[@]}" "${env_args[@]}" "$test_bin" \
    -b "$size" -e "$size" -f 2 -g 3 -n "$iterations" -w "$warmup" \
    > "$case_dir/nccl-tests.log" 2>&1; then
    test_rc=0
  else
    test_rc=$?
  fi
  printf '%s\n' "$test_rc" > "$case_dir/test.rc"
  printf '{"collective":"%s","size":"%s","warmup":%s,"iterations":%s,"repetition":%s,"mode":"%s","testRc":%s}\n' \
    "$collective" "$size" "$warmup" "$iterations" "$repetition" "$mode" "$test_rc" \
    > "$case_dir/status.json"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$collective" "$size" "$warmup" "$iterations" "$repetition" "$mode" "$case_dir" "$test_rc" \
    >> "$output_root/case-index.csv"
  if [[ "$test_rc" -ne 0 ]]; then
    echo "ERROR: Inspector calibration case failed: $case_dir (test=$test_rc)" >&2
    return 1
  fi
}

old_ifs=$IFS
IFS=,
for collective in $collectives; do
  for size in $sizes; do
    for ((repetition = 1; repetition <= repetitions; ++repetition)); do
      for mode in inspector-off inspector-on; do
        echo "[nccl-inspector] collective=$collective size=$size repetition=$repetition mode=$mode"
        run_case "$collective" "$size" "$repetition" "$mode"
      done
    done
  done
done
IFS=$old_ifs

printf '%s\n' "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/run-metadata.txt"
echo "NCCL Inspector calibration completed: $output_root"
