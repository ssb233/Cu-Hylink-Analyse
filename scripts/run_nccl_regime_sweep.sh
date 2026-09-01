#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$BASH_SOURCE")" && pwd -P)
repo_root=$(cd "$script_dir/.." && pwd -P)

devices=0,1,2
collectives=allgather,allreduce,reducescatter
sizes=64M
warmups=5,20
iterations=20,100,500
repetitions=3
background_size=255M
background_direction=d2h
background_devices=0,1,2
settle_seconds=1
stop_timeout_seconds=10
debug_level=INFO
debug_subsystems=INIT,GRAPH,TUNING,COLL,PROFILE
inspector_mode=off
inspector_interval_us=1000
nccl_build="$repo_root/build/nccl-v2.31.2-sm70-sys"
tests_build="$repo_root/build/nccl-tests-v2.31.2-sm70"
background_bin="$repo_root/build/cuda_copy/host_copy_background"
inspector_plugin="$nccl_build/inspector/libnccl-profiler-inspector.so"
output_root=

background_pid=
background_stop=
background_ready=
background_json=
background_log=
background_rc=0

usage() {
  cat <<'EOF'
Usage: run_nccl_regime_sweep.sh [options]

Run the first NCCL B-stage regime calibration. Each point is:
clean-before -> D2H-all -> clean-after.

Options:
  --devices=0,1,2                 physical GPUs exposed through CUDA_VISIBLE_DEVICES
  --collectives=allgather,...     allgather, allreduce, reducescatter
  --sizes=64M                     NCCL test sizes
  --warmups=5,20                  warmup counts
  --iterations=20,100,500         measured iteration counts
  --repetitions=3                 paired repetitions per point
  --backgroundSize=255M           one D2H/H2D memcpy payload
  --backgroundDirection=d2h       d2h or h2d
  --backgroundDevices=0,1,2        local devices receiving background traffic
  --settleSeconds=1               wait after background ready
  --stopTimeoutSeconds=10         graceful-stop timeout
  --inspector=off                  off or on; on writes Inspector JSONL traces
  --inspectorIntervalUs=1000      positive periodic Inspector dump interval
  --inspectorPlugin=<path>         Inspector profiler plugin
  --debug=INFO                    NCCL_DEBUG value
  --debugSubsystems=...           NCCL_DEBUG_SUBSYS value
  --ncclBuild=<path>              NCCL build root
  --testsBuild=<path>             nccl-tests build root
  --backgroundBin=<path>          host_copy_background binary
  --outputRoot=<path>             fresh result root
  --help

NCCL_ALGO=RING and NCCL_PROTO=SIMPLE are fixed by this runner.
Inspector is off by default; enable it only for representative diagnostic runs.
The current hardware adaptation requires exactly three GPUs.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

stop_background() {
  background_rc=0
  if [[ -z "$background_pid" ]]; then
    return 0
  fi

  if [[ -n "$background_stop" ]]; then
    touch "$background_stop"
  fi
  if kill -0 "$background_pid" 2>/dev/null; then
    kill -INT "$background_pid" 2>/dev/null || true
  fi

  local wait_steps=$((stop_timeout_seconds * 10))
  local step
  for ((step = 0; step < wait_steps; ++step)); do
    if ! kill -0 "$background_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  if kill -0 "$background_pid" 2>/dev/null; then
    echo "WARNING: background PID $background_pid did not stop gracefully; sending TERM" >&2
    kill -TERM "$background_pid" 2>/dev/null || true
  fi

  if wait "$background_pid"; then
    background_rc=0
  else
    background_rc=$?
  fi
  background_pid=
}

cleanup() {
  stop_background || true
}

on_signal() {
  cleanup
  exit 130
}

trap cleanup EXIT
trap on_signal INT TERM

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
    --warmups=*) warmups=${argument#*=}; shift ;;
    --warmups)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      warmups=$2
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
    --backgroundSize=*) background_size=${argument#*=}; shift ;;
    --backgroundSize)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      background_size=$2
      shift 2
      ;;
    --backgroundDirection=*) background_direction=${argument#*=}; shift ;;
    --backgroundDirection)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      background_direction=$2
      shift 2
      ;;
    --backgroundDevices=*) background_devices=${argument#*=}; shift ;;
    --backgroundDevices)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      background_devices=$2
      shift 2
      ;;
    --settleSeconds=*) settle_seconds=${argument#*=}; shift ;;
    --settleSeconds)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      settle_seconds=$2
      shift 2
      ;;
    --stopTimeoutSeconds=*) stop_timeout_seconds=${argument#*=}; shift ;;
    --stopTimeoutSeconds)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      stop_timeout_seconds=$2
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
    --inspector=*) inspector_mode=${argument#*=}; shift ;;
    --inspector)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      inspector_mode=$2
      shift 2
      ;;
    --inspectorIntervalUs=*) inspector_interval_us=${argument#*=}; shift ;;
    --inspectorIntervalUs)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      inspector_interval_us=$2
      shift 2
      ;;
    --inspectorPlugin=*) inspector_plugin=${argument#*=}; shift ;;
    --inspectorPlugin)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      inspector_plugin=$2
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
    --backgroundBin=*) background_bin=${argument#*=}; shift ;;
    --backgroundBin)
      [[ $# -ge 2 ]] || fail "$argument requires a value"
      background_bin=$2
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

[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || fail "--repetitions must be a positive integer"
[[ "$settle_seconds" =~ ^[0-9]+$ ]] || fail "--settleSeconds must be a non-negative integer"
[[ "$stop_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || fail "--stopTimeoutSeconds must be a positive integer"
case "${background_direction,,}" in
  d2h|h2d) background_direction=${background_direction,,} ;;
  *) fail "--backgroundDirection must be d2h or h2d" ;;
esac
background_scenario="${background_direction}-all"
case "${inspector_mode,,}" in
  off|on) inspector_mode=${inspector_mode,,} ;;
  *) fail "--inspector must be off or on" ;;
esac
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
local_devices=0,1,2

IFS=, read -r background_device0 background_device1 background_device2 background_extra <<< "$background_devices"
[[ -n "$background_device0" && -z "$background_extra" ]] || \
  fail "--backgroundDevices accepts one to three local devices"
for background_device in "$background_device0" "$background_device1" "$background_device2"; do
  [[ -z "$background_device" || "$background_device" =~ ^[0-2]$ ]] || \
    fail "background device must be a local device 0, 1 or 2: $background_device"
done
if [[ -n "$background_device1" && "$background_device0" == "$background_device1" ]]; then
  fail "duplicate local device in --backgroundDevices"
fi
if [[ -n "$background_device2" && \
  ( "$background_device0" == "$background_device2" || \
    "$background_device1" == "$background_device2" ) ]]; then
  fail "duplicate local device in --backgroundDevices"
fi
background_devices="$background_device0"
[[ -z "$background_device1" ]] || background_devices="$background_devices,$background_device1"
[[ -z "$background_device2" ]] || background_devices="$background_devices,$background_device2"

[[ -n "$collectives" && -n "$sizes" && -n "$warmups" && -n "$iterations" ]] || \
  fail "collectives, sizes, warmups and iterations cannot be empty"
old_ifs=$IFS
IFS=,
for collective in $collectives; do
  case "$collective" in
    allgather|allreduce|reducescatter) ;;
    *) fail "unsupported collective: $collective" ;;
  esac
done
IFS=$old_ifs

[[ -x "$background_bin" ]] || fail "background binary is not executable: $background_bin"
[[ -d "$nccl_build/lib" ]] || fail "NCCL lib directory is missing: $nccl_build/lib"
[[ -x "$tests_build/all_gather_perf" ]] || fail "all_gather_perf is missing: $tests_build"
[[ -x "$tests_build/all_reduce_perf" ]] || fail "all_reduce_perf is missing: $tests_build"
[[ -x "$tests_build/reduce_scatter_perf" ]] || fail "reduce_scatter_perf is missing: $tests_build"
if [[ "$inspector_mode" == on ]]; then
  [[ -f "$inspector_plugin" ]] || fail "Inspector plugin is missing: $inspector_plugin"
fi

if [[ -z "$output_root" ]]; then
  run_id=$(date -u +%Y%m%dT%H%M%SZ)
  output_root="$repo_root/doc/results/nccl-contention/regime-sweep/$run_id"
elif [[ "$output_root" != /* ]]; then
  output_root="$repo_root/$output_root"
fi
[[ ! -e "$output_root" ]] || fail "output root already exists: $output_root"
mkdir -p "$output_root"

printf '%s\n' \
  "runner=run_nccl_regime_sweep.sh" \
  "devices=$devices" \
  "visible_local_devices=$local_devices" \
  "gpu_count=3 (current hardware adaptation)" \
  "collectives=$collectives" \
  "sizes=$sizes" \
  "warmups=$warmups" \
  "iterations=$iterations" \
  "repetitions=$repetitions" \
  "background_size=$background_size" \
  "background_direction=$background_direction" \
  "background_devices=$background_devices" \
  "background_scenario=$background_scenario" \
  "settle_seconds=$settle_seconds" \
  "stop_timeout_seconds=$stop_timeout_seconds" \
  "inspector_mode=$inspector_mode" \
  "inspector_interval_us=$inspector_interval_us" \
  "inspector_plugin=$inspector_plugin" \
  "NCCL_ALGO=RING" \
  "NCCL_PROTO=SIMPLE" \
  "NCCL_DEBUG=$debug_level" \
  "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
  "nccl_build=$nccl_build" \
  "tests_build=$tests_build" \
  "background_bin=$background_bin" \
  "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$output_root/run-metadata.txt"
env | sort > "$output_root/runner-environment.txt"
nvidia-smi -L > "$output_root/gpu-list.txt"
nvidia-smi topo -m > "$output_root/topology.txt"
printf 'collective,size,warmup,iterations,repetition,scenario,case_dir,test_rc,background_rc\n' \
  > "$output_root/case-index.csv"

collective_binary() {
  case "$1" in
    allgather) printf '%s\n' "$tests_build/all_gather_perf" ;;
    allreduce) printf '%s\n' "$tests_build/all_reduce_perf" ;;
    reducescatter) printf '%s\n' "$tests_build/reduce_scatter_perf" ;;
  esac
}

start_background() {
  local case_dir=$1
  background_stop="$case_dir/background.stop"
  background_ready="$case_dir/background.ready"
  background_json="$case_dir/background.json"
  background_log="$case_dir/background.log"
  background_rc=0

  CUDA_VISIBLE_DEVICES="$devices" "$background_bin" \
    --devList="$background_devices" \
    --direction="$background_direction" \
    --size="$background_size" \
    --readyFile="$background_ready" \
    --stopFile="$background_stop" \
    --reportSec=2 \
    --output="$background_json" > "$background_log" 2>&1 &
  background_pid=$!

  local wait_steps=300
  local step
  for ((step = 0; step < wait_steps; ++step)); do
    if [[ -f "$background_ready" ]]; then
      sleep "$settle_seconds"
      return 0
    fi
    if ! kill -0 "$background_pid" 2>/dev/null; then
      if wait "$background_pid"; then
        background_rc=0
      else
        background_rc=$?
      fi
      background_pid=
      return 1
    fi
    sleep 0.1
  done
  echo "ERROR: background did not become ready: $case_dir" >&2
  return 1
}

run_nccl_case() {
  local collective=$1
  local size=$2
  local warmup=$3
  local measured_iterations=$4
  local case_dir=$5
  local test_bin
  test_bin=$(collective_binary "$collective")
  local debug_file="$case_dir/nccl-debug.log"
  local library_path="$nccl_build/lib:/usr/local/cuda/lib64"
  local -a unset_inspector=(
    -u NCCL_PROFILER_PLUGIN
    -u NCCL_INSPECTOR_ENABLE
    -u NCCL_INSPECTOR_ENABLE_P2P
    -u NCCL_INSPECTOR_DUMP_THREAD_ENABLE
    -u NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS
    -u NCCL_INSPECTOR_DUMP_VERBOSE
    -u NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING
    -u NCCL_INSPECTOR_DUMP_DIR
  )
  local -a inspector_env=()
  if [[ "$inspector_mode" == on ]]; then
    mkdir -p "$case_dir/inspector"
    inspector_env=(
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
    printf ' %q' "${unset_inspector[@]}" \
      "LD_LIBRARY_PATH=$library_path" \
      "CUDA_VISIBLE_DEVICES=$devices" \
      "NCCL_ALGO=RING" \
      "NCCL_PROTO=SIMPLE" \
      "NCCL_DEBUG=$debug_level" \
      "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
      "NCCL_DEBUG_FILE=$debug_file" \
      "${inspector_env[@]}" \
      "$test_bin"
    printf ' -b %q -e %q -f 2 -g 3 -n %q -w %q\n' \
      "$size" "$size" "$measured_iterations" "$warmup"
  } > "$case_dir/command.txt"
  printf '%s\n' \
    "collective=$collective" \
    "size=$size" \
    "warmup=$warmup" \
    "iterations=$measured_iterations" \
    "physical_devices=$devices" \
    "visible_local_devices=$local_devices" \
    "background_devices=$background_devices" \
    "scenario=$(basename "$case_dir")" \
    "NCCL_ALGO=RING" \
    "NCCL_PROTO=SIMPLE" \
    "NCCL_DEBUG=$debug_level" \
    "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
    "NCCL_DEBUG_FILE=$debug_file" \
    "LD_LIBRARY_PATH=$library_path" \
    "inspector_mode=$inspector_mode" \
    "inspector_interval_us=$inspector_interval_us" \
    "inspector_plugin=$inspector_plugin" \
    > "$case_dir/environment.txt"
  if [[ "$inspector_mode" == on ]]; then
    printf '%s\n' \
      "NCCL_INSPECTOR_ENABLE=1" \
      "NCCL_INSPECTOR_ENABLE_P2P=0" \
      "NCCL_INSPECTOR_DUMP_THREAD_ENABLE=1" \
      "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=$inspector_interval_us" \
      "NCCL_INSPECTOR_DUMP_VERBOSE=1" \
      "NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=1" \
      "NCCL_INSPECTOR_DUMP_DIR=$case_dir/inspector" \
      >> "$case_dir/environment.txt"
  fi

  local test_rc=0
  if env "${unset_inspector[@]}" \
    "LD_LIBRARY_PATH=$library_path" \
    "CUDA_VISIBLE_DEVICES=$devices" \
    "NCCL_ALGO=RING" \
    "NCCL_PROTO=SIMPLE" \
    "NCCL_DEBUG=$debug_level" \
    "NCCL_DEBUG_SUBSYS=$debug_subsystems" \
    "NCCL_DEBUG_FILE=$debug_file" \
    "${inspector_env[@]}" \
    "$test_bin" -b "$size" -e "$size" -f 2 -g 3 \
    -n "$measured_iterations" -w "$warmup" > "$case_dir/nccl-tests.log" 2>&1; then
    test_rc=0
  else
    test_rc=$?
  fi
  printf '%s\n' "$test_rc" > "$case_dir/test.rc"
  return "$test_rc"
}

record_case_index() {
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" >> "$output_root/case-index.csv"
}

run_one_case() {
  local collective=$1
  local size=$2
  local warmup=$3
  local measured_iterations=$4
  local repetition=$5
  local scenario=$6
  local case_dir="$output_root/$collective/size_$size/warmup_$warmup/iters_$measured_iterations/rep_$repetition/$scenario"
  mkdir -p "$case_dir"

  local test_rc=0
  local bg_rc=0
  if [[ "$scenario" == "$background_scenario" ]]; then
    if ! start_background "$case_dir"; then
      bg_rc=${background_rc:-1}
      stop_background || true
      record_case_index "$collective" "$size" "$warmup" "$measured_iterations" "$repetition" "$scenario" "$case_dir" 125 "$bg_rc"
      return 125
    fi
  fi

  if run_nccl_case "$collective" "$size" "$warmup" "$measured_iterations" "$case_dir"; then
    test_rc=0
  else
    test_rc=$?
  fi

  if [[ "$scenario" == "$background_scenario" ]]; then
    stop_background || true
    bg_rc=$background_rc
    [[ -s "$background_json" ]] || bg_rc=126
  fi

  printf '{"collective":"%s","size":"%s","warmup":%s,"iterations":%s,"repetition":%s,"scenario":"%s","testRc":%s,"backgroundRc":%s}\n' \
    "$collective" "$size" "$warmup" "$measured_iterations" "$repetition" \
    "$scenario" "$test_rc" "$bg_rc" > "$case_dir/status.json"
  record_case_index "$collective" "$size" "$warmup" "$measured_iterations" "$repetition" "$scenario" "$case_dir" "$test_rc" "$bg_rc"

  if [[ "$test_rc" -ne 0 || "$bg_rc" -ne 0 ]]; then
    echo "ERROR: case failed: $case_dir (test=$test_rc, background=$bg_rc)" >&2
    return 1
  fi
}

old_ifs=$IFS
IFS=,
for collective in $collectives; do
  for size in $sizes; do
    for warmup in $warmups; do
      for measured_iterations in $iterations; do
        for ((repetition = 1; repetition <= repetitions; ++repetition)); do
  for scenario in clean-before "$background_scenario" clean-after; do
            echo "[nccl-regime] collective=$collective size=$size warmup=$warmup iterations=$measured_iterations repetition=$repetition scenario=$scenario"
            run_one_case "$collective" "$size" "$warmup" "$measured_iterations" "$repetition" "$scenario"
          done
        done
      done
    done
  done
done
IFS=$old_ifs

printf '%s\n' "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/run-metadata.txt"
echo "NCCL regime sweep completed: $output_root"
