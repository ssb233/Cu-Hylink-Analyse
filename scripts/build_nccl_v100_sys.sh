#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
nccl_src="$repo_root/third_party/nccl"
tests_src="$repo_root/third_party/nccl-tests"
build_root="${1:-$repo_root/build/nccl-v2.31.2-sm70-sys}"
tests_build="${2:-$repo_root/build/nccl-tests-v2.31.2-sm70}"
inspector_build="$build_root/inspector"
jobs="${JOBS:-2}"

if [[ ! -d "$nccl_src/.git" || ! -d "$tests_src/.git" ]]; then
  echo "expected third_party NCCL and nccl-tests git trees" >&2
  exit 2
fi

if ! rg -q 'fence_acq_rel_sys\(\)' "$nccl_src/src/device/prims_simple.h"; then
  echo "third_party NCCL source is not the expected .sys source" >&2
  exit 2
fi
if rg -q 'fence_acq_rel_gpu\(\)' "$nccl_src/src/device/prims_simple.h"; then
  echo "third_party NCCL source unexpectedly contains the .gpu experiment" >&2
  exit 2
fi

mkdir -p "$build_root" "$tests_build" "$inspector_build"
gencode='-gencode=arch=compute_70,code=sm_70'

make -C "$nccl_src" -j"$jobs" \
  BUILDDIR="$build_root" \
  NVCC_GENCODE="$gencode"

make -C "$nccl_src/plugins/profiler/inspector" -j"$jobs" \
  OBJDIR="$inspector_build/obj" \
  TARGET="$inspector_build/libnccl-profiler-inspector.so" \
  CUDA_HOME=/usr/local/cuda

make -C "$tests_src/src" -j"$jobs" build \
  BUILDDIR="$tests_build" \
  NCCL_HOME="$build_root" \
  NVCC_GENCODE="$gencode"

echo "NCCL build: $build_root"
echo "Inspector build: $inspector_build/libnccl-profiler-inspector.so"
echo "nccl-tests build: $tests_build"
