#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
nccl_source="$repo_root/third_party/nccl"
nccl_build="${1:-$repo_root/build/nccl-v2.31.2-sm70-sys}"
tests_build="${2:-$repo_root/build/nccl-tests-v2.31.2-sm70}"

test -f "$nccl_source/src/device/prims_simple.h"
rg -q 'fence_acq_rel_sys\(\)' "$nccl_source/src/device/prims_simple.h"
rg -q '#if NCCL_EXPERIMENT_FENCE_SCOPE_SYS' "$nccl_source/src/device/prims_simple.h"
rg -q 'fence_acq_rel_gpu\(\); // EXPERIMENT: Ring\+Simple post-send fence' "$nccl_source/src/device/prims_simple.h"
rg -q 'PRIMITIVE_TRACE_FENCE_SCOPE' "$nccl_source/makefiles/common.mk"

rg -q '^#define NCCL_MAJOR 2$' "$nccl_build/include/nccl.h"
rg -q '^#define NCCL_MINOR 31$' "$nccl_build/include/nccl.h"
rg -q '^#define NCCL_PATCH 2$' "$nccl_build/include/nccl.h"
test -f "$nccl_build/lib/libnccl.so.2.31.2"
test -f "$nccl_build/inspector/libnccl-profiler-inspector.so"

for binary in all_gather_perf all_reduce_perf reduce_scatter_perf; do
  test -x "$tests_build/$binary"
  linked=$(LD_LIBRARY_PATH="$nccl_build/lib:/usr/local/cuda/lib64" \
    ldd "$tests_build/$binary")
  echo "$linked" | rg -q "$nccl_build/lib"
done

echo "NCCL sys build contract passed"
echo "nccl_build=$nccl_build"
echo "tests_build=$tests_build"
