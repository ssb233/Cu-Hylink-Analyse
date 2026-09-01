#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
common_mk="$repo_root/third_party/nccl/makefiles/common.mk"
device_h="$repo_root/third_party/nccl/src/include/device.h"
common_h="$repo_root/third_party/nccl/src/device/common.h"
prims_h="$repo_root/third_party/nccl/src/device/prims_simple.h"
init_cc="$repo_root/third_party/nccl/src/init.cc"
trace_cc="$repo_root/third_party/nccl/src/plugin/primitive_trace.cc"

test -f "$common_mk"
test -f "$device_h"
test -f "$common_h"
test -f "$prims_h"
test -f "$init_cc"
test -f "$trace_cc"

rg -q 'PRIMITIVE_TRACE' "$common_mk"
rg -q 'struct ncclDevPrimitiveTraceRecord' "$device_h"
rg -q 'struct ncclDevPrimitiveTrace' "$device_h"
rg -q 'primitiveTrace' "$device_h"
rg -q 'NCCL_EXPERIMENT_PRIMITIVE_TRACE_KIND' "$common_h"
rg -q 'globaltimer\(\)' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_RECORDS_PER_CHANNEL' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_KIND_WAIT' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_KIND_FENCE' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_KIND_STORE' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_KIND_POST' "$prims_h"
rg -q 'NCCL_PRIMITIVE_TRACE_KIND_COPY' "$prims_h"
rg -q 'ncclPrimitiveTraceDump' "$init_cc"
rg -q 'NCCL_PRIMITIVE_TRACE_FILE' "$trace_cc"
rg -q 'overflow' "$trace_cc"

echo "NCCL primitive trace contract passed"
