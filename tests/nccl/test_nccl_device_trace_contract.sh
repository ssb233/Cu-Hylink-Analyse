#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE[0]")/../.." && pwd -P)
device_h="$repo_root/third_party/nccl/src/include/device.h"
prims_h="$repo_root/third_party/nccl/src/device/prims_simple.h"
init_cc="$repo_root/third_party/nccl/src/init.cc"
trace_cc="$repo_root/third_party/nccl/src/plugin/primitive_trace.cc"
runner="$repo_root/scripts/run_nccl_primitive_trace.py"

test -f "$device_h"
test -f "$prims_h"
test -f "$init_cc"
test -f "$trace_cc"
test -f "$runner"

# Device-only storage and deterministic, non-wrapping event slots.
rg -q 'NCCL_PRIMITIVE_TRACE_MAX_SAMPLED_WORKS' "$device_h"
rg -q '#define NCCL_PRIMITIVE_TRACE_MAX_SAMPLED_WORKS 128' "$device_h"
rg -q 'NCCL_PRIMITIVE_TRACE_EVENTS_PER_ROLE' "$device_h"
rg -q 'eventCursors' "$device_h"
rg -q 'eventIndex' "$device_h"
rg -q 'droppedRecords' "$device_h"
rg -q 'NCCL_PRIMITIVE_TRACE_MAX_SAMPLED_WORKS' "$prims_h"
rg -q 'eventIndex' "$prims_h"
rg -q 'atomicAdd\(&channel->eventCursors' "$prims_h"
rg -q 'uint64_t work = ncclShmem\.workCounter' "$prims_h"
! rg -q 'ordinal[[:space:]]*%[[:space:]]*NCCL_PRIMITIVE_TRACE_WORK_SLOTS' "$prims_h"
! rg -q '__threadfence_system\(\)' "$prims_h"

# A batch can contain multiple work items; tracing must key each item separately.
rg -q 'ncclShmem\.workCounter = ncclShmem\.channel\.workCounter \+ w' "$repo_root/third_party/nccl/src/device/common.h"

# The trace is a device allocation and is freed as device memory.
rg -q 'ncclCudaCallocAsync\(&comm->profiler\.primitiveTrace' "$init_cc"
rg -q 'ncclCommPushCudaFree\(comm, comm->profiler\.primitiveTrace\)' "$init_cc"
! rg -q 'ncclCudaHostCalloc\(&comm->profiler\.primitiveTrace' "$init_cc"
! rg -q 'ncclCommPushCudaHostFree\(comm, comm->profiler\.primitiveTrace\)' "$init_cc"

# Host export happens after synchronization through one bulk CUDA copy.
rg -q 'ncclCudaMemcpy\(hostTrace, trace, 1\)' "$trace_cc"
rg -Fq '"storage\":\"device' "$trace_cc"
rg -q 'droppedRecords' "$trace_cc"

# Formal sampling periods are capacity-checked before launching a case.
rg -q '512,256,128' "$runner"
rg -q 'maxSampledWorks' "$runner"
rg -q 'capacity' "$runner"
rg -q 'NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER' "$runner"
rg -q 'workCounterUpperBound' "$runner"

echo "NCCL device-only primitive trace contract passed"
