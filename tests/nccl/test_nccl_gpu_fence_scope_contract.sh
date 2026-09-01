#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
source_file="$repo_root/third_party/nccl/src/device/prims_simple.h"

test -f "$source_file"
gpu_lines=$(rg -n '^ *fence_acq_rel_gpu\(\); // EXPERIMENT: Ring\+Simple post-send fence$' "$source_file" || true)
test "$(printf '%s\n' "$gpu_lines" | sed '/^$/d' | wc -l)" -eq 6
rg -q 'fence_acq_rel_sys\(\)' "$source_file"
rg -q 'st_relaxed_sys_global' "$source_file"
rg -q 'PRIMITIVE_TRACE_FENCE_SCOPE' "$repo_root/third_party/nccl/makefiles/common.mk"
rg -q 'NCCL_EXPERIMENT_FENCE_SCOPE_SYS' "$source_file"

echo "NCCL gpu fence scope contract passed"
