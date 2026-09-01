#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
runner="$repo_root/scripts/run_nccl_primitive_trace.py"

test -f "$runner"
rg -q 'NCCL_PRIMITIVE_TRACE_FILE' "$runner"
rg -q 'clean-before' "$runner"
rg -q 'concurrent' "$runner"
rg -q 'clean-after' "$runner"
rg -q 'run_treatment' "$runner"
rg -q 'primitive-trace' "$runner"

echo "NCCL primitive trace runner contract passed"
