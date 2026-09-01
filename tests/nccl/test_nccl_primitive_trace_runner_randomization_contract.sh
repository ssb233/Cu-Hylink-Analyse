#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
runner="$repo_root/scripts/run_nccl_primitive_trace.py"

test -f "$runner"
rg -q '^import random$' "$runner"
rg -q 'random\.Random\(args\.randomSeed\)' "$runner"
rg -q 'rng\.shuffle' "$runner"
rg -q 'case_specs' "$runner"

echo "NCCL primitive trace runner randomization contract passed"
