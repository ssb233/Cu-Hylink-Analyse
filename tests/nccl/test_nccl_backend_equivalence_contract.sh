#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
runner="$repo_root/scripts/run_nccl_backend_equivalence.py"

test -f "$runner"
python3 "$runner" --self-test
help_output=$(python3 "$runner" --help)
echo "$help_output" | rg -q -- '--backends( |$)'
echo "$help_output" | rg -q -- '--collectives( |$)'
echo "$help_output" | rg -q -- '--sizes( |$)'
echo "$help_output" | rg -q -- 'official-sys'
echo "$help_output" | rg -q -- 'experiment-'
if echo "$help_output" | rg -q -- --ncclBuild; then
  echo "runner must expose named backends instead of arbitrary --ncclBuild" >&2
  exit 1
fi

echo "NCCL backend equivalence contract passed"
