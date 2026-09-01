#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
runner="$repo_root/scripts/run_nccl_inspector_calibration.sh"
analyzer="$repo_root/scripts/analyze_nccl_inspector_calibration.py"

test -x "$runner"
test -x "$analyzer"

help_output=$(bash "$runner" --help)
echo "$help_output" | rg -q -- '--inspectorIntervalUs='
echo "$help_output" | rg -q -- '--collectives='
echo "$help_output" | rg -q -- 'Inspector OFF and ON'

if bash "$runner" --devices=0,1 >/dev/null 2>&1; then
  echo "runner accepted a non-three-GPU device list" >&2
  exit 1
fi

python3 -m py_compile "$analyzer"
bash -n "$runner"

echo "NCCL Inspector calibration contract passed"
