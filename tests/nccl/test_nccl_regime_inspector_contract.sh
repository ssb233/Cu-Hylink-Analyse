#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
runner="$repo_root/scripts/run_nccl_regime_sweep.sh"

test -x "$runner"
help_output=$(bash "$runner" --help)
echo "$help_output" | rg -q -- '--inspector=off|on'
echo "$help_output" | rg -q -- '--inspectorIntervalUs='
echo "$help_output" | rg -q -- '--backgroundDevices='

echo "NCCL regime Inspector contract passed"
