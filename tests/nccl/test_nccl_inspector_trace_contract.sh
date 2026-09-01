#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
analyzer="$repo_root/scripts/analyze_nccl_inspector_trace.py"
test -x "$analyzer"
python3 -m py_compile "$analyzer"

smoke_root="$repo_root/doc/results/nccl-contention/inspector/d2h-all-smoke-20260828"
if [[ -d "$smoke_root" ]]; then
  python3 "$analyzer" "$smoke_root" >/dev/null
  test -s "$smoke_root/trace-records.csv"
  test -s "$smoke_root/trace-by-rank.csv"
  test -s "$smoke_root/trace-by-channel.csv"
fi

echo "NCCL Inspector trace contract passed"
