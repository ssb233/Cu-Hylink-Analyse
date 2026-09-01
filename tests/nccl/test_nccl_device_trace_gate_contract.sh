#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
analyzer="$repo_root/scripts/analyze_nccl_device_trace_gate.py"

test -f "$analyzer"
rg -q 'runtimeDisabledRoot' "$analyzer"
rg -q 'traceRoot' "$analyzer"
rg -q '95% CI|95%.*CI|confidence' "$analyzer"
rg -q 'maxSampledWorks|droppedRecords|overflow' "$analyzer"
rg -q 'clean.*overhead|cleanOverhead' "$analyzer"
rg -q 'concurrent.*slowdown|slowdown' "$analyzer"
rg -q 'coveredByTelemetry|markerCountBegin' "$analyzer"

echo "NCCL device trace gate analyzer contract passed"
