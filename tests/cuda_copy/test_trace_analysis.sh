#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_p2p_trace.py"
trace_dir="${repo_root}/doc/traces/gpu-contention/phase-lock-diagnostic"
tmp_dir="$(mktemp -d -t cuda-copy-trace-analysis.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${analyzer}" ]] || fail "missing analyzer: ${analyzer}"

run_check() {
  local trace_name="$1"
  local expected_slow="$2"
  local expected_second="$3"
  local output="${tmp_dir}/${trace_name}.json"
  local sqlite="${trace_dir}/${trace_name}.sqlite"

  [[ -f "${sqlite}" ]] || fail "missing SQLite trace: ${sqlite}"
  python3 "${analyzer}" --sqlite="${sqlite}" --output="${output}" >/dev/null
  python3 - "${output}" "${expected_slow}" "${expected_second}" <<'PY'
import json
import sys

path, expected_slow, expected_second = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    result = json.load(stream)

assert result["p2pCount"] == 3720, result
assert result["slowCount"] == int(expected_slow), result
assert result["slowByQueuePosition"][1]["count"] == int(expected_second), result
assert result["slowByQueuePosition"][1]["fraction"] > 0.90 if expected_second != "0" else result["slowByQueuePosition"][1]["fraction"] == 0.0
assert result["p2pRows"][0]["round"] == 0, result["p2pRows"][0]
assert {"start", "end", "durationMs", "round", "queuePosition"}.issubset(
    result["p2pRows"][0]
), result["p2pRows"][0]
assert len(result["firstMeasuredStartBySource"]) == 4, result
assert all(
    entry["relativeStartMs"] >= 0.0
    for entry in result["firstMeasuredStartBySource"]
), result["firstMeasuredStartBySource"]
assert "sourceOffsetDelayKernels" in result, result
if int(expected_slow) > 0:
    assert result["slowWaveCount"] > 0, result["slowWaveCount"]
    assert result["slowWaveGpuCountStats"]["max"] >= 1, result["slowWaveGpuCountStats"]
else:
    assert result["slowWaveCount"] == 0, result["slowWaveCount"]
PY
}

run_check source-major-per-source-d2h 262 242
run_check destination-major-per-source-d2h 265 245
run_check source-major-per-edge-d2h 0 0

echo "PASS: SQLite P2P trace analysis"
