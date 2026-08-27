#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_q4_source_offset_matrix.sh"
trace_runner="${repo_root}/scripts/run_q1_trace_case.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing Q4 runner: ${runner}"
[[ -x "${trace_runner}" ]] || fail "missing trace helper: ${trace_runner}"

runner_help="$(${runner} --help 2>&1)" || fail "Q4 runner --help failed"
grep -F -- "--offsetsList=0,0,0,0" <<<"${runner_help}" >/dev/null || fail "offset list help missing"
grep -F -- "--repeatsList=20,300" <<<"${runner_help}" >/dev/null || fail "repeats help missing"
grep -F -- "--scenarios=none,d2h-all" <<<"${runner_help}" >/dev/null || fail "scenarios help missing"

invalid_root="$(mktemp -d -t q4-source-offset-invalid.XXXXXX)"
trap 'rm -rf "${invalid_root}"' EXIT
if "${runner}" --runs=0 --outputRoot="${invalid_root}/runs" \
  >/dev/null 2>&1; then
  fail "Q4 runner accepted zero runs"
fi
if "${runner}" --offsetsList=0,1,2 --outputRoot="${invalid_root}/offsets" \
  >/dev/null 2>&1; then
  fail "Q4 runner accepted a short source offset list"
fi
if "${runner}" --scenarios=invalid --outputRoot="${invalid_root}/scenario" \
  >/dev/null 2>&1; then
  fail "Q4 runner accepted invalid scenario"
fi

trace_help="$(${trace_runner} --help 2>&1)" || fail "trace helper --help failed"
grep -F -- "--sourceOffsetsUs=0,250,500,750" <<<"${trace_help}" >/dev/null || fail "trace source-offset help missing"
if "${trace_runner}" --streamMode=per-source --streamsPerSource=1 \
  --sourceOffsetsUs=0,1,2 --scenario=none >/dev/null 2>&1; then
  fail "trace helper accepted a short source offset list"
fi
if "${trace_runner}" --streamsPerSource=2 \
  --sourceOffsetsUs=0,1,2,3 --scenario=none >/dev/null 2>&1; then
  fail "trace helper accepted source offsets with multiple streams"
fi

echo "PASS: Q4 source-offset runner CLI contracts"
