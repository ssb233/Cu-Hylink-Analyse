#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_q3_edge_permutation_matrix.sh"
trace_runner="${repo_root}/scripts/run_q1_trace_case.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing Q3 runner: ${runner}"
[[ -x "${trace_runner}" ]] || fail "missing trace helper: ${trace_runner}"

runner_help="$(${runner} --help 2>&1)" || fail "Q3 runner --help failed"
grep -F -- "--deviceList=0,1,2,3" <<<"${runner_help}" >/dev/null || fail "device list help missing"
grep -F -- "--permutations=0,1,2" <<<"${runner_help}" >/dev/null || fail "permutation help missing"
grep -F -- "--repeatsList=20,300" <<<"${runner_help}" >/dev/null || fail "repeats help missing"
grep -F -- "--scenarios=none,d2h-all" <<<"${runner_help}" >/dev/null || fail "scenarios help missing"

invalid_root="$(mktemp -d -t q3-edge-permutation-invalid.XXXXXX)"
trap 'rm -rf "${invalid_root}"' EXIT
if "${runner}" --runs=0 --outputRoot="${invalid_root}/runs" \
  >/dev/null 2>&1; then
  fail "Q3 runner accepted zero runs"
fi
if "${runner}" --permutations=0,0,1 --outputRoot="${invalid_root}/permutation" \
  >/dev/null 2>&1; then
  fail "Q3 runner accepted invalid permutation"
fi
if "${runner}" --scenarios=invalid --outputRoot="${invalid_root}/scenario" \
  >/dev/null 2>&1; then
  fail "Q3 runner accepted invalid scenario"
fi

trace_help="$(${trace_runner} --help 2>&1)" || fail "trace helper --help failed"
grep -F -- "--edgePermutation=0,1,2" <<<"${trace_help}" >/dev/null || fail "trace permutation help missing"

echo "PASS: Q3 edge-permutation runner CLI contracts"
