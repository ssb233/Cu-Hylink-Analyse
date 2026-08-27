#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_q2_stream_count_matrix.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing Q2 runner: ${runner}"

runner_help="$(${runner} --help 2>&1)" || fail "Q2 runner --help failed"
grep -F -- "--deviceList=0,1,2,3" <<<"${runner_help}" >/dev/null || fail "device list help missing"
grep -F -- "--streamCounts=1,2,3" <<<"${runner_help}" >/dev/null || fail "stream count help missing"
grep -F -- "--repeatsList=20,300" <<<"${runner_help}" >/dev/null || fail "repeats help missing"
grep -F -- "--scenarios=none,d2h-all" <<<"${runner_help}" >/dev/null || fail "scenarios help missing"

invalid_root="$(mktemp -d -t q2-stream-count-invalid.XXXXXX)"
trap 'rm -rf "${invalid_root}"' EXIT
if "${runner}" --runs=0 --outputRoot="${invalid_root}/runs" \
  >/dev/null 2>&1; then
  fail "Q2 runner accepted zero runs"
fi
if "${runner}" --streamCounts=1,4 --outputRoot="${invalid_root}/counts" \
  >/dev/null 2>&1; then
  fail "Q2 runner accepted invalid stream count"
fi
if "${runner}" --scenarios=invalid --outputRoot="${invalid_root}/scenario" \
  >/dev/null 2>&1; then
  fail "Q2 runner accepted invalid scenario"
fi

echo "PASS: Q2 stream-count runner CLI contracts"
