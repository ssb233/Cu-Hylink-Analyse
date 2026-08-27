#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_q1_source_chain_matrix.sh"
trace_runner="${repo_root}/scripts/run_q1_trace_case.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing Q1 runner: ${runner}"
[[ -x "${trace_runner}" ]] || fail "missing Q1 trace runner: ${trace_runner}"

runner_help="$(${runner} --help 2>&1)" || fail "Q1 runner --help failed"
grep -F -- "--deviceList=0,1,2,3" <<<"${runner_help}" >/dev/null || fail "device list help missing"
grep -F -- "--edgeOrder=source-major" <<<"${runner_help}" >/dev/null || fail "edge order help missing"
grep -F -- "--repeatsList=20,300" <<<"${runner_help}" >/dev/null || fail "repeats help missing"
grep -F -- "--scenarios=none,d2h-all" <<<"${runner_help}" >/dev/null || fail "scenarios help missing"
grep -F -- "per-source + none" <<<"${runner_help}" >/dev/null || fail "per-source configuration missing"
grep -F -- "per-edge + none" <<<"${runner_help}" >/dev/null || fail "per-edge none configuration missing"
grep -F -- "per-edge + source-chain" <<<"${runner_help}" >/dev/null || fail "source-chain configuration missing"

invalid_root="$(mktemp -d -t q1-source-chain-invalid.XXXXXX)"
trap 'rm -rf "${invalid_root}"' EXIT
if "${runner}" --runs=0 --outputRoot="${invalid_root}/runs" \
  >/dev/null 2>&1; then
  fail "Q1 runner accepted zero runs"
fi
if "${runner}" --edgeOrder=invalid --outputRoot="${invalid_root}/order" \
  >/dev/null 2>&1; then
  fail "Q1 runner accepted invalid edge order"
fi
if "${runner}" --scenarios=invalid --outputRoot="${invalid_root}/scenario" \
  >/dev/null 2>&1; then
  fail "Q1 runner accepted invalid scenario"
fi

trace_help="$(${trace_runner} --help 2>&1)" || fail "Q1 trace runner --help failed"
grep -F -- "--streamMode=per-source|per-edge" <<<"${trace_help}" >/dev/null || fail "trace stream mode help missing"
grep -F -- "--streamDependency=none|source-chain" <<<"${trace_help}" >/dev/null || fail "trace dependency help missing"
grep -F -- "--streamsPerSource=1|2|3" <<<"${trace_help}" >/dev/null || fail "trace stream count help missing"
grep -F -- "--scenario=d2h-all" <<<"${trace_help}" >/dev/null || fail "trace scenario help missing"
if "${trace_runner}" --streamMode=per-source \
  --streamDependency=source-chain --outputRoot="${invalid_root}/trace" \
  >/dev/null 2>&1; then
  fail "Q1 trace runner accepted source-chain with per-source streams"
fi
if "${trace_runner}" --streamMode=per-edge --streamsPerSource=4 \
  --outputRoot="${invalid_root}/trace-count" >/dev/null 2>&1; then
  fail "Q1 trace runner accepted invalid streams-per-source value"
fi

echo "PASS: Q1 source-chain runner CLI contracts"
