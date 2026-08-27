#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
runner="${repo_root}/scripts/run_stream_assignment_matrix.sh"
trace_runner="${repo_root}/scripts/run_q1_trace_case.sh"
tmp_dir="$(mktemp -d -t cuda-copy-stream-assignment-cli.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${multi_bin}" ]] || fail "missing executable: ${multi_bin}"

multi_help="$(${multi_bin} --help 2>&1)" || fail "benchmark --help failed"
grep -F -- "--streamAssignment=0,1,1|1,0,1|1,1,0" <<<"${multi_help}" >/dev/null || \
  fail "stream assignment help missing"

valid_log="${tmp_dir}/valid.log"
if ! "${multi_bin}" --pattern=allpairs --deviceList=0,1,2,3 \
  --streamsPerSource=2 --streamAssignment=0,1,1 --repeats=1 --size=1M \
  --output="${tmp_dir}/valid.json" >"${valid_log}" 2>&1; then
  cat "${valid_log}" >&2
  fail "valid one-vs-two stream assignment was rejected"
fi
grep -F -- "streamAssignment=0,1,1" "${valid_log}" >/dev/null || \
  fail "stream assignment was not reported"
grep -F -- '"streamAssignment": [0, 1, 1]' "${tmp_dir}/valid.json" >/dev/null || \
  fail "stream assignment was not recorded in JSON"

if "${multi_bin}" --pattern=allpairs --deviceList=0,1,2,3 \
  --streamsPerSource=2 --streamAssignment=0,0,0 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "invalid all-on-one-stream assignment was accepted"
fi
if "${multi_bin}" --pattern=allpairs --deviceList=0,1,2,3 \
  --streamsPerSource=1 --streamAssignment=0,1,1 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "stream assignment was accepted with streamsPerSource=1"
fi

[[ -x "${runner}" ]] || fail "missing stream-assignment runner: ${runner}"
runner_help="$(${runner} --help 2>&1)" || fail "runner --help failed"
grep -F -- "--assignments=0,1,1:1,0,1:1,1,0" <<<"${runner_help}" >/dev/null || \
  fail "runner assignment help missing"
grep -F -- "--d2dSize=255M" <<<"${runner_help}" >/dev/null || \
  fail "runner D2D size help missing"
grep -F -- "--backgroundSize=255M" <<<"${runner_help}" >/dev/null || \
  fail "runner background size help missing"

split_size_root="${tmp_dir}/split-size"
if ! "${runner}" --assignments=0,1,1 --repeatsList=1 \
  --scenarios=none,d2h-all --runs=1 --d2dSize=1M --backgroundSize=16K \
  --outputRoot="${split_size_root}" >/dev/null 2>&1; then
  fail "runner rejected independent D2D/background sizes"
fi
grep -F -- --size=1M "${split_size_root}/assignment-011/repeats-1/repetition-1/none/command.txt" >/dev/null || \
  fail "D2D command did not use --d2dSize"
grep -F -- --size=16K "${split_size_root}/assignment-011/repeats-1/repetition-1/d2h-all/command.txt" >/dev/null || \
  fail "background command did not use --backgroundSize"
grep -F -- "d2dSize=1M" "${split_size_root}/environment.txt" >/dev/null || \
  fail "environment did not record D2D size"
grep -F -- "backgroundSize=16K" "${split_size_root}/environment.txt" >/dev/null || \
  fail "environment did not record background size"
[[ "$(wc -l < "${split_size_root}/summary.csv")" -eq 3 ]] || \
  fail "summary row count is not one row per matrix case"
[[ "$(grep -c ',pass,' "${split_size_root}/summary.csv")" -eq 2 ]] || \
  fail "summary does not contain one pass row per matrix case"
if "${runner}" --assignments=0,0,0 --outputRoot="${tmp_dir}/bad-assignment" \
  >/dev/null 2>&1; then
  fail "runner accepted an all-on-one-stream assignment"
fi

[[ -x "${trace_runner}" ]] || fail "missing trace helper: ${trace_runner}"
trace_help="$(${trace_runner} --help 2>&1)" || fail "trace helper --help failed"
grep -F -- "--streamAssignment=0,1,1" <<<"${trace_help}" >/dev/null || \
  fail "trace stream assignment help missing"
if "${trace_runner}" --streamsPerSource=2 --streamAssignment=0,0,0 \
  --scenario=none >/dev/null 2>&1; then
  fail "trace helper accepted an all-on-one-stream assignment"
fi
if "${trace_runner}" --streamsPerSource=1 --streamAssignment=0,1,1 \
  --scenario=none >/dev/null 2>&1; then
  fail "trace helper accepted assignment with streamsPerSource=1"
fi

echo "PASS: two-stream assignment CLI contracts"
