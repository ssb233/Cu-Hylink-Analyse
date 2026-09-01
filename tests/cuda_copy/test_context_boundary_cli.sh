#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_context_boundary_diagnostic.sh"
tmp_dir="$(mktemp -d -t cuda-copy-context-boundary.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing context-boundary runner: ${runner}"

help_output="$(${runner} --help)" || fail "runner --help failed"
grep -q -- '--contextModes=two-process,same-process' <<< "${help_output}" \
  || fail "help does not expose context modes"
grep -q -- 'default=two-process' <<< "${help_output}" \
  || fail "help does not document the two-process default"

if "${runner}" --contextModes=unknown --outputRoot="${tmp_dir}/invalid" \
    >/dev/null 2>&1; then
  fail "unknown context mode was accepted"
fi

output_root="${tmp_dir}/same-process"
"${runner}" \
  --deviceList=0,1,2 \
  --contextModes=same-process \
  --topologies=single-two-copy \
  --scenarios=none \
  --runs=1 \
  --repeats=2 \
  --outputRoot="${output_root}" \
  >"${tmp_dir}/runner.log"

grep -q '^contextModes=same-process$' "${output_root}/environment.txt" \
  || fail "same-process mode was not recorded"
[[ "$(wc -l < "${output_root}/summary.csv")" -eq 2 ]] \
  || fail "unexpected smoke summary row count"
python3 - "${output_root}/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1, rows
assert rows[0]['contextMode'] == 'same-process', rows[0]
assert rows[0]['status'] == 'pass', rows[0]
assert rows[0]['resultJson'] != 'NA', rows[0]
PY

echo "PASS: context-boundary runner CLI contract"
