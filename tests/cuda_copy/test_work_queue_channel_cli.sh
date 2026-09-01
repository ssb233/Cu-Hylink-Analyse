#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_work_queue_channel_diagnostic.sh"
tmp_dir="$(mktemp -d -t cuda-copy-work-queue-cli.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing Stage F runner: ${runner}"

runner_help="$(${runner} --help 2>&1)" || fail "runner --help failed"
grep -F -- "--deviceList=0,1,2" <<<"${runner_help}" >/dev/null || \
  fail "three-GPU device-list default is not documented"
grep -F -- "--connectionValues=unset,1,2,4,8,16,32" <<<"${runner_help}" >/dev/null || \
  fail "connection scan options are not documented"
grep -F -- "single-two-copy" <<<"${runner_help}" >/dev/null || \
  fail "single-two-copy topology is not documented"

smoke_root="${tmp_dir}/smoke"
if ! "${runner}" \
  --deviceList=0,1,2 \
  --connectionValues=unset \
  --topologies=single-two-copy,edge-independent,edge-source-chain \
  --scenarios=none \
  --runs=1 \
  --repeats=2 \
  --outputRoot="${smoke_root}" >/dev/null 2>&1; then
  fail "three-GPU Stage F smoke matrix failed"
fi

grep -F -- "deviceList=0,1,2" "${smoke_root}/environment.txt" >/dev/null || \
  fail "environment did not record the three-GPU device list"
single_command="${smoke_root}/topology-single-two-copy/connection-unset/scenario-none/repetition-1/command.txt"
edge_command="${smoke_root}/topology-edge-independent/connection-unset/scenario-none/repetition-1/command.txt"
chain_command="${smoke_root}/topology-edge-source-chain/connection-unset/scenario-none/repetition-1/command.txt"
grep -F -- '--deviceList=0\,1\,2' "${single_command}" >/dev/null || \
  fail "single-two-copy command did not use three GPUs"
grep -F -- "--streamsPerSource=1" "${single_command}" >/dev/null || \
  fail "single-two-copy was not mapped to one stream per source"
grep -F -- "--streamMode=per-edge" "${edge_command}" >/dev/null || \
  fail "edge-independent was not mapped to per-edge streams"
grep -F -- "--streamDependency=source-chain" "${chain_command}" >/dev/null || \
  fail "edge-source-chain dependency was not recorded"

[[ "$(wc -l < "${smoke_root}/summary.csv")" -eq 4 ]] || \
  fail "summary does not contain one header plus three topology rows"
[[ "$(grep -c ',pass,' "${smoke_root}/summary.csv")" -eq 3 ]] || \
  fail "summary does not contain three passing topology rows"
python3 - "${smoke_root}/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
none_rows = [row for row in rows if row["scenario"] == "none"]
assert len(none_rows) == 3, none_rows
assert all(row["backgroundJson"] == "NA" for row in none_rows), none_rows
assert all(row["backgroundLog"] == "NA" for row in none_rows), none_rows
PY

if "${runner}" --deviceList=0,1,2,3 --connectionValues=unset \
  --topologies=single-two-copy --scenarios=none --runs=1 --repeats=2 \
  --outputRoot="${tmp_dir}/four-gpu" >/dev/null 2>&1; then
  fail "runner accepted a four-GPU Stage F device list"
fi

echo "PASS: three-GPU Stage F runner CLI contract"
