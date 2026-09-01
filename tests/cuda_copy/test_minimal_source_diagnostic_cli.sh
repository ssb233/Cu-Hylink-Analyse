#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
benchmark="${repo_root}/build/cuda_copy/minimal_source_pair_bw"
runner="${repo_root}/scripts/run_minimal_source_diagnostic.sh"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

[[ -x "${runner}" ]] || { echo "missing ${runner}" >&2; exit 1; }
help_text="$("${runner}" --help)"
grep -F -- '--backgroundSets=none,0,1,2,all' <<<"${help_text}" >/dev/null
grep -F -- '--edgeOrders=forward,reverse' <<<"${help_text}" >/dev/null
grep -F -- '--streamModes=shared,independent' <<<"${help_text}" >/dev/null

if "${runner}" --deviceList=0,1,2,3 --runs=1 --repeats=2 \
    --outputRoot="${temp_root}/invalid" >/dev/null 2>&1; then
  echo "runner accepted an invalid four-GPU list" >&2
  exit 1
fi

[[ -x "${benchmark}" ]] || { echo "missing ${benchmark}" >&2; exit 1; }
"${benchmark}" --deviceList=0,1,2 \
  --edgeOrder='0->1,0->2' --streamMode=independent \
  --streamDependency=source-chain --repeats=2 --size=1M \
  --output="${temp_root}/benchmark.json" >/dev/null

python3 - "${temp_root}/benchmark.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
assert data["devices"] == [0, 1, 2]
assert data["edgeOrder"] == ["0->1", "0->2"]
assert data["streamMode"] == "independent"
assert data["streamDependency"] == "source-chain"
assert data["warmup"] == 10 and data["repeats"] == 2
assert len(data["edgeResults"]) == 2
assert {(row["source"], row["destination"]) for row in data["edgeResults"]} == {
    (0, 1), (0, 2)
}
assert len(data["sourceResults"]) == 1
assert data["sourceResults"][0]["device"] == 0
PY

output_root="${temp_root}/runner"
"${runner}" \
  --deviceList=0,1,2 \
  --backgroundSets=none,0,1,2,all \
  --sizes=1M \
  --edgeOrders=forward,reverse \
  --streamModes=shared \
  --streamDependencies=none \
  --runs=1 --repeats=2 \
  --outputRoot="${output_root}" >/dev/null

[[ -s "${output_root}/summary.csv" ]]
[[ "$(wc -l < "${output_root}/summary.csv")" -eq 11 ]]
[[ "$(rg -c ',pass,' "${output_root}/summary.csv")" -eq 10 ]]
rg -q '^1M,none,shared,none,forward,' "${output_root}/summary.csv"
rg -q '^1M,all,shared,none,reverse,' "${output_root}/summary.csv"

echo "PASS: minimal source diagnostic CLI contract"
