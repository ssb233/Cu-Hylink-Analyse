#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/scripts/run_copy_path_ablation.sh"
tmp_dir="$(mktemp -d -t cuda-copy-path-ablation.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${runner}" ]] || fail "missing copy-path runner: ${runner}"
help_output="$(${runner} --help)" || fail "runner --help failed"
for path in original-d2h local-d2d-ce streaming-hbm-read streaming-hbm-write l2-resident-read; do
  grep -q -- "${path}" <<< "${help_output}" \
    || fail "help does not expose background path ${path}"
done
for mode in original-p2p-ce local-d2d-ce peer-read peer-write; do
  grep -q -- "${mode}" <<< "${help_output}" \
    || fail "help does not expose victim mode ${mode}"
done

if "${runner}" --backgroundPaths=unknown --outputRoot="${tmp_dir}/invalid" \
    >/dev/null 2>&1; then
  fail "unknown background path was accepted"
fi

output_root="${tmp_dir}/smoke"
"${runner}" \
  --deviceList=0,1,2 \
  --backgroundPaths=none \
  --victimModes=original-p2p-ce \
  --topologies=single-two-copy \
  --runs=1 \
  --repeats=2 \
  --outputRoot="${output_root}" \
  >"${tmp_dir}/runner.log"

[[ "$(wc -l < "${output_root}/summary.csv")" -eq 2 ]] \
  || fail "unexpected copy-path smoke summary row count"
python3 - "${output_root}/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1, rows
assert rows[0]['backgroundPath'] == 'none', rows[0]
assert rows[0]['victimMode'] == 'original-p2p-ce', rows[0]
assert rows[0]['topology'] == 'single-two-copy', rows[0]
assert rows[0]['status'] == 'pass', rows[0]
assert rows[0]['victimJson'] != 'NA', rows[0]
PY

echo "PASS: copy-path ablation CLI contract"
