#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_copy_path_ablation.py"
tmp_dir="$(mktemp -d -t cuda-copy-path-analysis.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${analyzer}" ]] || fail "missing copy-path analyzer: ${analyzer}"

python3 - "${tmp_dir}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
victim = root / 'victim.json'
background = root / 'background.json'
victim.write_text(json.dumps({
    'aggregateGBps': 100.0,
    'devices': [0, 1, 2],
    'sourceResults': [
        {'device': 0, 'GBps': 33.0},
        {'device': 1, 'GBps': 34.0},
        {'device': 2, 'GBps': 33.0},
    ],
    'streamCount': 3,
}), encoding='utf-8')
background.write_text(json.dumps({
    'path': 'streaming-hbm-read',
    'devices': [0, 1, 2],
    'totalBytes': 600,
    'aggregateGBps': 12.0,
    'perDeviceBytes': [100, 200, 300],
    'perDeviceGBps': [2.0, 4.0, 6.0],
    'perDeviceOperations': [1, 2, 3],
    'workingSetBytes': 100,
}), encoding='utf-8')
with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'backgroundPath', 'victimMode', 'topology', 'repetition', 'deviceList',
        'victimSize', 'backgroundSize', 'targetGBps', 'dutyCycle',
        'victimAggregateGBps', 'backgroundAggregateGBps', 'victimExit',
        'backgroundExit', 'status', 'backgroundJson', 'victimJson',
        'victimLog', 'backgroundLog',
    ])
    writer.writerow([
        'streaming-hbm-read', 'original-p2p-ce', 'single-two-copy', 1,
        '0,1,2', '255M', '255M', 4.0, 1.0, 100.0, 12.0, 0, 0, 'pass',
        background, victim, root / 'victim.log', root / 'background.log',
    ])
PY

analysis_json="$(python3 "${analyzer}" --summary "${tmp_dir}/summary.csv")" \
  || fail "copy-path analyzer execution failed"
python3 - "${analysis_json}" <<'PY'
import json
import sys

result = json.loads(sys.argv[1])
assert result['caseCount'] == 1, result
assert result['deviceList'] == [0, 1, 2], result
group = result['groups'][0]
assert group['backgroundPath'] == 'streaming-hbm-read', group
assert group['victimMode'] == 'original-p2p-ce', group
assert group['victimAggregateGBps']['mean'] == 100.0, group
assert group['backgroundPerDeviceGBps']['meanByDevice'] == {
    '0': 2.0, '1': 4.0, '2': 6.0
}, group
PY

echo "PASS: copy-path ablation analysis contract"
