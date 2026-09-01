#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_work_queue_channel.py"
tmp_dir="$(mktemp -d -t cuda-copy-work-queue-analysis.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${analyzer}" ]] || fail "missing Stage F analyzer: ${analyzer}"

python3 - "${tmp_dir}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
clean_json = root / "clean.json"
d2h_json = root / "d2h.json"
background_json = root / "background.json"
clean_json.write_text(json.dumps({
    "aggregateGBps": 100.0,
    "devices": [0, 1, 2],
    "sourceResults": [
        {"device": 0, "GBps": 33.0},
        {"device": 1, "GBps": 34.0},
        {"device": 2, "GBps": 35.0},
    ],
}), encoding="utf-8")
d2h_json.write_text(json.dumps({
    "aggregateGBps": 80.0,
    "devices": [0, 1, 2],
    "sourceResults": [
        {"device": 0, "GBps": 26.0},
        {"device": 1, "GBps": 27.0},
        {"device": 2, "GBps": 27.0},
    ],
}), encoding="utf-8")
background_json.write_text(json.dumps({
    "aggregateGBps": 12.0,
    "devices": [0, 1, 2],
    "perDeviceBytes": [100, 200, 300],
    "perDeviceGBps": [1.0, 2.0, 3.0],
    "totalBytes": 600,
}), encoding="utf-8")
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow([
        "topology", "connectionValue", "scenario", "repetition", "deviceList",
        "d2dSize", "backgroundSize", "d2dAggregateGBps",
        "backgroundAggregateGBps", "d2dExit", "backgroundExit", "status",
        "d2dJson", "backgroundJson", "d2dLog", "backgroundLog",
        "traceD2d", "traceBackground",
    ])
    writer.writerow([
        "single-two-copy", "unset", "none", 1, "0,1,2", "255M", "255M",
        100.0, "NA", 0, 0, "pass", clean_json, "NA", "clean.log", "NA",
        "NA", "NA",
    ])
    writer.writerow([
        "single-two-copy", "unset", "d2h-all", 1, "0,1,2", "255M", "255M",
        80.0, 12.0, 0, 0, "pass", d2h_json, background_json, "d2h.log",
        "background.log", "NA", "NA",
    ])
PY

analysis_json="$(${analyzer} --summary "${tmp_dir}/summary.csv")" || fail "analyzer execution failed"
python3 - "${analysis_json}" <<'PY'
import json
import sys

result = json.loads(sys.argv[1])
adaptation = result["threeGpuAdaptation"]
assert adaptation["devices"] == 3, adaptation
assert adaptation["singleTwoCopyCopiesPerSource"] == 2, adaptation
groups = {
    (item["topology"], item["connectionValue"], item["scenario"]): item
    for item in result["groups"]
}
clean = groups[("single-two-copy", "unset", "none")]
d2h = groups[("single-two-copy", "unset", "d2h-all")]
assert clean["caseCount"] == 1, clean
assert clean["d2dAggregateGBps"]["mean"] == 100.0, clean
assert d2h["d2dAggregateGBps"]["mean"] == 80.0, d2h
assert d2h["backgroundPerDeviceGBps"]["meanByDevice"] == {
    "0": 1.0, "1": 2.0, "2": 3.0
}, d2h
PY

echo "PASS: Stage F work-queue analysis contract"
