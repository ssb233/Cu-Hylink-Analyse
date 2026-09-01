#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_copy_path_temporal_ablation.py"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

cat > "${temp_root}/background.json" <<'EOF'
{
  "program": "copy_path_background",
  "path": "streaming-hbm-read",
  "devices": [0, 1, 2],
  "requestedBytes": 1048576,
  "workingSetBytes": 1048576,
  "targetGBps": 4.0,
  "dutyCycle": 0.1,
  "elapsedSec": 1.0,
  "totalBytes": 300,
  "perDeviceBytes": [100, 100, 100],
  "perDeviceOperations": [2, 2, 2],
  "perDeviceWallActiveSec": [0.2, 0.2, 0.2],
  "perDeviceWallActiveDuty": [0.2, 0.2, 0.2],
  "perDeviceGpuActivitySec": [null, null, null],
  "perDeviceGpuActivityDuty": [null, null, null],
  "perDeviceOperationDurationMs": [
    {"count": 2, "p50": 10.0, "p90": 12.0, "p99": 12.0, "max": 12.0},
    {"count": 2, "p50": 10.0, "p90": 12.0, "p99": 12.0, "max": 12.0},
    {"count": 2, "p50": 10.0, "p90": 12.0, "p99": 12.0, "max": 12.0}
  ],
  "perDeviceSubmitIntervalMs": [
    {"count": 1, "p50": 50.0, "p90": 50.0, "p99": 50.0, "max": 50.0},
    {"count": 1, "p50": 50.0, "p90": 50.0, "p99": 50.0, "max": 50.0},
    {"count": 1, "p50": 50.0, "p90": 50.0, "p99": 50.0, "max": 50.0}
  ],
  "perDeviceIdleGapMs": [
    {"count": 1, "p50": 40.0, "p90": 40.0, "p99": 40.0, "max": 40.0},
    {"count": 1, "p50": 40.0, "p90": 40.0, "p99": 40.0, "max": 40.0},
    {"count": 1, "p50": 40.0, "p90": 40.0, "p99": 40.0, "max": 40.0}
  ]
}
EOF

cat > "${temp_root}/victim-clean.json" <<'EOF'
{"program":"d2d_multi_peer_bw","devices":[0,1,2],"aggregateGBps":100.0,
 "sourceResults":[{"device":0,"GBps":33.3},{"device":1,"GBps":33.3},{"device":2,"GBps":33.4}],
 "p2pSlowCount":0}
EOF
cat > "${temp_root}/victim-d2h.json" <<'EOF'
{"program":"d2d_multi_peer_bw","devices":[0,1,2],"aggregateGBps":60.0,
 "sourceResults":[{"device":0,"GBps":20.0},{"device":1,"GBps":20.0},{"device":2,"GBps":20.0}],
 "p2pSlowCount":1}
EOF

cat > "${temp_root}/trace.csv" <<'EOF'
pid,deviceId,contextId,streamId,channelID,channelType,copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,durationMs,correlationId,activityKind
77,0,9,3,5,2,10,0,1,255,1000000000,1010000000,10.0,101,21
77,0,9,4,6,2,2,0,-1,255,1005000000,1070000000,20.0,102,20
77,1,9,3,5,2,10,1,2,255,2000000000,2005000000,5.0,103,21
EOF
cat > "${temp_root}/trace.csv.meta.json" <<'EOF'
{"droppedRecords":0}
EOF

cat > "${temp_root}/summary.csv" <<EOF
pressureLevel,backgroundPath,victimMode,topology,repetition,deviceList,targetGBps,dutyCycle,bandwidthClass,victimAggregateGBps,backgroundAggregateGBps,status,backgroundJson,victimJson,traceBackground,traceVictim
clean,none,original-p2p-ce,single-two-copy,1,"0,1,2",0,1.0,bandwidth-matched,100.0,NA,pass,NA,${temp_root}/victim-clean.json,NA,NA
duty-0.1,streaming-hbm-read,original-p2p-ce,single-two-copy,1,"0,1,2",4.0,0.1,duty-matched,60.0,0.0003,pass,${temp_root}/background.json,${temp_root}/victim-d2h.json,${temp_root}/trace.csv,${temp_root}/trace.csv
EOF

[[ -x "${analyzer}" ]] || { echo "missing ${analyzer}" >&2; exit 1; }
python3 "${analyzer}" --summary "${temp_root}/summary.csv" \
  --output "${temp_root}/analysis.json"

python3 - "${temp_root}/analysis.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
assert data["deviceList"] == [0, 1, 2]
assert data["totalRows"] == 2 and data["ignoredRows"] == 0
clean_case = next(item for item in data["cases"] if item["backgroundPath"] == "none")
assert clean_case["p2pSlowCount"] is None
case = next(item for item in data["cases"] if item["pressureLevel"] == "duty-0.1")
assert case["backgroundWallActiveDuty"] == [0.2, 0.2, 0.2]
assert case["backgroundOperationDurationMs"][0]["p99"] == 12.0
assert case["p2pSlowCount"] == 1
assert case["p2pBackgroundOverlapNs"] > 0
group = next(item for item in data["groups"] if item["backgroundPath"] != "none")
assert group["cleanToTreatmentDropPct"] == 40.0
assert data["threeGpuAdaptation"]["singleTwoCopyDescription"]
print("PASS: temporal ablation analysis contract")
PY
