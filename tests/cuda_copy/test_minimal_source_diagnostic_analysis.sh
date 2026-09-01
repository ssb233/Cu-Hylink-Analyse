#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_minimal_source_diagnostic.py"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

cat > "${temp_root}/victim-clean-64K.json" <<'EOF'
{"program":"minimal_source_pair_bw","devices":[0,1,2],"edgeOrder":["0->1","0->2"],
 "streamMode":"shared","streamDependency":"none","repeats":2,"warmup":10,
 "aggregateGBps":100.0,"sourceResults":[{"device":0,"GBps":100.0}],
 "edgeResults":[{"source":0,"destination":1,"order":0,"stream":0,"operations":2,"elapsedMs":20.0,"GBps":100.0,"operationDurationMs":{"count":2,"p50":10.0,"p90":11.0,"p99":11.0,"max":11.0}},
                {"source":0,"destination":2,"order":1,"stream":0,"operations":2,"elapsedMs":20.0,"GBps":100.0,"operationDurationMs":{"count":2,"p50":10.0,"p90":11.0,"p99":11.0,"max":11.0}}]}
EOF
cat > "${temp_root}/victim-bg-64K.json" <<'EOF'
{"program":"minimal_source_pair_bw","devices":[0,1,2],"edgeOrder":["0->1","0->2"],
 "streamMode":"shared","streamDependency":"none","repeats":2,"warmup":10,
 "aggregateGBps":99.0,"sourceResults":[{"device":0,"GBps":99.0}],
 "edgeResults":[{"source":0,"destination":1,"order":0,"stream":0,"operations":2,"elapsedMs":20.2,"GBps":99.0,"operationDurationMs":{"count":2,"p50":10.1,"p90":11.1,"p99":11.1,"max":11.1}},
                {"source":0,"destination":2,"order":1,"stream":0,"operations":2,"elapsedMs":20.2,"GBps":99.0,"operationDurationMs":{"count":2,"p50":10.1,"p90":11.1,"p99":11.1,"max":11.1}}]}
EOF
cat > "${temp_root}/victim-clean-4M.json" <<'EOF'
{"program":"minimal_source_pair_bw","devices":[0,1,2],"edgeOrder":["0->1","0->2"],
 "streamMode":"shared","streamDependency":"none","repeats":2,"warmup":10,
 "aggregateGBps":100.0,"sourceResults":[{"device":0,"GBps":100.0}],
 "edgeResults":[{"source":0,"destination":1,"order":0,"stream":0,"operations":2,"elapsedMs":20.0,"GBps":100.0,"operationDurationMs":{"count":2,"p50":10.0,"p90":11.0,"p99":11.0,"max":11.0}},
                {"source":0,"destination":2,"order":1,"stream":0,"operations":2,"elapsedMs":20.0,"GBps":100.0,"operationDurationMs":{"count":2,"p50":10.0,"p90":11.0,"p99":11.0,"max":11.0}}]}
EOF
cat > "${temp_root}/victim-bg-4M.json" <<'EOF'
{"program":"minimal_source_pair_bw","devices":[0,1,2],"edgeOrder":["0->1","0->2"],
 "streamMode":"shared","streamDependency":"none","repeats":2,"warmup":10,
 "aggregateGBps":50.0,"sourceResults":[{"device":0,"GBps":50.0}],
 "edgeResults":[{"source":0,"destination":1,"order":0,"stream":0,"operations":2,"elapsedMs":40.0,"GBps":50.0,"operationDurationMs":{"count":2,"p50":20.0,"p90":21.0,"p99":21.0,"max":21.0}},
                {"source":0,"destination":2,"order":1,"stream":0,"operations":2,"elapsedMs":40.0,"GBps":50.0,"operationDurationMs":{"count":2,"p50":20.0,"p90":21.0,"p99":21.0,"max":21.0}}]}
EOF

cat > "${temp_root}/summary.csv" <<EOF
size,backgroundSet,streamMode,streamDependency,edgeOrder,repetition,deviceList,victimSize,backgroundSize,victimAggregateGBps,backgroundAggregateGBps,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog,traceBackground,traceVictim
64K,none,shared,none,forward,1,"0,1,2",64K,64K,100.0,NA,0,0,pass,NA,${temp_root}/victim-clean-64K.json,NA,NA,NA,NA
64K,0,shared,none,forward,1,"0,1,2",64K,64K,99.0,10.0,0,0,pass,${temp_root}/background.json,${temp_root}/victim-bg-64K.json,NA,NA,NA,NA
4M,none,shared,none,forward,1,"0,1,2",4M,4M,100.0,NA,0,0,pass,NA,${temp_root}/victim-clean-4M.json,NA,NA,NA,NA
4M,0,shared,none,forward,1,"0,1,2",4M,4M,50.0,10.0,0,0,pass,${temp_root}/background.json,${temp_root}/victim-bg-4M.json,NA,NA,NA,NA
EOF
cat > "${temp_root}/background.json" <<'EOF'
{"devices":[0],"direction":"d2h","bytesPerMemcpy":65536,"totalBytes":100,
 "perDeviceBytes":[100],"perDeviceOperations":[1],"perDeviceWallActiveSec":[0.1],
 "perDeviceWallActiveDuty":[0.5],"perDeviceGpuActivitySec":[null],"perDeviceGpuActivityDuty":[null],
 "perDeviceOperationDurationMs":[{"count":1,"p50":1.0,"p90":1.0,"p99":1.0,"max":1.0}],
 "perDeviceSubmitIntervalMs":[{"count":0,"p50":0.0,"p90":0.0,"p99":0.0,"max":0.0}],
 "perDeviceIdleGapMs":[{"count":0,"p50":0.0,"p90":0.0,"p99":0.0,"max":0.0}]}
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
assert data["sourceSharing"] == {"none": False, "0": True, "1": False, "2": False, "all": True}
assert len(data["cases"]) == 4
case = next(item for item in data["cases"] if item["size"] == "4M" and item["backgroundSet"] == "0")
assert case["edgeGBps"]["0->1"] == 50.0
assert case["backgroundWallActiveDuty"] == [0.5]
sweep = next(item for item in data["sweeps"] if item["backgroundSet"] == "0")
assert sweep["lastImmuneSize"] == "64K"
assert sweep["firstDropSize"] == "4M"
assert sweep["firstDropPct"] == 50.0
print("PASS: minimal source diagnostic analysis contract")
PY
