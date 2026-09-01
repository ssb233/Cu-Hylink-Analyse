#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_directional_dma_trace.py"
temp_root="$(mktemp -d -t directional-dma-trace-analysis.XXXXXX)"
trap 'rm -rf "${temp_root}"' EXIT

victim="${temp_root}/victim.csv"
background="${temp_root}/background.csv"
output="${temp_root}/analysis.json"

printf '%s\n' \
  'pid,deviceId,contextId,streamId,channelID,channelType,copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,durationMs,correlationId,activityKind' \
  '100,0,1,33,16,2,10,0,1,100,100,200,0.000100,10,22' \
  '100,0,1,33,16,2,10,0,2,100,300,400,0.000100,11,22' \
  '100,0,1,33,16,2,10,0,1,100,500,600,0.000100,12,22' \
  '100,0,1,33,16,2,10,0,2,100,800,900,0.000100,13,22' \
  '100,0,1,33,16,2,10,0,1,100,1000,10001000,10.000000,14,22' \
  '100,0,1,33,16,2,10,0,2,100,10002000,10003000,0.001000,15,22' \
  >"${victim}"

printf '%s\n' \
  'pid,deviceId,contextId,streamId,channelID,channelType,copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,durationMs,correlationId,activityKind' \
  '200,0,1,15,12,2,2,-1,-1,100,900,10001000,10.000100,20,1' \
  >"${background}"

python3 "${analyzer}" --victim "${victim}" --background "${background}" \
  --edge-order forward --warmup 1 --repeats 2 --output "${output}"

python3 - "${output}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)

assert data["p2p"]["measuredCount"] == 4, data
assert data["p2p"]["slowCount"] == 1, data
assert data["p2p"]["byQueuePosition"]["1"]["count"] == 2, data
assert data["p2p"]["byQueuePosition"]["2"]["count"] == 2, data
edge = data["p2p"]["byEdge"]["0->1"]
assert edge["queuePosition"] == 1, edge
assert edge["correlationIdRange"] == [12, 14], edge
assert edge["backgroundOverlapCount"] == 1, edge
assert edge["backgroundOverlapNs"] == 10000000, edge
activity = [
    item for item in data["p2p"]["activities"]
    if item["correlationId"] == 14
][0]
assert activity["queuePosition"] == 1, activity
assert activity["previousActivityGapNs"] == 100, activity
assert activity["backgroundOverlapNs"] == 10000000, activity
assert activity["scopedChannel"] == "pid=100/device=0/context=1/type=2/channel=16", activity
PY

echo "PASS: directional DMA trace analysis contract"
