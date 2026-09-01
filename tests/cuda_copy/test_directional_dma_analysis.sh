#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_directional_dma_diagnostic.py"
temp_root="$(mktemp -d -t directional-dma-analysis.XXXXXX)"
trap 'rm -rf "${temp_root}"' EXIT

summary="${temp_root}/summary.csv"
printf '%s\n' \
  'backgroundSize,victimSize,direction,dutyCycle,backgroundSet,edgeOrder,topology,repetition,deviceList,victimAggregateGBps,backgroundAggregateGBps,targetDuty,victimExit,backgroundExit,status,backgroundJson,victimJson,victimLog,backgroundLog,traceBackground,traceVictim' \
  '64K,255M,none,NA,none,forward,shared,1,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '64K,255M,none,NA,none,forward,shared,2,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '64K,255M,none,NA,none,forward,shared,3,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,none,NA,none,forward,shared,1,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,none,NA,none,forward,shared,2,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,none,NA,none,forward,shared,3,"0,1,2",100,NA,NA,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '64K,255M,d2h,1.0,0,forward,shared,1,"0,1,2",90,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '64K,255M,d2h,1.0,0,forward,shared,2,"0,1,2",90,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '64K,255M,d2h,1.0,0,forward,shared,3,"0,1,2",90,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,d2h,1.0,0,forward,shared,1,"0,1,2",80,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,d2h,1.0,0,forward,shared,2,"0,1,2",80,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  '128K,255M,d2h,1.0,0,forward,shared,3,"0,1,2",80,1.0,1.0,0,0,pass,NA,NA,NA,NA,NA,NA' \
  >"${summary}"

python3 "${analyzer}" "${temp_root}" >/dev/null
[[ -s "${temp_root}/analysis.json" ]]
[[ -s "${temp_root}/summary.md" ]]
python3 - "${temp_root}/analysis.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)
assert data["parsed_rows"] == 12, data
assert data["treatment_rows"] == 6, data
assert len(data["threshold_candidates"]) == 1, data
candidate = data["threshold_candidates"][0]
assert candidate["backgroundSet"] == "0", candidate
assert candidate["first_drop_size"] == "64K", candidate
assert candidate["next_drop_size"] == "128K", candidate
assert abs(candidate["first_drop_pct"] - 10.0) < 1e-6, candidate
assert abs(candidate["next_drop_pct"] - 20.0) < 1e-6, candidate
PY
rg -q '64K.*10\.00%' "${temp_root}/summary.md"

echo "PASS: directional DMA analysis contract"
