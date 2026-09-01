#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
analyzer="${repo_root}/scripts/analyze_copy_channel_trace.py"
tmp_dir="$(mktemp -d -t cuda-copy-channel-analysis.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${analyzer}" ]] || fail "missing memcpy channel analyzer: ${analyzer}"

cat > "${tmp_dir}/trace.csv" <<'EOF'
pid,deviceId,contextId,streamId,channelID,channelType,copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,durationMs,correlationId,activityKind
100,0,11,21,3,1,10,0,1,267386880,1000,2000000,1.999,41,22
100,0,11,22,4,1,10,0,2,267386880,3000,13003000,13.000,42,22
100,0,11,23,5,1,2,-1,-1,267386880,4000,5000000,4.996,43,20
EOF
cat > "${tmp_dir}/trace.csv.meta.json" <<'EOF'
{"droppedRecords": 0}
EOF

analysis_json="$(python3 "${analyzer}" --trace "${tmp_dir}/trace.csv")" \
  || fail "channel analyzer execution failed"

python3 - "${analysis_json}" <<'PY'
import json
import sys

result = json.loads(sys.argv[1])
assert result["droppedRecords"] == 0, result
assert result["totalRows"] == 3, result
assert result["kindTotals"]["p2p"]["count"] == 2, result
assert result["kindTotals"]["p2p"]["slowCount"] == 1, result
assert result["kindTotals"]["d2h"]["count"] == 1, result
assert result["kindTotals"]["d2h"]["slowCount"] == 0, result
assert result["slowByChannel"][0]["channelID"] == 4, result
assert result["slowByChannel"][0]["slowCount"] == 1, result
assert result["slowByChannel"][0]["pSlowGivenChannel"] == 1.0, result
assert result["slowByChannel"][0]["pChannelGivenSlowP2p"] == 1.0, result
assert result["streamCount"] == 3, result
PY

echo "PASS: CUPTI memcpy channel analysis contract"
