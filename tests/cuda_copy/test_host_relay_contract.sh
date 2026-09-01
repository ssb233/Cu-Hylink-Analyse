#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
relay_bin="${repo_root}/build/cuda_copy/host_relay"
tmp_dir="$(mktemp -d -t cuda-host-relay.XXXXXX)"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${relay_bin}" ]] || fail "missing executable: ${relay_bin}"

help_output="$(${relay_bin} --help 2>&1)" || fail "--help failed"
grep -F -- "--mode=relay|d2h|h2d" <<<"${help_output}" >/dev/null || \
  fail "mode help missing"
grep -F -- "--edges=0:1,2:3" <<<"${help_output}" >/dev/null || \
  fail "edge help missing"
grep -F -- "--startFile=<path>" <<<"${help_output}" >/dev/null || \
  fail "start-file help missing"
grep -F -- "--telemetryFile=<path>" <<<"${help_output}" >/dev/null || \
  fail "telemetry-file help missing"
grep -F -- "--reportMs=250" <<<"${help_output}" >/dev/null || \
  fail "millisecond report help missing"

if "${relay_bin}" --mode=invalid --edges=0:1 --size=64K --warmup=1 \
  --iterations=1 >/dev/null 2>&1; then
  fail "invalid mode was accepted"
fi

if "${relay_bin}" --mode=relay --edges=0:0 --size=64K --warmup=1 \
  --iterations=1 >/dev/null 2>&1; then
  fail "same-device relay was accepted"
fi

json_file="${tmp_dir}/relay.json"
log_file="${tmp_dir}/relay.log"
"${relay_bin}" --mode=relay --edges=0:1 --size=64K --warmup=1 \
  --iterations=2 --output="${json_file}" >"${log_file}" 2>&1 || {
  cat "${log_file}" >&2
  fail "relay smoke failed"
}

python3 - "${json_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)

assert result["mode"] == "relay", result
assert result["bytesPerIteration"] == 64 * 1024, result
assert result["edges"] == [{"src": 0, "dst": 1}], result
edge = result["perEdge"][0]
assert edge["operations"] == 2, edge
assert edge["bytesCompleted"] == 2 * 64 * 1024, edge
assert edge["d2hMs"]["count"] == 2, edge
assert edge["h2dMs"]["count"] == 2, edge
assert edge["endToEndMs"]["count"] == 2, edge
assert edge["usefulGBps"] > 0, edge
assert edge["trafficGBps"] > edge["usefulGBps"], edge
PY

telemetry_jsonl="${tmp_dir}/relay.telemetry.jsonl"
telemetry_result="${tmp_dir}/relay.telemetry.result.json"
"${relay_bin}" --mode=relay --edges=0:1 --size=64K --warmup=1 \
  --iterations=6000 --dutyCycle=0.2 --reportMs=100 \
  --telemetryFile="${telemetry_jsonl}" --output="${telemetry_result}" \
  >"${tmp_dir}/telemetry.log" 2>&1 || {
  cat "${tmp_dir}/telemetry.log" >&2
  fail "telemetry run failed"
}

python3 - "${telemetry_jsonl}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]

assert records, records
timestamps = [record["timestampNs"] for record in records]
assert timestamps == sorted(timestamps), timestamps
assert any(record["event"] == "measurement_start" for record in records), records
assert any(record["event"] == "measurement_end" for record in records), records
snapshots = [record for record in records if record["event"] == "snapshot"]
assert snapshots, records
edge = snapshots[-1]["perEdge"][0]
assert edge["operations"] > 0, edge
assert edge["bytesCompleted"] > 0, edge
assert edge["latest"]["d2hMs"] >= 0, edge
assert edge["latest"]["h2dMs"] >= 0, edge
assert edge["latest"]["endToEndMs"] > 0, edge
PY

echo "PASS: host relay CLI and one-edge smoke"
