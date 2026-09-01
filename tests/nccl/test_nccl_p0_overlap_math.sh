#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
python3 "$repo_root/scripts/run_nccl_p0_overlap.py" --self-test
python3 - "$repo_root" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "scripts"))
from run_nccl_p0_overlap import Scenario, relay_command, relay_rate_slowdown

command = relay_command(
    relay_bin=repo_root / "build/cuda_copy/host_relay",
    scenario=Scenario("ring", "relay", "0:1,1:2,2:3,3:0"),
    relay_size="255M",
    relay_cpus="0,2,4,6",
    warmup=20,
    report_ms=100,
    ready_file=repo_root / "ready",
    start_file=repo_root / "start",
    stop_file=repo_root / "stop",
    telemetry_file=repo_root / "telemetry",
    output_file=repo_root / "output",
    relay_duty_cycle=0.25,
)
assert "--dutyCycle=0.25" in command, command

direction_command = relay_command(
    relay_bin=repo_root / "build/cuda_copy/host_relay",
    scenario=Scenario("d2h4", "relay", "0:1"),
    relay_size="255M",
    relay_cpus="0,2,4,6",
    warmup=20,
    report_ms=100,
    ready_file=repo_root / "ready",
    start_file=repo_root / "start",
    stop_file=repo_root / "stop",
    telemetry_file=repo_root / "telemetry",
    output_file=repo_root / "output",
    relay_duty_cycle=0.5,
    relay_mode="d2h",
    relay_edges="0:0,1:1,2:2,3:3",
)
assert "--mode=d2h" in direction_command, direction_command
assert "--edges=0:0,1:1,2:2,3:3" in direction_command, direction_command
assert abs(relay_rate_slowdown(98.0, 100.0) - 2.0) < 1e-9
assert abs(relay_rate_slowdown(102.0, 100.0) + 2.0) < 1e-9
assert relay_rate_slowdown(None, 100.0) is None
PY
echo "PASS: P0 overlap-window math"
