#!/usr/bin/env python3
"""Run P0 NCCL/host-relay cases with an exact victim measured window.

The relay emits cumulative JSONL counters on the same monotonic clock used by
the instrumented nccl-tests binary.  This runner uses the victim's measured
begin/end markers, rather than the whole relay process lifetime, to estimate
background bytes in the overlap window.  The two surrounding telemetry
samples are retained so the boundary uncertainty is visible in the result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FLOAT = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
MEASUREMENT_RE = re.compile(
    rf"(?m)^[ \t]*[0-9]+\s+[0-9]+\s+\S+\s+\S+\s+-?[0-9]+\s+"
    rf"({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+([0-9]+)"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    edges: str


SCENARIOS: Dict[str, Scenario] = {
    "d2h": Scenario("d2h", "d2h", "0:0"),
    "h2d": Scenario("h2d", "h2d", "1:1"),
    "single": Scenario("single", "relay", "0:1"),
    "disjoint": Scenario("disjoint", "relay", "0:1,2:3"),
    "ring": Scenario("ring", "relay", "0:1,1:2,2:3,3:0"),
}

COLLECTIVE_BINARIES = {
    "allgather": "all_gather_perf",
    "allreduce": "all_reduce_perf",
    "reducescatter": "reduce_scatter_perf",
}


def monotonic_ns() -> int:
    return time.monotonic_ns()


def parse_csv(value: str) -> List[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("comma-separated value cannot be empty")
    return result


def parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def parse_nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def marker_window(path: Path) -> Dict[str, Any]:
    records = load_jsonl(path)
    begins = [
        int(record["timestampNs"])
        for record in records
        if record.get("event") == "measured_begin"
    ]
    ends = [
        int(record["timestampNs"])
        for record in records
        if record.get("event") == "measured_end"
    ]
    if not begins or not ends:
        raise RuntimeError(f"missing measured-window markers in {path}")
    begin_ns = min(begins)
    end_ns = max(ends)
    if end_ns <= begin_ns:
        raise RuntimeError(f"invalid measured window in {path}: {begin_ns}, {end_ns}")
    return {
        "beginNs": begin_ns,
        "endNs": end_ns,
        "durationSec": (end_ns - begin_ns) / 1.0e9,
        "markerCountBegin": len(begins),
        "markerCountEnd": len(ends),
    }


def _counter_value(snapshot: Dict[str, Any], edge_index: int) -> int:
    edges = snapshot.get("perEdge")
    if not isinstance(edges, list) or edge_index >= len(edges):
        raise RuntimeError("telemetry snapshot has no requested edge")
    value = edges[edge_index].get("bytesCompleted")
    if not isinstance(value, int):
        raise RuntimeError("telemetry snapshot has a non-integer byte counter")
    return value


def counter_at(
    samples: Sequence[Tuple[int, int]], timestamp_ns: int
) -> Dict[str, Any]:
    """Interpolate a cumulative counter and retain a conservative boundary range."""

    if not samples:
        raise ValueError("counter samples cannot be empty")
    ordered = sorted(samples)
    if timestamp_ns <= ordered[0][0]:
        timestamp = ordered[0][0]
        return {
            "estimate": ordered[0][1],
            "lower": ordered[0][1],
            "upper": ordered[0][1],
            "leftTimestampNs": timestamp,
            "rightTimestampNs": timestamp,
        }
    if timestamp_ns >= ordered[-1][0]:
        timestamp = ordered[-1][0]
        return {
            "estimate": ordered[-1][1],
            "lower": ordered[-1][1],
            "upper": ordered[-1][1],
            "leftTimestampNs": timestamp,
            "rightTimestampNs": timestamp,
        }

    for (left_ns, left_value), (right_ns, right_value) in zip(
        ordered, ordered[1:]
    ):
        if timestamp_ns == left_ns:
            return {
                "estimate": left_value,
                "lower": left_value,
                "upper": left_value,
                "leftTimestampNs": left_ns,
                "rightTimestampNs": left_ns,
            }
        if timestamp_ns == right_ns:
            return {
                "estimate": right_value,
                "lower": right_value,
                "upper": right_value,
                "leftTimestampNs": right_ns,
                "rightTimestampNs": right_ns,
            }
        if left_ns < timestamp_ns < right_ns:
            if right_ns == left_ns:
                estimate = float(right_value)
            else:
                fraction = (timestamp_ns - left_ns) / (right_ns - left_ns)
                estimate = left_value + fraction * (right_value - left_value)
            return {
                "estimate": estimate,
                "lower": min(left_value, right_value),
                "upper": max(left_value, right_value),
                "leftTimestampNs": left_ns,
                "rightTimestampNs": right_ns,
            }
    raise AssertionError("timestamp was not bracketed by sorted samples")


def overlap_from_snapshots(
    snapshots: Sequence[Dict[str, Any]],
    begin_ns: int,
    end_ns: int,
    edge_index: int = 0,
) -> Dict[str, Any]:
    if end_ns <= begin_ns:
        raise ValueError("overlap end must be greater than begin")
    ordered = sorted(
        (record for record in snapshots if record.get("event") == "snapshot"),
        key=lambda record: int(record["timestampNs"]),
    )
    samples = [
        (int(record["timestampNs"]), _counter_value(record, edge_index))
        for record in ordered
    ]
    if not samples:
        raise ValueError("telemetry has no snapshots")

    start = counter_at(samples, begin_ns)
    end = counter_at(samples, end_ns)
    estimate_bytes = max(0.0, float(end["estimate"]) - float(start["estimate"]))
    lower_bytes = max(0, int(end["lower"]) - int(start["upper"]))
    upper_bytes = max(0, int(end["upper"]) - int(start["lower"]))
    return {
        "beginNs": begin_ns,
        "endNs": end_ns,
        "durationSec": (end_ns - begin_ns) / 1.0e9,
        "bytesEstimate": estimate_bytes,
        "bytesLowerBound": lower_bytes,
        "bytesUpperBound": upper_bytes,
        "boundaryUncertaintyBytes": max(
            estimate_bytes - lower_bytes, upper_bytes - estimate_bytes
        ),
        "startCounter": start,
        "endCounter": end,
        "sampleCount": len(samples),
        "coveredByTelemetry": samples[0][0] <= begin_ns <= samples[-1][0]
        and samples[0][0] <= end_ns <= samples[-1][0],
    }


def aggregate_overlap(
    snapshots: Sequence[Dict[str, Any]], begin_ns: int, end_ns: int
) -> Dict[str, Any]:
    edge_count = 0
    for record in snapshots:
        if record.get("event") == "snapshot":
            per_edge = record.get("perEdge")
            if isinstance(per_edge, list):
                edge_count = max(edge_count, len(per_edge))
    if edge_count == 0:
        raise ValueError("telemetry has no per-edge snapshots")
    per_edge = [
        overlap_from_snapshots(snapshots, begin_ns, end_ns, edge_index)
        for edge_index in range(edge_count)
    ]
    result = {
        "beginNs": begin_ns,
        "endNs": end_ns,
        "durationSec": (end_ns - begin_ns) / 1.0e9,
        "bytesEstimate": sum(item["bytesEstimate"] for item in per_edge),
        "bytesLowerBound": sum(item["bytesLowerBound"] for item in per_edge),
        "bytesUpperBound": sum(item["bytesUpperBound"] for item in per_edge),
        "perEdge": per_edge,
    }
    result["boundaryUncertaintyBytes"] = max(
        result["bytesEstimate"] - result["bytesLowerBound"],
        result["bytesUpperBound"] - result["bytesEstimate"],
    )
    result["coveredByTelemetry"] = all(
        item["coveredByTelemetry"] for item in per_edge
    )
    duration = result["durationSec"]
    if duration > 0:
        result["usefulGBps"] = result["bytesEstimate"] / duration / 1.0e9
        result["usefulGBpsLowerBound"] = (
            result["bytesLowerBound"] / duration / 1.0e9
        )
        result["usefulGBpsUpperBound"] = (
            result["bytesUpperBound"] / duration / 1.0e9
        )
    else:
        result["usefulGBps"] = 0.0
        result["usefulGBpsLowerBound"] = 0.0
        result["usefulGBpsUpperBound"] = 0.0
    return result


def relay_rate_slowdown(
    concurrent_gbps: Optional[float], standalone_gbps: Optional[float]
) -> Optional[float]:
    """Return CE slowdown (%) relative to a same-configuration standalone rate."""

    if concurrent_gbps is None or standalone_gbps is None:
        return None
    if not math.isfinite(concurrent_gbps) or not math.isfinite(standalone_gbps):
        return None
    if standalone_gbps <= 0.0:
        return None
    return (1.0 - concurrent_gbps / standalone_gbps) * 100.0


def parse_measurement(log_path: Path) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(MEASUREMENT_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"no NCCL measurement row in {log_path}")
    match = matches[-1]
    time_us = float(match.group(1))
    algbw = float(match.group(2))
    busbw = float(match.group(3))
    return {
        "timeUs": time_us,
        "algbwGBps": algbw if math.isfinite(algbw) else None,
        "busbwGBps": busbw if math.isfinite(busbw) else None,
        "wrong": int(match.group(4)),
        "measurementRows": len(matches),
    }


def numeric_or_text(value: str) -> Any:
    value = value.strip()
    if value in {"N/A", "[Not Supported]", "-"}:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def gpu_snapshot() -> Dict[str, Any]:
    query = (
        "index,clocks.current.graphics,clocks.current.sm,"
        "clocks.current.memory,pstate,temperature.gpu,power.draw,power.limit,"
        "utilization.gpu,clocks_throttle_reasons.active"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        return {
            "status": "error",
            "returnCode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "command": command,
        }
    fields = [
        "index",
        "graphicsMHz",
        "smMHz",
        "memoryMHz",
        "pstate",
        "temperatureC",
        "powerDrawW",
        "powerLimitW",
        "utilizationPct",
        "throttleReason",
    ]
    rows: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append({key: numeric_or_text(value) for key, value in zip(fields, values)})
    return {"status": "ok", "command": command, "gpus": rows}


def command_environment(
    devices: str, nccl_build: Path, marker_file: Path
) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "NCCL_ALGO": "Ring",
            "NCCL_PROTO": "Simple",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_TESTS_WINDOW_FILE": str(marker_file),
            "LD_LIBRARY_PATH": f"{nccl_build / 'lib'}:/usr/local/cuda/lib64",
        }
    )
    return environment


def victim_command(
    tests_build: Path,
    collective: str,
    size: str,
    devices: str,
    warmup: int,
    iterations: int,
    victim_cpus: str,
) -> List[str]:
    binary = COLLECTIVE_BINARIES[collective]
    return [
        "taskset",
        "-c",
        victim_cpus,
        "numactl",
        "--membind=0",
        str(tests_build / binary),
        "-b",
        size,
        "-e",
        size,
        "-f",
        "2",
        "-g",
        str(len(parse_csv(devices))),
        "-n",
        str(iterations),
        "-w",
        str(warmup),
    ]


def relay_command(
    relay_bin: Path,
    scenario: Scenario,
    relay_size: str,
    relay_cpus: str,
    warmup: int,
    report_ms: int,
    ready_file: Path,
    start_file: Path,
    stop_file: Path,
    telemetry_file: Path,
    output_file: Path,
    relay_duty_cycle: float = 1.0,
    relay_mode: Optional[str] = None,
    relay_edges: Optional[str] = None,
) -> List[str]:
    mode = scenario.mode if relay_mode is None else relay_mode
    edges = scenario.edges if relay_edges is None else relay_edges
    return [
        "taskset",
        "-c",
        relay_cpus,
        "numactl",
        "--membind=0",
        str(relay_bin),
        f"--mode={mode}",
        f"--edges={edges}",
        f"--size={relay_size}",
        f"--warmup={warmup}",
        "--iterations=0",
        f"--dutyCycle={relay_duty_cycle:g}",
        f"--reportMs={report_ms}",
        f"--readyFile={ready_file}",
        f"--startFile={start_file}",
        f"--stopFile={stop_file}",
        f"--telemetryFile={telemetry_file}",
        f"--output={output_file}",
    ]


def wait_for_file(path: Path, process: subprocess.Popen[Any], timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"process exited before creating {path}: rc={process.returncode}")
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


def wait_process(process: subprocess.Popen[Any], timeout_sec: float) -> int:
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10)


def stop_relay(
    process: Optional[subprocess.Popen[Any]], stop_file: Path, timeout_sec: float
) -> Optional[int]:
    if process is None:
        return None
    stop_file.touch(exist_ok=True)
    try:
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait(timeout=10)


def run_relay_standalone(
    *,
    case_dir: Path,
    relay_bin: Path,
    relay_size: str,
    relay_warmup: int,
    relay_report_ms: int,
    relay_cpus: str,
    devices: str,
    settle_ms: int,
    timeout_sec: float,
    duration_sec: float,
    scenario: Scenario,
    relay_duty_cycle: float = 1.0,
    relay_mode: Optional[str] = None,
    relay_edges: Optional[str] = None,
) -> Dict[str, Any]:
    ready_file = case_dir / "relay-standalone.ready"
    start_file = case_dir / "relay-standalone.start"
    stop_file = case_dir / "relay-standalone.stop"
    telemetry_file = case_dir / "relay-standalone.telemetry.jsonl"
    relay_output = case_dir / "relay-standalone.json"
    relay_log = case_dir / "relay-standalone.log"
    relay_cmd = relay_command(
        relay_bin,
        scenario,
        relay_size,
        relay_cpus,
        relay_warmup,
        relay_report_ms,
        ready_file,
        start_file,
        stop_file,
        telemetry_file,
        relay_output,
        relay_duty_cycle,
        relay_mode,
        relay_edges,
    )
    relay_env = os.environ.copy()
    relay_env["CUDA_VISIBLE_DEVICES"] = devices
    before = gpu_snapshot()
    relay_process: Optional[subprocess.Popen[Any]] = None
    started_ns = monotonic_ns()
    with relay_log.open("w", encoding="utf-8") as relay_stream:
        try:
            relay_process = subprocess.Popen(
                relay_cmd,
                stdout=relay_stream,
                stderr=subprocess.STDOUT,
                env=relay_env,
            )
            wait_for_file(ready_file, relay_process, timeout_sec)
            if settle_ms:
                time.sleep(settle_ms / 1000.0)
            start_file.touch()
            time.sleep(duration_sec)
        finally:
            return_code = stop_relay(relay_process, stop_file, timeout_sec)
    ended_ns = monotonic_ns()
    after = gpu_snapshot()
    actual_mode = scenario.mode if relay_mode is None else relay_mode
    actual_edges = scenario.edges if relay_edges is None else relay_edges
    result: Dict[str, Any] = {
        "label": "relay-standalone",
        "scenario": scenario.name,
        "mode": actual_mode,
        "edges": actual_edges,
        "command": relay_cmd,
        "returnCode": return_code,
        "launchBeginNs": started_ns,
        "launchEndNs": ended_ns,
        "log": str(relay_log),
        "telemetryFile": str(telemetry_file),
        "output": str(relay_output),
        "durationRequestedSec": duration_sec,
        "gpuBefore": before,
        "gpuAfter": after,
    }
    if relay_output.is_file():
        try:
            result["relay"] = json.loads(
                relay_output.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            result["jsonError"] = str(error)
    return result


def run_victim(
    *,
    case_dir: Path,
    tests_build: Path,
    nccl_build: Path,
    collective: str,
    size: str,
    devices: str,
    warmup: int,
    iterations: int,
    victim_cpus: str,
    label: str,
) -> Dict[str, Any]:
    marker_file = case_dir / f"{label}.window.jsonl"
    log_file = case_dir / f"{label}.log"
    command = victim_command(
        tests_build, collective, size, devices, warmup, iterations, victim_cpus
    )
    environment = command_environment(devices, nccl_build, marker_file)
    before = gpu_snapshot()
    started_ns = monotonic_ns()
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        return_code = wait_process(process, timeout_sec=max(120.0, iterations / 20.0))
    ended_ns = monotonic_ns()
    after = gpu_snapshot()
    result: Dict[str, Any] = {
        "label": label,
        "command": command,
        "returnCode": return_code,
        "launchBeginNs": started_ns,
        "launchEndNs": ended_ns,
        "log": str(log_file),
        "windowFile": str(marker_file),
        "gpuBefore": before,
        "gpuAfter": after,
    }
    try:
        result["window"] = marker_window(marker_file)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        result["windowError"] = str(error)
    try:
        result["measurement"] = parse_measurement(log_file)
    except (OSError, RuntimeError, ValueError) as error:
        result["measurementError"] = str(error)
    return result


def run_treatment(
    *,
    case_dir: Path,
    relay_bin: Path,
    tests_build: Path,
    nccl_build: Path,
    scenario: Scenario,
    collective: str,
    victim_size: str,
    relay_size: str,
    devices: str,
    victim_warmup: int,
    victim_iterations: int,
    relay_warmup: int,
    relay_report_ms: int,
    relay_cpus: str,
    victim_cpus: str,
    settle_ms: int,
    timeout_sec: float,
    relay_duty_cycle: float = 1.0,
    relay_mode: Optional[str] = None,
    relay_edges: Optional[str] = None,
) -> Dict[str, Any]:
    ready_file = case_dir / "relay.ready"
    start_file = case_dir / "relay.start"
    stop_file = case_dir / "relay.stop"
    telemetry_file = case_dir / "relay.telemetry.jsonl"
    relay_output = case_dir / "relay.json"
    relay_log = case_dir / "relay.log"
    marker_file = case_dir / "concurrent.window.jsonl"
    victim_log = case_dir / "concurrent.log"
    relay_cmd = relay_command(
        relay_bin,
        scenario,
        relay_size,
        relay_cpus,
        relay_warmup,
        relay_report_ms,
        ready_file,
        start_file,
        stop_file,
        telemetry_file,
        relay_output,
        relay_duty_cycle,
        relay_mode,
        relay_edges,
    )
    victim_cmd = victim_command(
        tests_build,
        collective,
        victim_size,
        devices,
        victim_warmup,
        victim_iterations,
        victim_cpus,
    )
    victim_env = command_environment(devices, nccl_build, marker_file)
    relay_env = os.environ.copy()
    relay_env["CUDA_VISIBLE_DEVICES"] = devices
    relay_env["LD_LIBRARY_PATH"] = f"{nccl_build / 'lib'}:/usr/local/cuda/lib64"
    gpu_before = gpu_snapshot()
    relay_process: Optional[subprocess.Popen[Any]] = None
    victim_return_code: Optional[int] = None
    victim_started_ns = 0
    victim_ended_ns = 0
    gpu_after_victim: Dict[str, Any] = {}
    relay_return_code: Optional[int] = None
    with relay_log.open("w", encoding="utf-8") as relay_stream:
        try:
            relay_process = subprocess.Popen(
                relay_cmd,
                stdout=relay_stream,
                stderr=subprocess.STDOUT,
                env=relay_env,
            )
            wait_for_file(ready_file, relay_process, timeout_sec)
            if settle_ms:
                time.sleep(settle_ms / 1000.0)
            start_file.touch()
            victim_started_ns = monotonic_ns()
            with victim_log.open("w", encoding="utf-8") as victim_stream:
                victim_process = subprocess.Popen(
                    victim_cmd,
                    stdout=victim_stream,
                    stderr=subprocess.STDOUT,
                    env=victim_env,
                )
                victim_return_code = wait_process(victim_process, timeout_sec)
            victim_ended_ns = monotonic_ns()
            gpu_after_victim = gpu_snapshot()
        finally:
            relay_return_code = stop_relay(relay_process, stop_file, timeout_sec)
    gpu_after_cleanup = gpu_snapshot()
    actual_mode = scenario.mode if relay_mode is None else relay_mode
    actual_edges = scenario.edges if relay_edges is None else relay_edges
    result: Dict[str, Any] = {
        "label": "concurrent",
        "scenario": scenario.name,
        "mode": actual_mode,
        "edges": actual_edges,
        "command": victim_cmd,
        "relayCommand": relay_cmd,
        "victimReturnCode": victim_return_code,
        "relayReturnCode": relay_return_code,
        "launchBeginNs": victim_started_ns,
        "launchEndNs": victim_ended_ns,
        "log": str(victim_log),
        "windowFile": str(marker_file),
        "relayLog": str(relay_log),
        "telemetryFile": str(telemetry_file),
        "relayOutput": str(relay_output),
        "gpuBefore": gpu_before,
        "gpuAfterVictim": gpu_after_victim,
        "gpuAfterCleanup": gpu_after_cleanup,
    }
    try:
        window = marker_window(marker_file)
        result["window"] = window
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        result["windowError"] = str(error)
        window = None
    try:
        result["measurement"] = parse_measurement(victim_log)
    except (OSError, RuntimeError, ValueError) as error:
        result["measurementError"] = str(error)
    try:
        snapshots = load_jsonl(telemetry_file)
        result["telemetryRecordCount"] = len(snapshots)
        if window is not None:
            overlap = aggregate_overlap(
                snapshots, window["beginNs"], window["endNs"]
            )
            factor = 2.0 if actual_mode == "relay" else 1.0
            overlap["trafficGBps"] = overlap["usefulGBps"] * factor
            overlap["trafficGBpsLowerBound"] = (
                overlap["usefulGBpsLowerBound"] * factor
            )
            overlap["trafficGBpsUpperBound"] = (
                overlap["usefulGBpsUpperBound"] * factor
            )
            result["relayOverlap"] = overlap
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        result["telemetryError"] = str(error)
    return result


def self_test() -> None:
    samples = [(0, 0), (100, 100), (200, 200)]
    at_50 = counter_at(samples, 50)
    assert at_50["estimate"] == 50
    assert at_50["lower"] == 0
    assert at_50["upper"] == 100
    aligned = counter_at(samples, 100)
    assert aligned["estimate"] == 100
    assert aligned["lower"] == aligned["upper"] == 100
    snapshots = [
        {"event": "snapshot", "timestampNs": 0, "perEdge": [{"bytesCompleted": 0}]},
        {"event": "snapshot", "timestampNs": 100, "perEdge": [{"bytesCompleted": 100}]},
        {"event": "snapshot", "timestampNs": 200, "perEdge": [{"bytesCompleted": 200}]},
    ]
    overlap = aggregate_overlap(snapshots, 50, 150)
    assert overlap["bytesEstimate"] == 100
    assert overlap["bytesLowerBound"] == 0
    assert overlap["bytesUpperBound"] == 200
    assert overlap["coveredByTelemetry"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repoRoot", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collectives", default="allgather")
    parser.add_argument("--sizes", default="64M")
    parser.add_argument("--scenarios", default="d2h,h2d,single,disjoint,ring")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--victimIterations", type=int, default=10000)
    parser.add_argument("--victimWarmup", type=int, default=100)
    parser.add_argument("--relaySize", default="255M")
    parser.add_argument("--relayWarmup", type=int, default=20)
    parser.add_argument("--relayReportMs", type=int, default=100)
    parser.add_argument("--relayDutyCycle", type=float, default=1.0)
    parser.add_argument("--relayMode", choices=("d2h", "h2d", "relay"))
    parser.add_argument("--relayEdges")
    parser.add_argument(
        "--relayStandaloneSec",
        type=float,
        default=0.0,
        help="measure same-configuration relay-only baseline before treatment; 0 disables",
    )
    parser.add_argument("--settleMs", type=int, default=250)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--relayCpus", default="0,2,4,6")
    parser.add_argument("--victimCpus", default="8,10")
    parser.add_argument("--randomSeed", type=int, default=20260831)
    parser.add_argument("--timeoutSec", type=float, default=180.0)
    parser.add_argument("--ncclBuild", type=Path)
    parser.add_argument("--testsBuild", type=Path)
    parser.add_argument("--relayBin", type=Path)
    parser.add_argument("--outputRoot", type=Path, required=False)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo_root = args.repoRoot.resolve()
    nccl_build = (args.ncclBuild or repo_root / "build/nccl-v2.31.2-sm70-sys").resolve()
    tests_build = (args.testsBuild or repo_root / "build/nccl-tests-p0-sm70").resolve()
    relay_bin = (args.relayBin or repo_root / "build/cuda_copy/host_relay").resolve()
    output_root = args.outputRoot
    if output_root is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_root = repo_root / "doc/results/nccl-hybrid-path/p0-overlap" / stamp
    output_root = output_root.resolve()

    collectives = parse_csv(args.collectives)
    sizes = parse_csv(args.sizes)
    scenario_names = parse_csv(args.scenarios)
    if any(item not in COLLECTIVE_BINARIES for item in collectives):
        raise ValueError(f"unknown collective; choose from {sorted(COLLECTIVE_BINARIES)}")
    if any(item not in SCENARIOS for item in scenario_names):
        raise ValueError(f"unknown scenario; choose from {sorted(SCENARIOS)}")
    if args.repetitions <= 0 or args.victimIterations <= 0 or args.victimWarmup < 0:
        raise ValueError("repetitions/victimIterations must be positive and victimWarmup non-negative")
    if args.relayWarmup < 0 or args.relayReportMs <= 0 or args.settleMs < 0:
        raise ValueError("relayWarmup must be non-negative; relayReportMs must be positive; settleMs non-negative")
    if not math.isfinite(args.relayDutyCycle) or args.relayDutyCycle <= 0.0 or args.relayDutyCycle > 1.0:
        raise ValueError("relayDutyCycle must be in (0,1]")
    if args.relayEdges is not None and not args.relayEdges.strip():
        raise ValueError("relayEdges must not be empty")
    if not math.isfinite(args.relayStandaloneSec) or args.relayStandaloneSec < 0.0:
        raise ValueError("relayStandaloneSec must be finite and non-negative")
    if not relay_bin.is_file():
        raise FileNotFoundError(relay_bin)
    for collective in collectives:
        if not (tests_build / COLLECTIVE_BINARIES[collective]).is_file():
            raise FileNotFoundError(tests_build / COLLECTIVE_BINARIES[collective])
    if not (nccl_build / "lib/libnccl.so.2.31.2").is_file():
        raise FileNotFoundError(nccl_build / "lib/libnccl.so.2.31.2")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.randomSeed)
    manifest: Dict[str, Any] = {
        "schemaVersion": 1,
        "clock": "steady_clock/time.monotonic_ns",
        "createdAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "collectives": collectives,
            "sizes": sizes,
            "scenarios": scenario_names,
            "repetitions": args.repetitions,
            "victimIterations": args.victimIterations,
            "victimWarmup": args.victimWarmup,
            "relaySize": args.relaySize,
            "relayWarmup": args.relayWarmup,
            "relayReportMs": args.relayReportMs,
            "relayDutyCycle": args.relayDutyCycle,
            "relayMode": args.relayMode,
            "relayEdges": args.relayEdges,
            "relayStandaloneSec": args.relayStandaloneSec,
            "settleMs": args.settleMs,
            "devices": args.devices,
            "relayCpus": args.relayCpus,
            "victimCpus": args.victimCpus,
            "randomSeed": args.randomSeed,
            "ncclBuild": str(nccl_build),
            "testsBuild": str(tests_build),
            "relayBin": str(relay_bin),
        },
        "cases": [],
    }

    for collective in collectives:
        for size in sizes:
            for repetition in range(1, args.repetitions + 1):
                order = list(scenario_names)
                rng.shuffle(order)
                for scenario_name in order:
                    scenario = SCENARIOS[scenario_name]
                    case_dir = output_root / collective / size / f"rep-{repetition}" / scenario_name
                    case_dir.mkdir(parents=True, exist_ok=True)
                    case: Dict[str, Any] = {
                        "collective": collective,
                        "size": size,
                        "repetition": repetition,
                        "scenario": scenario_name,
                        "randomizedOrder": order,
                    }
                    print(
                        f"[{collective} {size} rep-{repetition} {scenario_name}] "
                        f"clean-before -> treatment -> clean-after",
                        flush=True,
                    )
                    before = run_victim(
                        case_dir=case_dir,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        collective=collective,
                        size=size,
                        devices=args.devices,
                        warmup=args.victimWarmup,
                        iterations=args.victimIterations,
                        victim_cpus=args.victimCpus,
                        label="clean-before",
                    )
                    relay_standalone = None
                    if args.relayStandaloneSec > 0.0:
                        relay_standalone = run_relay_standalone(
                            case_dir=case_dir,
                            relay_bin=relay_bin,
                            relay_size=args.relaySize,
                            relay_warmup=args.relayWarmup,
                            relay_report_ms=args.relayReportMs,
                            relay_cpus=args.relayCpus,
                            devices=args.devices,
                            settle_ms=args.settleMs,
                            timeout_sec=args.timeoutSec,
                            duration_sec=args.relayStandaloneSec,
                            scenario=scenario,
                            relay_duty_cycle=args.relayDutyCycle,
                            relay_mode=args.relayMode,
                            relay_edges=args.relayEdges,
                        )
                    treatment = run_treatment(
                        case_dir=case_dir,
                        relay_bin=relay_bin,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        scenario=scenario,
                        collective=collective,
                        victim_size=size,
                        relay_size=args.relaySize,
                        devices=args.devices,
                        victim_warmup=args.victimWarmup,
                        victim_iterations=args.victimIterations,
                        relay_warmup=args.relayWarmup,
                        relay_report_ms=args.relayReportMs,
                        relay_cpus=args.relayCpus,
                        victim_cpus=args.victimCpus,
                        settle_ms=args.settleMs,
                        timeout_sec=args.timeoutSec,
                        relay_duty_cycle=args.relayDutyCycle,
                        relay_mode=args.relayMode,
                        relay_edges=args.relayEdges,
                    )
                    after = run_victim(
                        case_dir=case_dir,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        collective=collective,
                        size=size,
                        devices=args.devices,
                        warmup=args.victimWarmup,
                        iterations=args.victimIterations,
                        victim_cpus=args.victimCpus,
                        label="clean-after",
                    )
                    case.update(
                        {
                            "caseDir": str(case_dir),
                            "cleanBefore": before,
                            "relayStandalone": relay_standalone,
                            "treatment": treatment,
                            "cleanAfter": after,
                        }
                    )
                    if relay_standalone is not None:
                        standalone_payload = relay_standalone.get("relay", {})
                        standalone_rate = standalone_payload.get(
                            "aggregateUsefulGBps"
                        )
                        overlap_rate = treatment.get("relayOverlap", {}).get(
                            "usefulGBps"
                        )
                        case["relaySlowdownPct"] = relay_rate_slowdown(
                            overlap_rate, standalone_rate
                        )
                    (case_dir / "case.json").write_text(
                        json.dumps(case, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    manifest["cases"].append(case)
                    (output_root / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
    print(f"P0 overlap results: {output_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        return run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
