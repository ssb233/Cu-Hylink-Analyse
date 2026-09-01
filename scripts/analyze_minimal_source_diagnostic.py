#!/usr/bin/env python3
"""Analyze the H3 two-edge source-local diagnostic matrix."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "size",
    "backgroundSet",
    "streamMode",
    "streamDependency",
    "edgeOrder",
    "repetition",
    "deviceList",
    "victimAggregateGBps",
    "status",
    "backgroundJson",
    "victimJson",
}
SIZE_RE = re.compile(r"^(?P<amount>[0-9]+(?:\.[0-9]+)?)(?P<suffix>[KMG]?(?:[iI]?[bB])?)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate H3 minimal source diagnostic CSV/JSON results."
    )
    parser.add_argument("--summary", required=True, help="H3 summary CSV")
    parser.add_argument(
        "--dropThresholdPct",
        type=float,
        default=5.0,
        help="visible-drop threshold (default: 5%%)",
    )
    parser.add_argument("--output", default="-", help="JSON output path, or -")
    args = parser.parse_args()
    if not math.isfinite(args.dropThresholdPct) or args.dropThresholdPct <= 0.0:
        parser.error("--dropThresholdPct must be positive and finite")
    return args


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def size_bytes(value: str) -> int:
    match = SIZE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid size: {value!r}")
    amount = float(match.group("amount"))
    suffix = match.group("suffix").lower()
    multiplier = {"": 1, "k": 1024, "kb": 1024, "m": 1024**2,
                  "mb": 1024**2, "g": 1024**3, "gb": 1024**3}.get(suffix)
    if multiplier is None:
        raise ValueError(f"unsupported size suffix: {value!r}")
    result = amount * multiplier
    if not math.isfinite(result) or result < 1 or result > (1 << 63) - 1:
        raise ValueError(f"size is outside the supported range: {value!r}")
    return int(result)


def load_json(path_value: str, field: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object: {path}")
    return value


def optional_path(value: str | None) -> Path | None:
    if not value or value == "NA":
        return None
    return Path(value)


def trace_metrics(victim_path: Path | None, background_path: Path | None) -> dict[str, Any]:
    if victim_path is None:
        return {
            "p2pCount": None,
            "p2pSlowCount": None,
            "p2pByEdge": None,
            "backgroundCount": None,
            "backgroundP2pOverlapNs": None,
            "droppedRecords": None,
        }
    if not victim_path.is_file():
        raise ValueError(f"traceVictim does not exist: {victim_path}")
    victim_rows: list[dict[str, Any]] = []
    with victim_path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            victim_rows.append(
                {
                    "copyKind": int(raw["copyKind"]),
                    "src": int(raw["srcDeviceId"]),
                    "dst": int(raw["dstDeviceId"]),
                    "device": int(raw["deviceId"]),
                    "start": int(raw["startNs"]),
                    "end": int(raw["endNs"]),
                    "durationMs": finite_float(raw["durationMs"], "trace.durationMs"),
                }
            )
    p2p = [row for row in victim_rows if row["copyKind"] == 10]
    by_edge: dict[str, dict[str, Any]] = {}
    for edge in ("0->1", "0->2"):
        edge_rows = [row for row in p2p if f'{row["src"]}->{row["dst"]}' == edge]
        by_edge[edge] = {
            "count": len(edge_rows),
            "slowCount": sum(row["durationMs"] > 8.0 for row in edge_rows),
        }
    background_rows: list[dict[str, Any]] = []
    if background_path is not None:
        if not background_path.is_file():
            raise ValueError(f"traceBackground does not exist: {background_path}")
        with background_path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                if int(raw["copyKind"]) not in (1, 2, 8):
                    continue
                background_rows.append(
                    {
                        "device": int(raw["deviceId"]),
                        "start": int(raw["startNs"]),
                        "end": int(raw["endNs"]),
                    }
                )
    by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in background_rows:
        if row["end"] > row["start"]:
            by_device[row["device"]].append((row["start"], row["end"]))
    overlap = 0
    for device, intervals in by_device.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        starts = [start for start, _ in merged]
        for row in (item for item in p2p if item["device"] == device):
            index = max(0, bisect.bisect_right(starts, row["start"]) - 1)
            while index < len(merged) and merged[index][0] < row["end"]:
                overlap += max(
                    0,
                    min(row["end"], merged[index][1])
                    - max(row["start"], merged[index][0]),
                )
                index += 1
    meta_path = Path(f"{victim_path}.meta.json")
    dropped = None
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as stream:
            dropped = int(json.load(stream).get("droppedRecords", 0))
    if background_path is not None:
        background_meta = Path(f"{background_path}.meta.json")
        if background_meta.is_file():
            with background_meta.open(encoding="utf-8") as stream:
                dropped = (dropped or 0) + int(
                    json.load(stream).get("droppedRecords", 0)
                )
    return {
        "p2pCount": len(p2p),
        "p2pSlowCount": sum(row["durationMs"] > 8.0 for row in p2p),
        "p2pByEdge": by_edge,
        "backgroundCount": len(background_rows),
        "backgroundP2pOverlapNs": overlap,
        "droppedRecords": dropped,
    }


def timing_stats(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a timing-stat object")
    required = ("count", "p50", "p90", "p99", "max")
    if any(key not in value for key in required):
        raise ValueError(f"{field} is missing timing-stat fields")
    result = {"count": int(value["count"])}
    for key in required[1:]:
        result[key] = finite_float(value[key], f"{field}.{key}")
    return result


def validate_victim(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("devices") != [0, 1, 2]:
        raise ValueError("victim must use devices [0, 1, 2]")
    aggregate = finite_float(data.get("aggregateGBps"), "victim.aggregateGBps")
    edge_results = data.get("edgeResults")
    if not isinstance(edge_results, list) or len(edge_results) != 2:
        raise ValueError("victim must contain two edgeResults")
    edges: dict[str, dict[str, Any]] = {}
    for result in edge_results:
        if not isinstance(result, dict):
            raise ValueError("victim edge result must be an object")
        source = int(result.get("source", -1))
        destination = int(result.get("destination", -1))
        key = f"{source}->{destination}"
        if source != 0 or destination not in (1, 2) or key in edges:
            raise ValueError("victim edge results must be 0->1 and 0->2")
        edges[key] = {
            "GBps": finite_float(result.get("GBps"), f"victim.{key}.GBps"),
            "elapsedMs": finite_float(
                result.get("elapsedMs"), f"victim.{key}.elapsedMs"
            ),
            "operations": int(result.get("operations", 0)),
            "operationDurationMs": timing_stats(
                result.get("operationDurationMs"),
                f"victim.{key}.operationDurationMs",
            ),
            "order": int(result.get("order", -1)),
            "stream": int(result.get("stream", -1)),
        }
    if set(edges) != {"0->1", "0->2"}:
        raise ValueError("victim edge results are incomplete")
    return {"aggregateGBps": aggregate, "edges": edges}


def validate_background(data: dict[str, Any]) -> dict[str, Any]:
    devices = data.get("devices")
    if not isinstance(devices, list) or not devices or any(
        device not in (0, 1, 2) for device in devices
    ):
        raise ValueError("background devices must be a non-empty H3 subset")
    fields = (
        "perDeviceBytes",
        "perDeviceOperations",
        "perDeviceWallActiveSec",
        "perDeviceWallActiveDuty",
        "perDeviceGpuActivitySec",
        "perDeviceGpuActivityDuty",
        "perDeviceOperationDurationMs",
        "perDeviceSubmitIntervalMs",
        "perDeviceIdleGapMs",
    )
    for field in fields:
        values = data.get(field)
        if not isinstance(values, list) or len(values) != len(devices):
            raise ValueError(f"background {field} is not aligned with devices")
    for field in (
        "perDeviceWallActiveSec",
        "perDeviceWallActiveDuty",
    ):
        for value in data[field]:
            finite_float(value, f"background.{field}")
    for field in ("perDeviceBytes", "perDeviceOperations"):
        if any(int(value) < 0 for value in data[field]):
            raise ValueError(f"background {field} cannot be negative")
    for field in (
        "perDeviceOperationDurationMs",
        "perDeviceSubmitIntervalMs",
        "perDeviceIdleGapMs",
    ):
        for index, value in enumerate(data[field]):
            timing_stats(value, f"background.{field}[{index}]")
    total = data.get("totalBytes")
    if not isinstance(total, int) or sum(map(int, data["perDeviceBytes"])) != total:
        raise ValueError("background totalBytes does not match perDeviceBytes")
    return {
        "devices": devices,
        "bytes": data["perDeviceBytes"],
        "operations": data["perDeviceOperations"],
        "bytesPerOperation": [
            (int(byte) / int(operation) if int(operation) else None)
            for byte, operation in zip(data["perDeviceBytes"], data["perDeviceOperations"])
        ],
        "wallActiveSec": data["perDeviceWallActiveSec"],
        "wallActiveDuty": data["perDeviceWallActiveDuty"],
        "gpuActivitySec": data["perDeviceGpuActivitySec"],
        "gpuActivityDuty": data["perDeviceGpuActivityDuty"],
        "operationDurationMs": data["perDeviceOperationDurationMs"],
        "submitIntervalMs": data["perDeviceSubmitIntervalMs"],
        "idleGapMs": data["perDeviceIdleGapMs"],
        "aggregateGBps": finite_float(
            data.get("aggregateGBps", 0.0), "background.aggregateGBps"
        ),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def drop_pct(clean: float | None, treatment: float | None) -> float | None:
    if clean is None or treatment is None or clean == 0.0:
        return None
    return (clean - treatment) / clean * 100.0


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    cases: list[dict[str, Any]] = []
    rows_total = 0
    ignored = 0
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("summary CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError("summary CSV is missing: " + ", ".join(sorted(missing)))
        for row in reader:
            rows_total += 1
            if row["status"] != "pass":
                ignored += 1
                continue
            if row["deviceList"] != "0,1,2":
                raise ValueError("H3 summary must use deviceList=0,1,2")
            victim_data = validate_victim(load_json(row["victimJson"], "victimJson"))
            background_data = None
            background_path = optional_path(row.get("backgroundJson"))
            if row["backgroundSet"] != "none":
                if background_path is None:
                    raise ValueError("non-none background requires backgroundJson")
                background_data = validate_background(
                    load_json(str(background_path), "backgroundJson")
                )
            edge_order = row["edgeOrder"]
            expected_order = "forward" if row["edgeOrder"] == "forward" else "reverse"
            edge_names = list(victim_data["edges"])
            if expected_order == "forward":
                if edge_names != ["0->1", "0->2"]:
                    raise ValueError("forward edge order is not reflected in victim JSON")
            else:
                if edge_names != ["0->2", "0->1"]:
                    raise ValueError("reverse edge order is not reflected in victim JSON")
            trace_victim = optional_path(row.get("traceVictim"))
            trace_background = optional_path(row.get("traceBackground"))
            trace_data = trace_metrics(trace_victim, trace_background)
            cases.append(
                {
                    "size": row["size"],
                    "sizeBytes": size_bytes(row["size"]),
                    "backgroundSet": row["backgroundSet"],
                    "streamMode": row["streamMode"],
                    "streamDependency": row["streamDependency"],
                    "edgeOrder": edge_order,
                    "repetition": int(row["repetition"]),
                    "deviceList": [0, 1, 2],
                    "victimAggregateGBps": victim_data["aggregateGBps"],
                    "edgeGBps": {
                        key: value["GBps"] for key, value in victim_data["edges"].items()
                    },
                    "edgeElapsedMs": {
                        key: value["elapsedMs"]
                        for key, value in victim_data["edges"].items()
                    },
                    "edgeOperationDurationMs": {
                        key: value["operationDurationMs"]
                        for key, value in victim_data["edges"].items()
                    },
                    "backgroundAggregateGBps": (
                        background_data["aggregateGBps"]
                        if background_data is not None
                        else None
                    ),
                    "backgroundDevices": (
                        background_data["devices"] if background_data is not None else []
                    ),
                    "backgroundBytes": (
                        background_data["bytes"] if background_data is not None else []
                    ),
                    "backgroundOperations": (
                        background_data["operations"] if background_data is not None else []
                    ),
                    "backgroundBytesPerOperation": (
                        background_data["bytesPerOperation"]
                        if background_data is not None
                        else []
                    ),
                    "backgroundWallActiveSec": (
                        background_data["wallActiveSec"] if background_data else []
                    ),
                    "backgroundWallActiveDuty": (
                        background_data["wallActiveDuty"] if background_data else []
                    ),
                    "backgroundGpuActivitySec": (
                        background_data["gpuActivitySec"] if background_data else []
                    ),
                    "backgroundGpuActivityDuty": (
                        background_data["gpuActivityDuty"] if background_data else []
                    ),
                    "backgroundOperationDurationMs": (
                        background_data["operationDurationMs"] if background_data else []
                    ),
                    "backgroundSubmitIntervalMs": (
                        background_data["submitIntervalMs"] if background_data else []
                    ),
                    "backgroundIdleGapMs": (
                        background_data["idleGapMs"] if background_data else []
                    ),
                    "traceVictim": str(trace_victim) if trace_victim else "NA",
                    "traceBackground": str(trace_background)
                    if trace_background
                    else "NA",
                    "traceMetrics": trace_data,
                    "victimJson": row["victimJson"],
                    "backgroundJson": row.get("backgroundJson", "NA"),
                }
            )

    if not cases:
        raise ValueError("summary has no passing rows")

    clean: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for case in cases:
        if case["backgroundSet"] == "none":
            clean[
                (
                    case["streamMode"],
                    case["streamDependency"],
                    case["edgeOrder"],
                    case["sizeBytes"],
                )
            ].append(case["victimAggregateGBps"])

    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[
            (
                case["backgroundSet"],
                case["streamMode"],
                case["streamDependency"],
                case["edgeOrder"],
                case["sizeBytes"],
            )
        ].append(case)
    group_output = []
    for key in sorted(groups):
        background_set, mode, dependency, order, size_value = key
        values = groups[key]
        clean_values = clean.get((mode, dependency, order, size_value), [])
        treatment = mean([case["victimAggregateGBps"] for case in values])
        clean_mean = mean(clean_values)
        group_output.append(
            {
                "size": next(case["size"] for case in values),
                "backgroundSet": background_set,
                "streamMode": mode,
                "streamDependency": dependency,
                "edgeOrder": order,
                "caseCount": len(values),
                "cleanVictimMeanGBps": clean_mean,
                "victimAggregateGBps": treatment,
                "cleanToTreatmentDropPct": drop_pct(clean_mean, treatment),
                "edgeGBps": {
                    edge: mean([case["edgeGBps"][edge] for case in values])
                    for edge in ("0->1", "0->2")
                },
            }
        )

    sweep_cases: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["backgroundSet"] != "none":
            sweep_cases[
                (
                    case["backgroundSet"],
                    case["streamMode"],
                    case["streamDependency"],
                    case["edgeOrder"],
                )
            ].append(case)
    sweeps = []
    for key in sorted(sweep_cases):
        background_set, mode, dependency, order = key
        points = []
        for size_value in sorted({case["sizeBytes"] for case in sweep_cases[key]}):
            values = [case for case in sweep_cases[key] if case["sizeBytes"] == size_value]
            treatment = mean([case["victimAggregateGBps"] for case in values])
            clean_mean = mean(clean.get((mode, dependency, order, size_value), []))
            points.append(
                {
                    "size": values[0]["size"],
                    "sizeBytes": size_value,
                    "victimAggregateGBps": treatment,
                    "cleanVictimMeanGBps": clean_mean,
                    "cleanToTreatmentDropPct": drop_pct(clean_mean, treatment),
                    "edgeGBps": {
                        edge: mean([case["edgeGBps"][edge] for case in values])
                        for edge in ("0->1", "0->2")
                    },
                }
            )
        immune = [point for point in points if (point["cleanToTreatmentDropPct"] or 0.0) < args.dropThresholdPct]
        drops = [point for point in points if (point["cleanToTreatmentDropPct"] or 0.0) >= args.dropThresholdPct]
        first_drop = drops[0] if drops else None
        last_immune = immune[-1] if immune else None
        sweeps.append(
            {
                "backgroundSet": background_set,
                "streamMode": mode,
                "streamDependency": dependency,
                "edgeOrder": order,
                "dropThresholdPct": args.dropThresholdPct,
                "points": points,
                "lastImmuneSize": last_immune["size"] if last_immune else None,
                "firstDropSize": first_drop["size"] if first_drop else None,
                "firstDropPct": (
                    first_drop["cleanToTreatmentDropPct"] if first_drop else None
                ),
            }
        )

    return {
        "deviceList": [0, 1, 2],
        "totalRows": rows_total,
        "ignoredRows": ignored,
        "sourceSharing": {"none": False, "0": True, "1": False, "2": False, "all": True},
        "cases": cases,
        "groups": group_output,
        "sweeps": sweeps,
        "definition": (
            "GPU0 is the sole P2P source; background set 0 shares the source GPU, "
            "while sets 1 and 2 do not"
        ),
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        print(encoded, end="")
    else:
        Path(args.output).write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
