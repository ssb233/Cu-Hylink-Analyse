#!/usr/bin/env python3
"""Analyze Stage H temporal pressure and selected copy traces."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "pressureLevel",
    "backgroundPath",
    "victimMode",
    "topology",
    "repetition",
    "deviceList",
    "targetGBps",
    "dutyCycle",
    "bandwidthClass",
    "victimAggregateGBps",
    "backgroundAggregateGBps",
    "status",
    "backgroundJson",
    "victimJson",
}

COPY_KIND_NAMES = {
    1: "h2d",
    2: "d2h",
    8: "dtod",
    10: "p2p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage H temporal ablation CSV/JSON results."
    )
    parser.add_argument("--summary", required=True, help="temporal runner summary CSV")
    parser.add_argument("--output", default="-", help="JSON output path, or -")
    return parser.parse_args()


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"invalid deviceList: {value!r}") from error
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("Stage H requires exactly three unique devices")
    return devices


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


def copy_kind_name(value: Any) -> str:
    if isinstance(value, int):
        return COPY_KIND_NAMES.get(value, f"copykind-{value}")
    text = str(value).strip().lower()
    if text.isdigit():
        number = int(text)
        return COPY_KIND_NAMES.get(number, f"copykind-{number}")
    return text


def read_trace(path_value: str | None) -> tuple[list[dict[str, Any]], int]:
    path = optional_path(path_value)
    if path is None:
        return [], 0
    if not path.is_file():
        raise ValueError(f"trace does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                row["deviceId"] = int(row["deviceId"])
                row["pid"] = int(row["pid"])
                row["contextId"] = int(row["contextId"])
                row["startNs"] = int(row["startNs"])
                row["endNs"] = int(row["endNs"])
                row["durationMs"] = finite_float(row["durationMs"], "trace.durationMs")
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid trace row in {path}: {row}") from error
            rows.append(row)
    meta_path = Path(f"{path}.meta.json")
    dropped = 0
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as stream:
            meta = json.load(stream)
        dropped = int(meta.get("droppedRecords", 0))
    return rows, dropped


def stats(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a timing-stat object")
    required = ("count", "p50", "p90", "p99", "max")
    if any(key not in value for key in required):
        raise ValueError(f"{field} is missing timing-stat fields")
    result: dict[str, Any] = {"count": int(value["count"])}
    for key in required[1:]:
        result[key] = finite_float(value[key], f"{field}.{key}")
    return result


def validate_background(data: dict[str, Any], devices: list[int]) -> dict[str, Any]:
    if data.get("devices") != devices:
        raise ValueError("background devices do not match summary deviceList")
    per_device_fields = {
        "perDeviceBytes": True,
        "perDeviceOperations": True,
        "perDeviceWallActiveSec": True,
        "perDeviceWallActiveDuty": True,
        "perDeviceGpuActivitySec": False,
        "perDeviceGpuActivityDuty": False,
        "perDeviceOperationDurationMs": True,
        "perDeviceSubmitIntervalMs": True,
        "perDeviceIdleGapMs": True,
    }
    for field, numeric in per_device_fields.items():
        values = data.get(field)
        if not isinstance(values, list) or len(values) != len(devices):
            raise ValueError(f"background {field} must align with devices")
        if numeric and field.endswith("Sec") or numeric and field.endswith("Duty"):
            for value in values:
                finite_float(value, f"background.{field}")
        if field in {"perDeviceBytes", "perDeviceOperations"}:
            if any(int(value) < 0 for value in values):
                raise ValueError(f"background {field} cannot be negative")
        if field.endswith("Ms"):
            for index, value in enumerate(values):
                stats(value, f"background.{field}[{index}]")
    total = data.get("totalBytes")
    if not isinstance(total, int) or sum(int(value) for value in data["perDeviceBytes"]) != total:
        raise ValueError("background totalBytes does not equal perDeviceBytes sum")
    return data


def validate_victim(data: dict[str, Any], devices: list[int]) -> dict[str, Any]:
    if data.get("devices") != devices:
        raise ValueError("victim devices do not match summary deviceList")
    aggregate = finite_float(data.get("aggregateGBps"), "victim.aggregateGBps")
    source_results = data.get("sourceResults")
    if not isinstance(source_results, list) or len(source_results) != len(devices):
        raise ValueError("victim sourceResults do not align with devices")
    for device, result in zip(devices, source_results):
        if result.get("device") != device:
            raise ValueError("victim sourceResults are not in device order")
        finite_float(result.get("GBps"), "victim.sourceResults.GBps")
    return {"aggregateGBps": aggregate, "sourceResults": source_results}


def trace_overlap(
    victim_rows: list[dict[str, Any]], background_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    p2p = [row for row in victim_rows if copy_kind_name(row.get("copyKind")) == "p2p"]
    if not p2p:
        p2p = [
            row
            for row in victim_rows
            if row.get("activityKind", "").upper().endswith("PTO P4")
        ]
    if not p2p:
        return {
            "p2pActivityCount": None,
            "p2pSlowCount": None,
            "backgroundActivityCount": None,
            "p2pBackgroundOverlapNs": None,
            "p2pActivityNs": None,
            "p2pSpanNs": None,
            "p2pBackgroundOverlapRatio": None,
        }
    background = [
        row
        for row in background_rows
        if copy_kind_name(row.get("copyKind")) in {"d2h", "h2d", "dtod"}
    ]
    background_by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in background:
        start = row["startNs"]
        end = row["endNs"]
        if end > start:
            background_by_device[row["deviceId"]].append((start, end))
    merged_background: dict[int, list[tuple[int, int]]] = {}
    background_starts: dict[int, list[int]] = {}
    for device, intervals in background_by_device.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        merged_background[device] = merged
        background_starts[device] = [start for start, _ in merged]
    slow = sum(1 for row in p2p if row["durationMs"] > 8.0)
    overlap = 0
    p2p_span = 0
    p2p_activity = 0
    for row in p2p:
        p2p_span += max(0, row["endNs"] - row["startNs"])
        p2p_activity += max(0, row["endNs"] - row["startNs"])
        intervals = merged_background.get(row["deviceId"], [])
        starts = background_starts.get(row["deviceId"], [])
        interval_index = max(0, bisect.bisect_right(starts, row["startNs"]) - 1)
        while interval_index < len(intervals):
            other_start, other_end = intervals[interval_index]
            if other_start >= row["endNs"]:
                break
            overlap += max(
                0,
                min(row["endNs"], other_end)
                - max(row["startNs"], other_start),
            )
            interval_index += 1
    return {
        "p2pActivityCount": len(p2p),
        "p2pSlowCount": slow,
        "backgroundActivityCount": len(background),
        "p2pBackgroundOverlapNs": overlap,
        "p2pActivityNs": p2p_activity,
        "p2pSpanNs": p2p_span,
        "p2pBackgroundOverlapRatio": (
            overlap / p2p_activity if p2p_activity > 0 else None
        ),
    }


def classify(row: dict[str, Any], background: dict[str, Any] | None) -> str:
    if row["backgroundPath"] == "none":
        return "control"
    if row["backgroundPath"] == "original-d2h":
        return "original-d2h"
    if row["pressureLevel"] == "saturated":
        return "saturated"
    provided = row.get("bandwidthClass", "")
    if provided in {"bandwidth-matched", "duty-matched", "saturated"}:
        return provided
    if finite_float(row["targetGBps"], "targetGBps") == 0.0:
        return "saturated"
    if background is None:
        return provided or "unclassified"
    return "bandwidth-matched"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    rows_total = 0
    ignored = 0
    devices: list[int] | None = None
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
            row_devices = parse_devices(row["deviceList"])
            if devices is None:
                devices = row_devices
            elif devices != row_devices:
                raise ValueError("summary rows use different deviceList values")
            victim_data = load_json(row["victimJson"], "victimJson")
            victim = validate_victim(victim_data, row_devices)
            background_data = None
            if row["backgroundPath"] != "none":
                background_data = validate_background(
                    load_json(row["backgroundJson"], "backgroundJson"), row_devices
                )
            victim_trace, victim_dropped = read_trace(row.get("traceVictim"))
            background_trace, background_dropped = read_trace(row.get("traceBackground"))
            trace_metrics = trace_overlap(victim_trace, background_trace)
            background_rate = (
                finite_float(row["backgroundAggregateGBps"], "backgroundAggregateGBps")
                if row["backgroundPath"] != "none"
                else None
            )
            case = {
                "pressureLevel": row["pressureLevel"],
                "backgroundPath": row["backgroundPath"],
                "victimMode": row["victimMode"],
                "topology": row["topology"],
                "repetition": int(row["repetition"]),
                "deviceList": row_devices,
                "targetGBps": finite_float(row["targetGBps"], "targetGBps"),
                "dutyCycle": finite_float(row["dutyCycle"], "dutyCycle"),
                "bandwidthClass": row["bandwidthClass"],
                "measuredClass": classify(row, background_data),
                "victimAggregateGBps": victim["aggregateGBps"],
                "backgroundAggregateGBps": background_rate,
                "backgroundPerDeviceGBps": (
                    [
                        finite_float(value, "background.perDeviceGBps")
                        for value in background_data.get("perDeviceGBps", [])
                    ]
                    if background_data is not None
                    else []
                ),
                "backgroundWallActiveSec": (
                    background_data["perDeviceWallActiveSec"]
                    if background_data is not None
                    else []
                ),
                "backgroundWallActiveDuty": (
                    background_data["perDeviceWallActiveDuty"]
                    if background_data is not None
                    else []
                ),
                "backgroundGpuActivitySec": (
                    background_data["perDeviceGpuActivitySec"]
                    if background_data is not None
                    else []
                ),
                "backgroundGpuActivityDuty": (
                    background_data["perDeviceGpuActivityDuty"]
                    if background_data is not None
                    else []
                ),
                "backgroundOperationDurationMs": (
                    background_data["perDeviceOperationDurationMs"]
                    if background_data is not None
                    else []
                ),
                "backgroundSubmitIntervalMs": (
                    background_data["perDeviceSubmitIntervalMs"]
                    if background_data is not None
                    else []
                ),
                "backgroundIdleGapMs": (
                    background_data["perDeviceIdleGapMs"]
                    if background_data is not None
                    else []
                ),
                "p2pActivityCount": trace_metrics["p2pActivityCount"],
                "p2pSlowCount": trace_metrics["p2pSlowCount"],
                "backgroundActivityCount": trace_metrics["backgroundActivityCount"],
                "p2pBackgroundOverlapNs": trace_metrics["p2pBackgroundOverlapNs"],
                "p2pActivityNs": trace_metrics["p2pActivityNs"],
                "p2pSpanNs": trace_metrics["p2pSpanNs"],
                "p2pBackgroundOverlapRatio": trace_metrics["p2pBackgroundOverlapRatio"],
                "droppedRecords": victim_dropped + background_dropped,
                "victimJson": row["victimJson"],
                "backgroundJson": row["backgroundJson"],
                "traceVictim": row.get("traceVictim", "NA"),
                "traceBackground": row.get("traceBackground", "NA"),
            }
            cases.append(case)
            key = (
                case["pressureLevel"],
                case["backgroundPath"],
                case["victimMode"],
                case["topology"],
            )
            groups[key].append(case)

    if devices is None:
        raise ValueError("summary has no passing rows")

    clean_by_victim_topology: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in cases:
        if case["backgroundPath"] == "none":
            clean_by_victim_topology[(case["victimMode"], case["topology"])].append(
                case["victimAggregateGBps"]
            )

    group_output = []
    for key in sorted(groups):
        pressure, background, victim, topology = key
        values = groups[key]
        victim_values = [case["victimAggregateGBps"] for case in values]
        background_values = [
            case["backgroundAggregateGBps"]
            for case in values
            if case["backgroundAggregateGBps"] is not None
        ]
        clean_values = clean_by_victim_topology.get((victim, topology), [])
        treatment_mean = mean(victim_values)
        clean_mean = mean(clean_values)
        drop = (
            (clean_mean - treatment_mean) / clean_mean * 100.0
            if clean_mean is not None and treatment_mean is not None and clean_mean != 0
            else None
        )
        group_output.append(
            {
                "pressureLevel": pressure,
                "backgroundPath": background,
                "victimMode": victim,
                "topology": topology,
                "caseCount": len(values),
                "measuredClasses": sorted({case["measuredClass"] for case in values}),
                "victimAggregateGBps": {
                    "mean": treatment_mean,
                    "min": min(victim_values),
                    "max": max(victim_values),
                },
                "backgroundAggregateGBps": {
                    "mean": mean(background_values),
                    "min": min(background_values) if background_values else None,
                    "max": max(background_values) if background_values else None,
                },
                "cleanVictimMeanGBps": clean_mean,
        "cleanToTreatmentDropPct": drop,
                "meanWallActiveDuty": mean(
                    [
                        finite_float(value, "background.perDeviceWallActiveDuty")
                        for case in values
                        for value in case["backgroundWallActiveDuty"]
                    ]
                ),
                "p2pSlowCount": (
                    sum(
                        case["p2pSlowCount"]
                        for case in values
                        if case["p2pSlowCount"] is not None
                    )
                    if any(case["p2pSlowCount"] is not None for case in values)
                    else None
                ),
                "p2pBackgroundOverlapNs": (
                    sum(
                        case["p2pBackgroundOverlapNs"]
                        for case in values
                        if case["p2pBackgroundOverlapNs"] is not None
                    )
                    if any(
                        case["p2pBackgroundOverlapNs"] is not None for case in values
                    )
                    else None
                ),
                "droppedRecords": sum(case["droppedRecords"] for case in values),
            }
        )

    return {
        "deviceList": devices,
        "totalRows": rows_total,
        "ignoredRows": ignored,
        "cases": cases,
        "groups": group_output,
        "threeGpuAdaptation": {
            "singleTwoCopyDescription": (
                "one source stream carries two consecutive allpairs P2P copies"
            ),
            "edgeActiveCopiesPerSource": 2,
            "fourGpuOneVsTwoAssignmentComparable": False,
        },
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
