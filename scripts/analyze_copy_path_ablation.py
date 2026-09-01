#!/usr/bin/env python3
"""Aggregate Stage G copy-path ablation results without modifying raw data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "backgroundPath",
    "victimMode",
    "topology",
    "repetition",
    "deviceList",
    "victimSize",
    "backgroundSize",
    "targetGBps",
    "dutyCycle",
    "victimAggregateGBps",
    "backgroundAggregateGBps",
    "victimExit",
    "backgroundExit",
    "status",
    "backgroundJson",
    "victimJson",
}

BACKGROUND_ORDER = {
    "none": 0,
    "original-d2h": 1,
    "local-d2d-ce": 2,
    "streaming-hbm-read": 3,
    "streaming-hbm-write": 4,
    "l2-resident-read": 5,
}
VICTIM_ORDER = {
    "original-p2p-ce": 0,
    "local-d2d-ce": 1,
    "peer-read": 2,
    "peer-write": 3,
}
TOPOLOGY_ORDER = {"single-two-copy": 0, "edge-independent": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage G copy-path ablation CSV/JSON results."
    )
    parser.add_argument("--summary", required=True, help="runner summary CSV")
    parser.add_argument("--output", default="-", help="JSON output path, or -")
    return parser.parse_args()


def finite_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"invalid deviceList: {value!r}") from error
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError("Stage G requires exactly three unique devices")
    return devices


def load_json(path_value: str, field: str) -> dict:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object: {path}")
    return value


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def per_device_stats(
    values_by_device: dict[int, list[float]], devices: list[int]
) -> dict[str, dict[str, float | None]]:
    per_device = {
        str(device): stats(values_by_device.get(device, []))
        for device in devices
    }
    return {
        "meanByDevice": {
            device: values["mean"] for device, values in per_device.items()
        },
        "minByDevice": {
            device: values["min"] for device, values in per_device.items()
        },
        "maxByDevice": {
            device: values["max"] for device, values in per_device.items()
        },
    }


def validate_background(
    data: dict, devices: list[int], field: str
) -> tuple[list[float], dict[int, list[float]]]:
    if data.get("devices") != devices:
        raise ValueError(f"{field} devices do not match summary deviceList")
    bytes_values = data.get("perDeviceBytes")
    rate_values = data.get("perDeviceGBps")
    if not isinstance(bytes_values, list) or not isinstance(rate_values, list):
        raise ValueError(f"{field} must contain perDeviceBytes/perDeviceGBps")
    if len(bytes_values) != len(devices) or len(rate_values) != len(devices):
        raise ValueError(f"{field} per-device arrays have the wrong length")
    total = data.get("totalBytes")
    if not isinstance(total, int) or sum(bytes_values) != total:
        raise ValueError(f"{field} totalBytes does not equal perDeviceBytes sum")
    rates = [finite_float(str(value), f"{field}.perDeviceGBps") for value in rate_values]
    by_device = {device: [rate] for device, rate in zip(devices, rates)}
    return rates, by_device


def validate_victim(data: dict, devices: list[int], field: str) -> list[float]:
    if data.get("devices") != devices:
        raise ValueError(f"{field} devices do not match summary deviceList")
    source_results = data.get("sourceResults")
    if not isinstance(source_results, list) or len(source_results) != len(devices):
        raise ValueError(f"{field} sourceResults do not match device count")
    rates = []
    for expected_device, result in zip(devices, source_results):
        if result.get("device") != expected_device:
            raise ValueError(f"{field} sourceResults are not in device order")
        rates.append(finite_float(str(result.get("GBps")), f"{field}.GBps"))
    return rates


def analyze(args: argparse.Namespace) -> dict:
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary does not exist: {summary_path}")

    groups: dict[tuple[str, str, str], dict] = {}
    cases: list[dict] = []
    total_rows = 0
    ignored_rows = 0
    devices: list[int] | None = None
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("summary CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "summary CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            total_rows += 1
            if row["status"] != "pass":
                ignored_rows += 1
                continue
            row_devices = parse_devices(row["deviceList"])
            if devices is None:
                devices = row_devices
            elif devices != row_devices:
                raise ValueError("summary rows use different device lists")

            victim_path = row["victimJson"]
            victim_data = load_json(victim_path, "victimJson")
            victim_rates = validate_victim(victim_data, row_devices, "victimJson")
            victim_gbps = finite_float(row["victimAggregateGBps"], "victimAggregateGBps")

            background_path = row["backgroundPath"]
            background_data = None
            background_rates: list[float] = []
            background_by_device: dict[int, list[float]] = {}
            if background_path != "none":
                background_data = load_json(row["backgroundJson"], "backgroundJson")
                background_rates, background_by_device = validate_background(
                    background_data, row_devices, "backgroundJson"
                )
                background_gbps = finite_float(
                    row["backgroundAggregateGBps"], "backgroundAggregateGBps"
                )
            else:
                if row["backgroundJson"] != "NA":
                    raise ValueError("none case must have backgroundJson=NA")
                background_gbps = None

            key = (background_path, row["victimMode"], row["topology"])
            group = groups.setdefault(
                key,
                {
                    "backgroundPath": background_path,
                    "victimMode": row["victimMode"],
                    "topology": row["topology"],
                    "repetitions": [],
                    "victimAggregateGBps": [],
                    "victimPerSourceGBps": defaultdict(list),
                    "backgroundAggregateGBps": [],
                    "backgroundPerDeviceGBps": defaultdict(list),
                    "targetGBps": [],
                    "dutyCycle": [],
                    "workingSetBytes": [],
                    "victimStreamCounts": [],
                    "victimSlowCounts": [],
                },
            )
            group["repetitions"].append(int(row["repetition"]))
            group["victimAggregateGBps"].append(victim_gbps)
            for device, rate in zip(row_devices, victim_rates):
                group["victimPerSourceGBps"][device].append(rate)
            if background_gbps is not None:
                group["backgroundAggregateGBps"].append(background_gbps)
                for device, rate in zip(row_devices, background_rates):
                    group["backgroundPerDeviceGBps"][device].append(rate)
            group["targetGBps"].append(finite_float(row["targetGBps"], "targetGBps"))
            group["dutyCycle"].append(finite_float(row["dutyCycle"], "dutyCycle"))
            if background_data is not None:
                if isinstance(background_data.get("workingSetBytes"), int):
                    group["workingSetBytes"].append(background_data["workingSetBytes"])
            if isinstance(victim_data.get("streamCount"), int):
                group["victimStreamCounts"].append(victim_data["streamCount"])
            if isinstance(victim_data.get("p2pSlowCount"), int):
                group["victimSlowCounts"].append(victim_data["p2pSlowCount"])

            cases.append(
                {
                    "backgroundPath": background_path,
                    "victimMode": row["victimMode"],
                    "topology": row["topology"],
                    "repetition": int(row["repetition"]),
                    "victimAggregateGBps": victim_gbps,
                    "backgroundAggregateGBps": background_gbps,
                    "victimPerSourceGBps": dict(zip(row_devices, victim_rates)),
                    "backgroundPerDeviceGBps": (
                        dict(zip(row_devices, background_rates))
                        if background_gbps is not None
                        else {}
                    ),
                    "victimJson": victim_path,
                    "backgroundJson": row["backgroundJson"],
                }
            )

    if devices is None:
        raise ValueError("summary contains no pass rows")

    finished_groups = []
    for group in groups.values():
        item = {
            "backgroundPath": group["backgroundPath"],
            "victimMode": group["victimMode"],
            "topology": group["topology"],
            "caseCount": len(group["repetitions"]),
            "victimAggregateGBps": stats(group["victimAggregateGBps"]),
            "victimPerSourceGBps": per_device_stats(
                group["victimPerSourceGBps"], devices
            ),
            "backgroundAggregateGBps": stats(group["backgroundAggregateGBps"]),
            "backgroundPerDeviceGBps": per_device_stats(
                group["backgroundPerDeviceGBps"], devices
            ),
            "targetGBps": stats(group["targetGBps"]),
            "dutyCycle": stats(group["dutyCycle"]),
            "workingSetBytes": stats(group["workingSetBytes"]),
            "victimStreamCount": stats(group["victimStreamCounts"]),
            "victimSlowCount": stats(group["victimSlowCounts"]),
        }
        finished_groups.append(item)
    finished_groups.sort(
        key=lambda item: (
            BACKGROUND_ORDER.get(item["backgroundPath"], 999),
            VICTIM_ORDER.get(item["victimMode"], 999),
            TOPOLOGY_ORDER.get(item["topology"], 999),
        )
    )

    return {
        "summary": str(summary_path),
        "totalRows": total_rows,
        "caseCount": len(cases),
        "ignoredRows": ignored_rows,
        "deviceList": devices,
        "threeGpuAdaptation": {
            "singleTwoCopyDescription": "one source stream carries two consecutive allpairs P2P copies",
            "edgeActiveCopiesPerSource": 2,
            "fourGpuOneVsTwoAssignmentComparable": False,
        },
        "groups": finished_groups,
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    output = json.dumps(result, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(output)
    else:
        Path(args.output).expanduser().write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
