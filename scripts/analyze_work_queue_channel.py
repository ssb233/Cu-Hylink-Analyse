#!/usr/bin/env python3
"""Summarize the three-GPU Stage F work-queue matrix read-only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


REQUIRED_COLUMNS = {
    "topology",
    "connectionValue",
    "scenario",
    "repetition",
    "deviceList",
    "d2dSize",
    "backgroundSize",
    "d2dAggregateGBps",
    "backgroundAggregateGBps",
    "status",
    "d2dJson",
    "backgroundJson",
}

TOPOLOGY_ORDER = {
    "single-two-copy": 0,
    "edge-independent": 1,
    "edge-source-chain": 2,
}
SCENARIO_ORDER = {"none": 0, "d2h-all": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a three-GPU Stage F work-queue summary.csv."
    )
    parser.add_argument("--summary", required=True, help="runner summary.csv")
    parser.add_argument(
        "--output", default="-", help="JSON output path, or - for stdout"
    )
    return parser.parse_args()


def parse_devices(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise ValueError(f"invalid deviceList: {text}")
    devices = [int(item) for item in values]
    if len(devices) != 3 or len(set(devices)) != 3:
        raise ValueError(f"Stage F analyzer requires three unique devices: {text}")
    return devices


def parse_number(value: Any, field: str) -> float:
    if value is None or value == "" or value == "NA":
        raise ValueError(f"missing numeric {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric {field}: {value}")
    return number


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
    }


def load_json(summary_path: Path, value: str, field: str) -> dict[str, Any] | None:
    if value == "NA":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = summary_path.parent / path
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"{field} is not a JSON object: {path}")
    return loaded


def source_values(result: dict[str, Any], devices: list[int]) -> dict[int, float]:
    entries = result.get("sourceResults")
    if not isinstance(entries, list):
        raise ValueError("D2D JSON is missing sourceResults")
    values: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "device" not in entry or "GBps" not in entry:
            raise ValueError("D2D sourceResults contains an invalid entry")
        device = int(entry["device"])
        if device in values:
            raise ValueError(f"duplicate D2D source device: {device}")
        values[device] = parse_number(entry["GBps"], "sourceResults.GBps")
    if set(values) != set(devices):
        raise ValueError(
            f"D2D source devices {sorted(values)} do not match {sorted(devices)}"
        )
    return values


def background_values(
    result: dict[str, Any], devices: list[int]
) -> tuple[float, dict[int, float]]:
    aggregate = parse_number(result.get("aggregateGBps"), "background.aggregateGBps")
    result_devices = result.get("devices")
    if result_devices != devices:
        raise ValueError(
            f"background devices {result_devices} do not match {devices}"
        )
    bytes_values = result.get("perDeviceBytes")
    rate_values = result.get("perDeviceGBps")
    if not isinstance(bytes_values, list) or not isinstance(rate_values, list):
        raise ValueError("background JSON is missing per-device accounting")
    if len(bytes_values) != len(devices) or len(rate_values) != len(devices):
        raise ValueError("background per-device arrays do not match device count")
    parsed_bytes = [int(value) for value in bytes_values]
    if sum(parsed_bytes) != int(result.get("totalBytes", -1)):
        raise ValueError("background per-device bytes do not sum to totalBytes")
    return aggregate, {
        device: parse_number(rate, "background.perDeviceGBps")
        for device, rate in zip(devices, rate_values)
    }


def connection_sort_key(value: str) -> tuple[int, int | str]:
    if value == "unset":
        return (0, 0)
    return (1, int(value))


def analyze(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"summary is missing columns: {', '.join(missing)}")
        rows = [row for row in reader if row.get("status") == "pass"]
    if not rows:
        raise ValueError("summary contains no passing rows")

    devices = parse_devices(rows[0]["deviceList"])
    for row in rows:
        if parse_devices(row["deviceList"]) != devices:
            raise ValueError("deviceList changes within summary")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["topology"], row["connectionValue"], row["scenario"])].append(row)

    groups: list[dict[str, Any]] = []
    for (topology, connection_value, scenario), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            TOPOLOGY_ORDER.get(item[0][0], 99),
            connection_sort_key(item[0][1]),
            SCENARIO_ORDER.get(item[0][2], 99),
        ),
    ):
        d2d_aggregate: list[float] = []
        d2d_sources: dict[int, list[float]] = defaultdict(list)
        background_aggregate: list[float] = []
        background_rates: dict[int, list[float]] = defaultdict(list)
        for row in group_rows:
            d2d_json = load_json(summary_path, row["d2dJson"], "d2dJson")
            if d2d_json is None:
                raise ValueError("d2dJson cannot be NA for a passing row")
            json_devices = d2d_json.get("devices")
            if json_devices != devices:
                raise ValueError(f"D2D devices {json_devices} do not match {devices}")
            d2d_aggregate.append(parse_number(d2d_json.get("aggregateGBps"), "d2d.aggregateGBps"))
            for device, value in source_values(d2d_json, devices).items():
                d2d_sources[device].append(value)

            background_json = load_json(
                summary_path, row["backgroundJson"], "backgroundJson"
            )
            if background_json is not None:
                aggregate, rates = background_values(background_json, devices)
                background_aggregate.append(aggregate)
                for device, value in rates.items():
                    background_rates[device].append(value)

        groups.append(
            {
                "topology": topology,
                "connectionValue": connection_value,
                "scenario": scenario,
                "caseCount": len(group_rows),
                "d2dAggregateGBps": stats(d2d_aggregate),
                "d2dPerSourceGBps": {
                    "meanByDevice": {
                        str(device): fmean(d2d_sources[device]) for device in devices
                    },
                    "minByDevice": {
                        str(device): min(d2d_sources[device]) for device in devices
                    },
                    "maxByDevice": {
                        str(device): max(d2d_sources[device]) for device in devices
                    },
                },
                "backgroundAggregateGBps": stats(background_aggregate),
                "backgroundPerDeviceGBps": {
                    "meanByDevice": {
                        str(device): fmean(background_rates[device])
                        for device in devices
                    }
                    if background_rates
                    else {},
                    "minByDevice": {
                        str(device): min(background_rates[device]) for device in devices
                    }
                    if background_rates
                    else {},
                    "maxByDevice": {
                        str(device): max(background_rates[device]) for device in devices
                    }
                    if background_rates
                    else {},
                },
                "d2dSize": group_rows[0]["d2dSize"],
                "backgroundSize": group_rows[0]["backgroundSize"],
            }
        )

    return {
        "summary": str(summary_path.resolve()),
        "caseCount": len(rows),
        "deviceList": devices,
        "threeGpuAdaptation": {
            "devices": len(devices),
            "singleTwoCopyCopiesPerSource": 2,
            "singleTwoCopyDescription": (
                "one source stream carries two consecutive allpairs P2P copies"
            ),
            "edgeActiveCopiesPerSource": 2,
            "fourGpuOneVsTwoAssignmentComparable": False,
        },
        "groups": groups,
    }


def main() -> int:
    args = parse_args()
    try:
        result = analyze(Path(args.summary).expanduser())
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output == "-":
            sys.stdout.write(text)
        else:
            Path(args.output).write_text(text, encoding="utf-8")
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
