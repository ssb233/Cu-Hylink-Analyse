#!/usr/bin/env python3
"""Summarize one Stage I CUPTI memcpy trace.

The analyzer keeps CUPTI's runtime/driver correlation id and scoped channel
identity in every measured P2P activity.  Queue position is reconstructed from
the benchmark's edge issue order, while previous-activity gaps are computed
within the scoped ``(pid, device, context, channel type, channel)`` stream.
Background overlap is matched by device id, so a source-local victim is only
compared with background activity on that source device.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_COLUMNS = {
    "pid",
    "deviceId",
    "contextId",
    "streamId",
    "channelID",
    "channelType",
    "copyKind",
    "srcDeviceId",
    "dstDeviceId",
    "bytes",
    "startNs",
    "endNs",
    "durationMs",
    "correlationId",
    "activityKind",
}
P2P_COPY_KIND = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Stage I CUPTI victim/background memcpy trace."
    )
    parser.add_argument("--victim", required=True, help="victim CUPTI CSV")
    parser.add_argument("--background", default=None, help="optional background CUPTI CSV")
    parser.add_argument(
        "--edge-order",
        choices=("forward", "reverse"),
        required=True,
        help="victim issue order: forward=0->1,0->2; reverse=0->2,0->1",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--slow-threshold-ms", type=float, default=8.0)
    parser.add_argument("--output", default="-", help="JSON output path, or -")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not math.isfinite(args.slow_threshold_ms) or args.slow_threshold_ms <= 0:
        parser.error("--slow-threshold-ms must be positive and finite")
    return args


def parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field} must be an integer, got {value!r}"
        ) from error


def parse_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return result


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path} row {row_number}: malformed CSV row")
            start = parse_int(raw["startNs"], "startNs", row_number)
            end = parse_int(raw["endNs"], "endNs", row_number)
            if start < 0 or end < start:
                raise ValueError(f"{path} row {row_number}: invalid time interval")
            duration = parse_float(raw["durationMs"], "durationMs", row_number)
            derived = (end - start) / 1_000_000.0
            if duration < 0 or abs(duration - derived) > max(0.05, derived * 0.25):
                raise ValueError(
                    f"{path} row {row_number}: durationMs disagrees with timestamps"
                )
            rows.append(
                {
                    "pid": parse_int(raw["pid"], "pid", row_number),
                    "deviceId": parse_int(raw["deviceId"], "deviceId", row_number),
                    "contextId": parse_int(raw["contextId"], "contextId", row_number),
                    "streamId": parse_int(raw["streamId"], "streamId", row_number),
                    "channelID": parse_int(raw["channelID"], "channelID", row_number),
                    "channelType": parse_int(raw["channelType"], "channelType", row_number),
                    "copyKind": parse_int(raw["copyKind"], "copyKind", row_number),
                    "srcDeviceId": parse_int(raw["srcDeviceId"], "srcDeviceId", row_number),
                    "dstDeviceId": parse_int(raw["dstDeviceId"], "dstDeviceId", row_number),
                    "bytes": parse_int(raw["bytes"], "bytes", row_number),
                    "startNs": start,
                    "endNs": end,
                    "durationMs": duration,
                    "correlationId": parse_int(raw["correlationId"], "correlationId", row_number),
                    "activityKind": parse_int(raw["activityKind"], "activityKind", row_number),
                }
            )
    return rows


def load_dropped_records(path: Path) -> int | None:
    metadata = Path(f"{path}.meta.json")
    if not metadata.is_file():
        return None
    with metadata.open(encoding="utf-8") as stream:
        value = json.load(stream).get("droppedRecords")
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid droppedRecords in {metadata}")
    return value


def scoped_channel(row: dict[str, Any]) -> str:
    return (
        f"pid={row['pid']}/device={row['deviceId']}/context={row['contextId']}"
        f"/type={row['channelType']}/channel={row['channelID']}"
    )


def finite_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(materialized)

    def nearest_rank(quantile: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


def merged_intervals(rows: list[dict[str, Any]]) -> dict[int, list[tuple[int, int]]]:
    by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row["endNs"] > row["startNs"]:
            by_device[row["deviceId"]].append((row["startNs"], row["endNs"]))
    result: dict[int, list[tuple[int, int]]] = {}
    for device, intervals in by_device.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result[device] = merged
    return result


def overlap_ns(row: dict[str, Any], intervals: dict[int, list[tuple[int, int]]]) -> int:
    total = 0
    for start, end in intervals.get(row["deviceId"], []):
        if start >= row["endNs"]:
            break
        if end > row["startNs"]:
            total += max(0, min(row["endNs"], end) - max(row["startNs"], start))
    return total


def add_previous_gaps(rows: list[dict[str, Any]]) -> None:
    previous_end: dict[tuple[int, int, int, int, int], int] = {}
    for row in sorted(rows, key=lambda item: (item["startNs"], item["endNs"], item["correlationId"])):
        key = (
            row["pid"],
            row["deviceId"],
            row["contextId"],
            row["channelType"],
            row["channelID"],
        )
        prior = previous_end.get(key)
        row["previousActivityGapNs"] = None if prior is None else max(0, row["startNs"] - prior)
        previous_end[key] = max(previous_end.get(key, row["endNs"]), row["endNs"])


def edge_name(row: dict[str, Any]) -> str:
    return f"{row['srcDeviceId']}->{row['dstDeviceId']}"


def activity_record(
    row: dict[str, Any],
    occurrence: int,
    queue_position: int,
    overlap: int,
    slow_threshold_ms: float,
) -> dict[str, Any]:
    return {
        "edge": edge_name(row),
        "occurrence": occurrence,
        "repetition": occurrence,
        "queuePosition": queue_position,
        "pid": row["pid"],
        "deviceId": row["deviceId"],
        "contextId": row["contextId"],
        "streamId": row["streamId"],
        "channelID": row["channelID"],
        "channelType": row["channelType"],
        "scopedChannel": scoped_channel(row),
        "copyKind": row["copyKind"],
        "srcDeviceId": row["srcDeviceId"],
        "dstDeviceId": row["dstDeviceId"],
        "bytes": row["bytes"],
        "startNs": row["startNs"],
        "endNs": row["endNs"],
        "durationMs": row["durationMs"],
        "slow": row["durationMs"] > slow_threshold_ms,
        "correlationId": row["correlationId"],
        "activityKind": row["activityKind"],
        "previousActivityGapNs": row["previousActivityGapNs"],
        "backgroundOverlapNs": overlap,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    victim_path = Path(args.victim).expanduser().resolve()
    background_path = (
        Path(args.background).expanduser().resolve() if args.background else None
    )
    victim_rows = read_trace(victim_path)
    background_rows = read_trace(background_path) if background_path else []
    add_previous_gaps(victim_rows)
    background_intervals = merged_intervals(background_rows)

    order = [(0, 1), (0, 2)] if args.edge_order == "forward" else [(0, 2), (0, 1)]
    expected_edges = {f"{source}->{destination}" for source, destination in order}
    p2p_rows = [row for row in victim_rows if row["copyKind"] == P2P_COPY_KIND]
    by_edge_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(p2p_rows, key=lambda item: (item["startNs"], item["endNs"], item["correlationId"])):
        by_edge_rows[edge_name(row)].append(row)
    if set(by_edge_rows) != expected_edges:
        raise ValueError(
            "victim P2P edges do not match the requested order: "
            f"expected {sorted(expected_edges)}, got {sorted(by_edge_rows)}"
        )

    activities: list[dict[str, Any]] = []
    by_edge: dict[str, dict[str, Any]] = {}
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for queue_position, (source, destination) in enumerate(order, start=1):
        edge = f"{source}->{destination}"
        measured: list[dict[str, Any]] = []
        for occurrence, row in enumerate(by_edge_rows[edge]):
            if occurrence < args.warmup or occurrence >= args.warmup + args.repeats:
                continue
            record = activity_record(
                row,
                occurrence,
                queue_position,
                overlap_ns(row, background_intervals),
                args.slow_threshold_ms,
            )
            record["repetition"] = occurrence - args.warmup + 1
            measured.append(record)
            activities.append(record)
            by_position[str(queue_position)].append(record)
        durations = [row["durationMs"] for row in measured]
        gaps = [
            float(row["previousActivityGapNs"]) / 1_000_000.0
            for row in measured
            if row["previousActivityGapNs"] is not None
        ]
        overlaps = [row["backgroundOverlapNs"] for row in measured]
        correlation_ids = sorted({row["correlationId"] for row in measured})
        channels = sorted({row["scopedChannel"] for row in measured})
        by_edge[edge] = {
            "queuePosition": queue_position,
            "warmupCount": min(args.warmup, len(by_edge_rows[edge])),
            "count": len(measured),
            "durationMs": finite_stats(durations),
            "slowCount": sum(row["slow"] for row in measured),
            "slowFraction": (
                sum(row["slow"] for row in measured) / len(measured) if measured else 0.0
            ),
            "previousActivityGapMs": finite_stats(gaps),
            "backgroundOverlapCount": sum(value > 0 for value in overlaps),
            "backgroundOverlapFraction": (
                sum(value > 0 for value in overlaps) / len(overlaps) if overlaps else 0.0
            ),
            "backgroundOverlapNs": sum(overlaps),
            "correlationIdRange": [correlation_ids[0], correlation_ids[-1]] if correlation_ids else None,
            "correlationIdCount": len(correlation_ids),
            "scopedChannels": channels,
            "streamIds": sorted({row["streamId"] for row in measured}),
        }

    position_results: dict[str, dict[str, Any]] = {}
    for position, rows in sorted(by_position.items(), key=lambda item: int(item[0])):
        gaps = [
            float(row["previousActivityGapNs"]) / 1_000_000.0
            for row in rows
            if row["previousActivityGapNs"] is not None
        ]
        position_results[position] = {
            "count": len(rows),
            "edges": sorted({row["edge"] for row in rows}),
            "durationMs": finite_stats(row["durationMs"] for row in rows),
            "slowCount": sum(row["slow"] for row in rows),
            "slowFraction": sum(row["slow"] for row in rows) / len(rows) if rows else 0.0,
            "previousActivityGapMs": finite_stats(gaps),
            "backgroundOverlapCount": sum(row["backgroundOverlapNs"] > 0 for row in rows),
            "backgroundOverlapNs": sum(row["backgroundOverlapNs"] for row in rows),
        }

    dropped_victim = load_dropped_records(victim_path)
    dropped_background = load_dropped_records(background_path) if background_path else None
    return {
        "program": "analyze_directional_dma_trace",
        "victimTrace": str(victim_path),
        "backgroundTrace": str(background_path) if background_path else None,
        "edgeOrder": args.edge_order,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "slowThresholdMs": args.slow_threshold_ms,
        "victimRows": len(victim_rows),
        "backgroundRows": len(background_rows),
        "droppedRecords": {
            "victim": dropped_victim,
            "background": dropped_background,
            "total": (
                (dropped_victim or 0) + (dropped_background or 0)
                if dropped_victim is not None or dropped_background is not None
                else None
            ),
        },
        "p2p": {
            "totalCount": len(p2p_rows),
            "warmupCount": max(0, len(p2p_rows) - len(activities)),
            "measuredCount": len(activities),
            "slowCount": sum(row["slow"] for row in activities),
            "slowFraction": (
                sum(row["slow"] for row in activities) / len(activities)
                if activities
                else 0.0
            ),
            "byEdge": by_edge,
            "byQueuePosition": position_results,
            "activities": sorted(activities, key=lambda row: (row["startNs"], row["correlationId"])),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
        output = json.dumps(result, indent=2, sort_keys=False) + "\n"
        if args.output == "-":
            sys.stdout.write(output)
        else:
            Path(args.output).expanduser().write_text(output, encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
