#!/usr/bin/env python3
"""Analyze CSV rows emitted by the CUPTI memcpy activity tracer.

The CUPTI channel identifiers are intentionally reported as scoped activity
metadata.  They are not interpreted as physical copy-engine instance names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


P2P_COPY_KIND = 10
H2D_COPY_KIND = 1
D2H_COPY_KIND = 2
DTOD_COPY_KIND = 8

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CUPTI memcpy/channel CSV activity rows."
    )
    parser.add_argument("--trace", required=True, help="CSV trace to read")
    parser.add_argument(
        "--metadata",
        default=None,
        help="optional dropped-record JSON; defaults to <trace>.meta.json",
    )
    parser.add_argument(
        "--slow-threshold-ms",
        type=float,
        default=8.0,
        help="duration strictly above this value is slow (default: 8.0)",
    )
    parser.add_argument("--output", default="-", help="JSON output path, or -")
    args = parser.parse_args()
    if args.slow_threshold_ms <= 0.0 or not math.isfinite(args.slow_threshold_ms):
        parser.error("--slow-threshold-ms must be a positive finite number")
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
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: {field} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return parsed


def kind_name(copy_kind: int) -> str:
    return {
        P2P_COPY_KIND: "p2p",
        H2D_COPY_KIND: "h2d",
        D2H_COPY_KIND: "d2h",
        DTOD_COPY_KIND: "dtod",
    }.get(copy_kind, f"copyKind-{copy_kind}")


def load_dropped_records(trace: Path, metadata_argument: str | None) -> tuple[int | None, str]:
    metadata_path = Path(metadata_argument) if metadata_argument else Path(f"{trace}.meta.json")
    if not metadata_path.is_file():
        return None, "unavailable"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if "droppedRecords" not in metadata:
        raise ValueError(f"metadata is missing droppedRecords: {metadata_path}")
    dropped = metadata["droppedRecords"]
    if not isinstance(dropped, int) or dropped < 0:
        raise ValueError(f"metadata droppedRecords must be a non-negative integer: {metadata_path}")
    return dropped, str(metadata_path)


def duration_ms(row: dict[str, str], row_number: int) -> float:
    duration = parse_float(row["durationMs"], "durationMs", row_number)
    start = parse_int(row["startNs"], "startNs", row_number)
    end = parse_int(row["endNs"], "endNs", row_number)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"row {row_number}: invalid startNs/endNs interval")
    derived = (end - start) / 1_000_000.0
    if duration < 0.0 or abs(duration - derived) > max(0.05, derived * 0.25):
        raise ValueError(
            f"row {row_number}: durationMs disagrees with startNs/endNs "
            f"({duration} vs {derived})"
        )
    return duration


def read_rows(trace: Path, slow_threshold_ms: float) -> list[dict]:
    if not trace.is_file():
        raise FileNotFoundError(f"trace does not exist: {trace}")

    rows: list[dict] = []
    with trace.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("trace CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "trace CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"row {row_number}: malformed CSV row")
            parsed = {
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
                "startNs": parse_int(raw["startNs"], "startNs", row_number),
                "endNs": parse_int(raw["endNs"], "endNs", row_number),
                "durationMs": duration_ms(raw, row_number),
                "correlationId": parse_int(raw["correlationId"], "correlationId", row_number),
                "activityKind": parse_int(raw["activityKind"], "activityKind", row_number),
            }
            if parsed["bytes"] < 0:
                raise ValueError(f"row {row_number}: bytes must be non-negative")
            parsed["kind"] = kind_name(parsed["copyKind"])
            parsed["slow"] = parsed["durationMs"] > slow_threshold_ms
            rows.append(parsed)
    return rows


def increment_totals(target: dict, row: dict) -> None:
    target["count"] += 1
    target["slowCount"] += int(row["slow"])
    target["bytes"] += row["bytes"]
    target["totalDurationMs"] += row["durationMs"]
    target["channels"].add(row["channelKey"])
    target["streams"].add(row["streamKey"])


def finalize_totals(target: dict) -> dict:
    result = dict(target)
    result["meanDurationMs"] = (
        result["totalDurationMs"] / result["count"] if result["count"] else 0.0
    )
    result["slowFraction"] = (
        result["slowCount"] / result["count"] if result["count"] else 0.0
    )
    result["channelCount"] = len(result.pop("channels"))
    result["streamCount"] = len(result.pop("streams"))
    return result


def analyze(args: argparse.Namespace) -> dict:
    trace = Path(args.trace).expanduser().resolve()
    rows = read_rows(trace, args.slow_threshold_ms)
    dropped_records, metadata_source = load_dropped_records(trace, args.metadata)

    for row in rows:
        row["channelKey"] = (
            row["pid"],
            row["deviceId"],
            row["contextId"],
            row["channelType"],
            row["channelID"],
        )
        row["streamKey"] = (
            row["pid"],
            row["deviceId"],
            row["contextId"],
            row["streamId"],
        )

    zero_totals = lambda: {
        "count": 0,
        "slowCount": 0,
        "bytes": 0,
        "totalDurationMs": 0.0,
        "channels": set(),
        "streams": set(),
    }

    kind_totals: dict[str, dict] = defaultdict(zero_totals)
    activity_totals: Counter[str] = Counter()
    channel_totals: dict[tuple, dict] = {}
    stream_totals: dict[tuple, dict] = {}

    for row in rows:
        increment_totals(kind_totals[row["kind"]], row)
        activity_totals[str(row["activityKind"])] += 1

        channel = channel_totals.setdefault(
            row["channelKey"],
            {
                "pid": row["pid"],
                "deviceId": row["deviceId"],
                "contextId": row["contextId"],
                "channelType": row["channelType"],
                "channelID": row["channelID"],
                "count": 0,
                "slowCount": 0,
                "p2pCount": 0,
                "p2pSlowCount": 0,
                "d2hCount": 0,
                "h2dCount": 0,
                "bytes": 0,
                "totalDurationMs": 0.0,
            },
        )
        channel["count"] += 1
        channel["slowCount"] += int(row["slow"])
        channel["p2pCount"] += int(row["kind"] == "p2p")
        channel["p2pSlowCount"] += int(row["kind"] == "p2p" and row["slow"])
        channel["d2hCount"] += int(row["kind"] == "d2h")
        channel["h2dCount"] += int(row["kind"] == "h2d")
        channel["bytes"] += row["bytes"]
        channel["totalDurationMs"] += row["durationMs"]

        stream = stream_totals.setdefault(
            row["streamKey"],
            {
                "pid": row["pid"],
                "deviceId": row["deviceId"],
                "contextId": row["contextId"],
                "streamId": row["streamId"],
                "count": 0,
                "slowCount": 0,
                "p2pCount": 0,
                "p2pSlowCount": 0,
                "bytes": 0,
                "totalDurationMs": 0.0,
            },
        )
        stream["count"] += 1
        stream["slowCount"] += int(row["slow"])
        stream["p2pCount"] += int(row["kind"] == "p2p")
        stream["p2pSlowCount"] += int(row["kind"] == "p2p" and row["slow"])
        stream["bytes"] += row["bytes"]
        stream["totalDurationMs"] += row["durationMs"]

    def finish_group(group: dict) -> dict:
        result = dict(group)
        result["meanDurationMs"] = (
            result["totalDurationMs"] / result["count"] if result["count"] else 0.0
        )
        result["slowFraction"] = (
            result["slowCount"] / result["count"] if result["count"] else 0.0
        )
        result["p2pSlowFraction"] = (
            result["p2pSlowCount"] / result["p2pCount"]
            if result["p2pCount"]
            else 0.0
        )
        return result

    finished_channels = [finish_group(item) for item in channel_totals.values()]
    total_p2p_slow = sum(row["kind"] == "p2p" and row["slow"] for row in rows)
    for item in finished_channels:
        item["pSlowGivenChannel"] = item["p2pSlowFraction"]
        item["pChannelGivenSlowP2p"] = (
            item["p2pSlowCount"] / total_p2p_slow if total_p2p_slow else 0.0
        )
    finished_channels.sort(
        key=lambda item: (
            -item["p2pSlowCount"],
            -item["slowCount"],
            item["pid"],
            item["deviceId"],
            item["contextId"],
            item["channelType"],
            item["channelID"],
        )
    )
    finished_streams = [finish_group(item) for item in stream_totals.values()]
    finished_streams.sort(
        key=lambda item: (
            item["pid"], item["deviceId"], item["contextId"], item["streamId"]
        )
    )

    p2p_rows = [row for row in rows if row["kind"] == "p2p"]
    d2h_rows = [row for row in rows if row["kind"] == "d2h"]
    h2d_rows = [row for row in rows if row["kind"] == "h2d"]

    result = {
        "trace": str(trace),
        "slowThresholdMs": args.slow_threshold_ms,
        "totalRows": len(rows),
        "droppedRecords": dropped_records,
        "droppedRecordsSource": metadata_source,
        "streamCount": len(stream_totals),
        "channelCount": len(channel_totals),
        "p2pCount": len(p2p_rows),
        "p2pSlowCount": sum(row["slow"] for row in p2p_rows),
        "d2hCount": len(d2h_rows),
        "h2dCount": len(h2d_rows),
        "kindTotals": {
            name: finalize_totals(kind_totals[name])
            for name in sorted(kind_totals)
        },
        "activityKindTotals": dict(sorted(activity_totals.items())),
        "channelTotals": finished_channels,
        "slowByChannel": [
            item for item in finished_channels if item["slowCount"] > 0
        ],
        "streamTotals": finished_streams,
    }
    return result


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output == "-":
        sys.stdout.write(output)
    else:
        Path(args.output).expanduser().write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
