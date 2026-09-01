#!/usr/bin/env python3
"""Extract per-operation, rank, and channel metrics from NCCL Inspector JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CASE_PARTS = (
    "allgather",
    "allreduce",
    "reducescatter",
)
SCENARIOS = ("clean-before", "d2h-all", "clean-after")


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def median(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def numeric(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def load_metadata(root: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    path = root / "run-metadata.txt"
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
    except OSError:
        pass
    return metadata


def case_info(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 7 or parts[-1] != "nccl-tests.log":
        return None
    if parts[0] not in CASE_PARTS or not parts[1].startswith("size_"):
        return None
    if not parts[2].startswith("warmup_") or not parts[3].startswith("iters_"):
        return None
    if not parts[4].startswith("rep_") or parts[5] not in SCENARIOS:
        return None
    return {
        "collective": parts[0],
        "size": parts[1][len("size_") :],
        "warmup": int(parts[2][len("warmup_") :]),
        "iterations": int(parts[3][len("iters_") :]),
        "repetition": int(parts[4][len("rep_") :]),
        "scenario": parts[5],
        "case_dir": str(path.parent),
    }


def finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def parse_inspector_file(path: Path, root: Path) -> List[Dict[str, Any]]:
    info = case_info(path.parent.parent / "nccl-tests.log", root)
    if info is None:
        return []
    latest: Dict[Tuple[int, int], Dict[str, Any]] = {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    for line_number, line in enumerate(lines, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        header = item.get("header")
        coll_perf = item.get("coll_perf")
        if not isinstance(header, dict) or not isinstance(coll_perf, dict):
            continue
        rank_value = header.get("rank")
        coll_sn_value = coll_perf.get("coll_sn")
        if not isinstance(rank_value, int) or not isinstance(coll_sn_value, int):
            continue
        exec_us = finite_number(coll_perf.get("coll_exec_time_us"))
        if exec_us is None:
            continue
        channel_durations: Dict[int, float] = {}
        channel_starts: Dict[int, float] = {}
        channel_stops: Dict[int, float] = {}
        trace = coll_perf.get("event_trace_ts")
        if isinstance(trace, dict):
            events = trace.get("kernel_events")
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    channel = event.get("channel_id")
                    start = finite_number(event.get("kernel_start_ts"))
                    stop = finite_number(event.get("kernel_stop_ts"))
                    if not isinstance(channel, int) or start is None or stop is None:
                        continue
                    if stop >= start:
                        channel_durations[channel] = stop - start
                        channel_starts[channel] = start
                        channel_stops[channel] = stop
        dump_timestamp = finite_number(item.get("metadata", {}).get("dump_timestamp_us"))
        record = {
            **info,
            "rank": rank_value,
            "coll_sn": coll_sn_value,
            "coll": coll_perf.get("coll", ""),
            "coll_msg_size_bytes": coll_perf.get("coll_msg_size_bytes", ""),
            "coll_exec_time_us": exec_us,
            "coll_timing_source": coll_perf.get("coll_timing_source", ""),
            "coll_algobw_gbps": coll_perf.get("coll_algobw_gbs", ""),
            "coll_busbw_gbps": coll_perf.get("coll_busbw_gbs", ""),
            "channel_durations": channel_durations,
            "channel_starts": channel_starts,
            "channel_stops": channel_stops,
            "dump_timestamp_us": dump_timestamp if dump_timestamp is not None else -1,
            "source_file": str(path),
            "source_line": line_number,
        }
        key = (rank_value, coll_sn_value)
        previous = latest.get(key)
        if previous is None or record["dump_timestamp_us"] >= previous["dump_timestamp_us"]:
            latest[key] = record
    return list(latest.values())


def expand_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    operation_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    for record in records:
        durations = record["channel_durations"]
        duration_values = list(durations.values())
        channel_max = max(duration_values) if duration_values else None
        channel_min = min(duration_values) if duration_values else None
        channel_mean = statistics.mean(duration_values) if duration_values else None
        channel_cv = (
            statistics.pstdev(duration_values) / channel_mean * 100.0
            if len(duration_values) > 1 and channel_mean
            else 0.0 if duration_values else None
        )
        max_channel_id = (
            max(durations, key=durations.get) if durations else ""
        )
        starts = list(record["channel_starts"].values())
        stops = list(record["channel_stops"].values())
        operation_rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"channel_durations", "channel_starts", "channel_stops"}
            }
            | {
                "channel_count": len(duration_values),
                "channel_min_us": channel_min if channel_min is not None else "",
                "channel_mean_us": channel_mean if channel_mean is not None else "",
                "channel_max_us": channel_max if channel_max is not None else "",
                "channel_cv_pct": channel_cv if channel_cv is not None else "",
                "max_channel_id": max_channel_id,
                "kernel_span_us": (max(stops) - min(starts)) if starts and stops else "",
            }
        )
        for channel_id, duration in durations.items():
            channel_rows.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"channel_durations", "channel_starts", "channel_stops"}
                }
                | {
                    "channel_id": channel_id,
                    "channel_duration_us": duration,
                }
            )
    return operation_rows, channel_rows


OPERATION_FIELDS = [
    "collective",
    "size",
    "warmup",
    "iterations",
    "repetition",
    "scenario",
    "case_dir",
    "rank",
    "coll_sn",
    "coll",
    "coll_msg_size_bytes",
    "coll_exec_time_us",
    "coll_timing_source",
    "coll_algobw_gbps",
    "coll_busbw_gbps",
    "channel_count",
    "channel_min_us",
    "channel_mean_us",
    "channel_max_us",
    "channel_cv_pct",
    "max_channel_id",
    "kernel_span_us",
    "dump_timestamp_us",
    "source_file",
    "source_line",
]
CHANNEL_FIELDS = OPERATION_FIELDS[:]
CHANNEL_FIELDS.extend(["channel_id", "channel_duration_us"])


def write_csv(path: Path, fields: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_groups(
    rows: List[Dict[str, Any]],
    key_fields: Sequence[str],
    value_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result = dict(zip(key_fields, key))
        result["n_rows"] = len(group)
        for field in value_fields:
            values = numeric(group, field)
            result[f"{field}_p50"] = percentile(values, 0.50) or ""
            result[f"{field}_p90"] = percentile(values, 0.90) or ""
            result[f"{field}_p99"] = percentile(values, 0.99) or ""
            result[f"{field}_max"] = max(values) if values else ""
        result["timing_source_kernel_gpu_pct"] = (
            100.0
            * sum(row.get("coll_timing_source") == "kernel_gpu" for row in group)
            / len(group)
        )
        output.append(result)
    return output


def write_rank_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    key_fields = ["collective", "size", "warmup", "iterations", "scenario", "rank"]
    value_fields = ["coll_exec_time_us", "coll_busbw_gbps", "channel_max_us", "channel_cv_pct", "kernel_span_us"]
    summary = summarize_groups(rows, key_fields, value_fields)
    fields = key_fields + ["n_rows", "timing_source_kernel_gpu_pct"]
    for value_field in value_fields:
        fields.extend(
            [
                f"{value_field}_p50",
                f"{value_field}_p90",
                f"{value_field}_p99",
                f"{value_field}_max",
            ]
        )
    write_csv(path, fields, summary)


def write_channel_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    key_fields = ["collective", "size", "warmup", "iterations", "scenario", "rank", "channel_id"]
    summary = summarize_groups(rows, key_fields, ["channel_duration_us"])
    fields = key_fields + ["n_rows", "timing_source_kernel_gpu_pct"]
    fields.extend(
        [
            "channel_duration_us_p50",
            "channel_duration_us_p90",
            "channel_duration_us_p99",
            "channel_duration_us_max",
        ]
    )
    write_csv(path, fields, summary)


def write_markdown(root: Path, rows: List[Dict[str, Any]], path: Path) -> None:
    rank_keys = ["collective", "size", "warmup", "iterations", "scenario", "rank"]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in rank_keys), []).append(row)
    metadata = load_metadata(root)
    lines = [
        "# NCCL Inspector trace summary",
        "",
        f"- Input root: `{root}`",
        f"- Unique operation records: {len(rows)}",
        f"- Background devices: `{metadata.get('background_devices', 'unknown')}` local device list",
        "- Duplicate `(rank, coll_sn)` records are collapsed to the latest Inspector dump.",
        "- Channel durations are `kernel_stop_ts - kernel_start_ts` in the Inspector trace units, reported as microseconds for this build.",
        "",
        "| Collective | Size | Scenario | Rank | Ops | Exec p50 us | Exec p99 us | Channel max p50 us | Channel max p99 us | Channel CV p50 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(groups, key=lambda value: tuple(str(x) for x in value)):
        group = groups[key]
        exec_values = numeric(group, "coll_exec_time_us")
        channel_values = numeric(group, "channel_max_us")
        cv_values = numeric(group, "channel_cv_pct")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key[0]),
                    str(key[1]),
                    str(key[4]),
                    str(key[5]),
                    str(len(group)),
                    fmt(percentile(exec_values, 0.50)),
                    fmt(percentile(exec_values, 0.99)),
                    fmt(percentile(channel_values, 0.50)),
                    fmt(percentile(channel_values, 0.99)),
                    fmt(percentile(cv_values, 0.50)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This report is a localization aid. It can show that a collective/rank/channel execution interval changes under D2H, but it cannot identify a particular fence instruction as the cause.",
            "",
            "Per-operation data are in `trace-records.csv`; rank and channel summaries are in `trace-by-rank.csv` and `trace-by-channel.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Inspector-enabled regime result root")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"result root does not exist: {root}")
    raw_records: List[Dict[str, Any]] = []
    for inspector_file in sorted(root.rglob("inspector/*.log")):
        raw_records.extend(parse_inspector_file(inspector_file, root))
    operation_rows, channel_rows = expand_records(raw_records)
    if not operation_rows:
        parser.error(f"no Inspector coll_perf records found under: {root}")
    write_csv(root / "trace-records.csv", OPERATION_FIELDS, operation_rows)
    write_csv(root / "trace-channels.csv", CHANNEL_FIELDS, channel_rows)
    write_rank_summary(operation_rows, root / "trace-by-rank.csv")
    write_channel_summary(channel_rows, root / "trace-by-channel.csv")
    write_markdown(root, operation_rows, root / "trace-summary.md")
    print(f"unique_operation_records={len(operation_rows)} channel_rows={len(channel_rows)}")
    print(f"summary={root / 'trace-summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
