#!/usr/bin/env python3
"""Summarize clean NCCL Inspector OFF/ON calibration cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FLOAT = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"
FULL_MEASUREMENT_RE = re.compile(
    rf"(?m)^[ \t]*[0-9]+\s+[0-9]+\s+\S+\s+\S+\s+-?[0-9]+\s+"
    rf"({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+([0-9]+)"
)
SHORT_MEASUREMENT_RE = re.compile(
    rf"(?m)^[ \t]+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+([0-9]+)"
)
CASE_RE = re.compile(
    r"^(?P<collective>allgather|allreduce|reducescatter)/"
    r"size_(?P<size>[^/]+)/warmup_(?P<warmup>[0-9]+)/"
    r"iters_(?P<iterations>[0-9]+)/rep_(?P<repetition>[0-9]+)/"
    r"(?P<mode>inspector-off|inspector-on)/nccl-tests\.log$"
)

CASE_FIELDS = [
    "collective",
    "size",
    "warmup",
    "iterations",
    "repetition",
    "mode",
    "case_dir",
    "status",
    "time_ms",
    "algbw_gbps",
    "busbw_gbps",
    "error_count",
    "inspector_files",
    "inspector_records",
    "inspector_kernel_gpu_records",
    "inspector_channel_events",
    "parse_note",
]


def finite_or_none(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_measurement(path: Path) -> Tuple[Optional[Tuple[float, float, float, int]], str]:
    try:
        content = path.read_text(errors="replace")
    except OSError as error:
        return None, f"cannot read log: {error}"
    matches = list(FULL_MEASUREMENT_RE.finditer(content))
    if not matches:
        matches = list(SHORT_MEASUREMENT_RE.finditer(content))
    if not matches:
        return None, "no nccl-tests measurement row"
    match = matches[-1]
    values = [finite_or_none(match.group(index)) for index in range(1, 4)]
    if any(value is None for value in values):
        return None, "measurement contains non-finite values"
    return (values[0], values[1], values[2], int(match.group(4))), ""  # type: ignore[arg-type]


def inspector_counts(case_dir: Path) -> Tuple[int, int, int, int]:
    inspector_dir = case_dir / "inspector"
    if not inspector_dir.is_dir():
        return 0, 0, 0, 0
    files = 0
    records = 0
    kernel_gpu_records = 0
    channel_events = 0
    for path in inspector_dir.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            coll_perf = item.get("coll_perf") if isinstance(item, dict) else None
            if not isinstance(coll_perf, dict):
                continue
            records += 1
            if coll_perf.get("coll_timing_source") == "kernel_gpu":
                kernel_gpu_records += 1
            event_trace = coll_perf.get("event_trace_ts")
            if isinstance(event_trace, dict):
                events = event_trace.get("kernel_events")
                if isinstance(events, list):
                    channel_events += len(events)
    return files, records, kernel_gpu_records, channel_events


def parse_case(log_path: Path, root: Path) -> Dict[str, Any]:
    relative = str(log_path.relative_to(root)).replace("\\", "/")
    match = CASE_RE.match(relative)
    if match is None:
        return {}
    values = match.groupdict()
    case_dir = log_path.parent
    measurement, note = parse_measurement(log_path)
    status_json = load_json(case_dir / "status.json")
    test_rc = int(status_json.get("testRc", 1)) if status_json else 1
    status = "pass" if test_rc == 0 and measurement is not None else "fail"
    if status_json is None:
        note = (note + "; " if note else "") + "missing status.json"
    files, records, kernel_gpu_records, channel_events = inspector_counts(case_dir)
    row: Dict[str, Any] = {
        "collective": values["collective"],
        "size": values["size"],
        "warmup": int(values["warmup"]),
        "iterations": int(values["iterations"]),
        "repetition": int(values["repetition"]),
        "mode": values["mode"],
        "case_dir": str(case_dir),
        "status": status,
        "time_ms": "",
        "algbw_gbps": "",
        "busbw_gbps": "",
        "error_count": "",
        "inspector_files": files,
        "inspector_records": records,
        "inspector_kernel_gpu_records": kernel_gpu_records,
        "inspector_channel_events": channel_events,
        "parse_note": note,
    }
    if measurement is not None:
        row["time_ms"], row["algbw_gbps"], row["busbw_gbps"], row["error_count"] = measurement
    return row


def numeric(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def write_case_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_point_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    groups: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["collective"]),
            str(row["size"]),
            int(row["warmup"]),
            int(row["iterations"]),
            str(row["mode"]),
        )
        groups.setdefault(key, []).append(row)
    points = sorted(
        {(key[0], key[1], key[2], key[3]) for key in groups},
        key=lambda value: (value[0], value[1], value[2], value[3]),
    )
    fields = [
        "collective",
        "size",
        "warmup",
        "iterations",
        "off_n",
        "off_pass",
        "on_n",
        "on_pass",
        "off_time_median_ms",
        "on_time_median_ms",
        "time_overhead_pct",
        "off_busbw_median_gbps",
        "on_busbw_median_gbps",
        "busbw_drop_pct",
        "on_inspector_records_median",
        "on_kernel_gpu_record_pct",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            off = groups.get((*point, "inspector-off"), [])
            on = groups.get((*point, "inspector-on"), [])
            off_time = median(numeric(off, "time_ms"))
            on_time = median(numeric(on, "time_ms"))
            off_bus = median(numeric(off, "busbw_gbps"))
            on_bus = median(numeric(on, "busbw_gbps"))
            on_records = median(numeric(on, "inspector_records"))
            on_kernel = sum(numeric(on, "inspector_kernel_gpu_records"))
            on_total = sum(numeric(on, "inspector_records"))
            writer.writerow(
                {
                    "collective": point[0],
                    "size": point[1],
                    "warmup": point[2],
                    "iterations": point[3],
                    "off_n": len(off),
                    "off_pass": sum(row["status"] == "pass" for row in off),
                    "on_n": len(on),
                    "on_pass": sum(row["status"] == "pass" for row in on),
                    "off_time_median_ms": fmt(off_time),
                    "on_time_median_ms": fmt(on_time),
                    "time_overhead_pct": fmt((on_time / off_time - 1.0) * 100.0)
                    if off_time and on_time
                    else "n/a",
                    "off_busbw_median_gbps": fmt(off_bus),
                    "on_busbw_median_gbps": fmt(on_bus),
                    "busbw_drop_pct": fmt((1.0 - on_bus / off_bus) * 100.0)
                    if off_bus and on_bus
                    else "n/a",
                    "on_inspector_records_median": fmt(on_records, 0),
                    "on_kernel_gpu_record_pct": fmt(100.0 * on_kernel / on_total, 1)
                    if on_total
                    else "n/a",
                }
            )


def write_markdown(root: Path, rows: List[Dict[str, Any]], path: Path) -> None:
    grouped: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["collective"]),
            str(row["size"]),
            int(row["warmup"]),
            int(row["iterations"]),
            str(row["mode"]),
        )
        grouped.setdefault(key, []).append(row)
    points = sorted(
        {(key[0], key[1], key[2], key[3]) for key in grouped},
        key=lambda value: (value[0], value[1], value[2], value[3]),
    )
    failed = [row for row in rows if row["status"] != "pass"]
    lines = [
        "# NCCL Inspector calibration summary",
        "",
        f"- Input root: `{root}`",
        f"- Parsed cases: {len(rows)}; failed or incomplete: {len(failed)}",
        "- `time_overhead_pct` compares Inspector ON median time with OFF median time.",
        "- `busbw_drop_pct` is the corresponding reduction in nccl-tests bus bandwidth.",
        "- Inspector records are counted from JSONL records containing `coll_perf`.",
        "",
        "| Collective | Size | Warmup | Iters | OFF time ms | ON time ms | Time overhead | OFF busbw | ON busbw | Busbw drop | ON records | kernel_gpu |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        off = grouped.get((*point, "inspector-off"), [])
        on = grouped.get((*point, "inspector-on"), [])
        off_time = median(numeric(off, "time_ms"))
        on_time = median(numeric(on, "time_ms"))
        off_bus = median(numeric(off, "busbw_gbps"))
        on_bus = median(numeric(on, "busbw_gbps"))
        on_records = median(numeric(on, "inspector_records"))
        on_kernel = sum(numeric(on, "inspector_kernel_gpu_records"))
        on_total = sum(numeric(on, "inspector_records"))
        lines.append(
            "| "
            + " | ".join(
                [
                    point[0],
                    point[1],
                    str(point[2]),
                    str(point[3]),
                    fmt(off_time),
                    fmt(on_time),
                    fmt((on_time / off_time - 1.0) * 100.0) + "%"
                    if off_time and on_time
                    else "n/a",
                    fmt(off_bus),
                    fmt(on_bus),
                    fmt((1.0 - on_bus / off_bus) * 100.0) + "%"
                    if off_bus and on_bus
                    else "n/a",
                    fmt(on_records, 0),
                    fmt(100.0 * on_kernel / on_total, 1) + "%"
                    if on_total
                    else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The calibration uses the third_party NCCL v2.31.2 sm70 `.sys` build on three V100 GPUs. It is not an official-vs-third-party equivalence test and does not include D2H background traffic.",
            "",
            "Per-case data are in `summary.csv`; point-level data are in `summary-by-point.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Inspector calibration result root")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"result root does not exist: {root}")
    rows = [parse_case(path, root) for path in sorted(root.rglob("nccl-tests.log"))]
    rows = [row for row in rows if row]
    if not rows:
        parser.error(f"no calibration cases found under: {root}")
    write_case_csv(rows, root / "summary.csv")
    write_point_summary(rows, root / "summary-by-point.csv")
    write_markdown(root, rows, root / "summary.md")
    print(f"parsed_cases={len(rows)} failed={sum(row['status'] != 'pass' for row in rows)}")
    print(f"summary={root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
