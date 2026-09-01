#!/usr/bin/env python3
"""Parse and summarize the NCCL B-stage regime sweep.

The runner deliberately stores one nccl-tests invocation per case. This
analyzer keeps every parsed case in CSV and computes paired medians for the
clean-before, direction-specific background, and clean-after scenarios.
"""

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
# nccl-tests normally prints the size/count/type/root columns and then the
# out-of-place time/algbw/busbw/#wrong columns on one line. With NCCL debug
# output redirected elsewhere, this is the stable format used by the runner.
FULL_MEASUREMENT_RE = re.compile(
    rf"(?m)^[ \t]*[0-9]+\s+[0-9]+\s+\S+\s+\S+\s+-?[0-9]+\s+"
    rf"({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+([0-9]+)"
)
# Keep compatibility with older logs where stderr was interleaved after the
# nccl-tests header and the timing columns appeared on a standalone line.
SHORT_MEASUREMENT_RE = re.compile(
    rf"(?m)^[ \t]+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+([0-9]+)"
)
CASE_RE = re.compile(
    r"/(?P<collective>[^/]+)/size_(?P<size>[^/]+)/"
    r"warmup_(?P<warmup>[0-9]+)/iters_(?P<iterations>[0-9]+)/"
    r"rep_(?P<repetition>[0-9]+)/"
    r"(?P<scenario>clean-before|d2h-all|h2d-all|clean-after)/nccl-tests\.log$"
)
BACKGROUND_SCENARIOS = ("d2h-all", "h2d-all")

CASE_FIELDS = [
    "collective",
    "size",
    "warmup",
    "iterations",
    "repetition",
    "scenario",
    "case_dir",
    "status",
    "time_ms",
    "algbw_gbps",
    "busbw_gbps",
    "error_count",
    "background_aggregate_gbps",
    "background_per_device_gbps",
    "background_operations",
    "parse_note",
]


def finite_or_none(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_measurement(
    log_path: Path,
) -> Tuple[Optional[Tuple[float, float, float, int]], str]:
    try:
        text = log_path.read_text(errors="replace")
    except OSError as error:
        return None, f"cannot read log: {error}"
    matches = list(FULL_MEASUREMENT_RE.finditer(text))
    if not matches:
        matches = list(SHORT_MEASUREMENT_RE.finditer(text))
    if not matches:
        return None, "no nccl-tests measurement row"
    match = matches[-1]
    time_ms = finite_or_none(match.group(1))
    algbw_gbps = finite_or_none(match.group(2))
    busbw_gbps = finite_or_none(match.group(3))
    if time_ms is None or algbw_gbps is None or busbw_gbps is None:
        return None, "measurement contains non-finite values"
    return (time_ms, algbw_gbps, busbw_gbps, int(match.group(4))), ""


def parse_case(log_path: Path, root: Path) -> Dict[str, Any]:
    relative = "/" + str(log_path.relative_to(root)).replace("\\", "/")
    match = CASE_RE.search(relative)
    if match is None:
        return {}

    values: Dict[str, Any] = match.groupdict()
    case_dir = log_path.parent
    status_json = load_json(case_dir / "status.json")
    background_json = load_json(case_dir / "background.json")
    measurement, note = parse_measurement(log_path)

    test_rc = 0
    background_rc = 0
    if status_json is not None:
        test_rc = int(status_json.get("testRc", 1))
        background_rc = int(status_json.get("backgroundRc", 0))
    status = "pass" if test_rc == 0 and background_rc == 0 and measurement else "fail"
    if status_json is None:
        note = (note + "; " if note else "") + "missing status.json"
    if background_rc != 0:
        note = (note + "; " if note else "") + f"backgroundRc={background_rc}"

    row: Dict[str, Any] = {
        "collective": values["collective"],
        "size": values["size"],
        "warmup": int(values["warmup"]),
        "iterations": int(values["iterations"]),
        "repetition": int(values["repetition"]),
        "scenario": values["scenario"],
        "case_dir": str(case_dir),
        "status": status,
        "time_ms": "",
        "algbw_gbps": "",
        "busbw_gbps": "",
        "error_count": "",
        "background_aggregate_gbps": "",
        "background_per_device_gbps": "",
        "background_operations": "",
        "parse_note": note,
    }
    if measurement is not None:
        row["time_ms"], row["algbw_gbps"], row["busbw_gbps"], row["error_count"] = measurement
    if background_json is not None:
        aggregate = background_json.get("aggregateGBps")
        if isinstance(aggregate, (int, float)):
            row["background_aggregate_gbps"] = aggregate
        per_device = background_json.get("perDeviceGBps")
        if isinstance(per_device, list):
            row["background_per_device_gbps"] = json.dumps(
                per_device, separators=(",", ":")
            )
        operations = background_json.get("perDeviceOperations")
        if isinstance(operations, list):
            row["background_operations"] = json.dumps(
                operations, separators=(",", ":")
            )
    elif values["scenario"] in BACKGROUND_SCENARIOS:
        row["parse_note"] = (note + "; " if note else "") + "missing background.json"
        row["status"] = "fail"
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


def scenario_order(
    groups: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]],
    base_key: Tuple[str, str, int, int],
) -> List[str]:
    scenarios = ["clean-before"]
    scenarios.extend(
        scenario
        for scenario in BACKGROUND_SCENARIOS
        if (*base_key, scenario) in groups
    )
    scenarios.append("clean-after")
    return scenarios


def write_summary(root: Path, rows: List[Dict[str, Any]], output: Path) -> None:
    groups: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["collective"]),
            str(row["size"]),
            int(row["warmup"]),
            int(row["iterations"]),
            str(row["scenario"]),
        )
        groups.setdefault(key, []).append(row)

    base_keys = sorted(
        {(key[0], key[1], key[2], key[3]) for key in groups},
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "collective",
                "size",
                "warmup",
                "iterations",
                "scenario",
                "n_cases",
                "n_pass",
                "busbw_median_gbps",
                "busbw_mean_gbps",
                "algbw_median_gbps",
                "time_median_ms",
                "background_aggregate_median_gbps",
                "slowdown_vs_clean_pct",
                "recovery_vs_clean_pct",
            ]
        )
        for base_key in base_keys:
            for scenario in scenario_order(groups, base_key):
                key = (*base_key, scenario)
                case_rows = groups.get(key, [])
                busbw = numeric(case_rows, "busbw_gbps")
                algbw = numeric(case_rows, "algbw_gbps")
                times = numeric(case_rows, "time_ms")
                background = numeric(case_rows, "background_aggregate_gbps")
                clean_bus = median(
                    numeric(groups.get((*base_key, "clean-before"), []), "busbw_gbps")
                )
                current_bus = median(busbw)
                slowdown = ""
                recovery = ""
                if clean_bus is not None and clean_bus != 0 and current_bus is not None:
                    if scenario in BACKGROUND_SCENARIOS:
                        slowdown = (1.0 - current_bus / clean_bus) * 100.0
                    if scenario == "clean-after":
                        recovery = current_bus / clean_bus * 100.0
                writer.writerow(
                    [
                        *base_key,
                        scenario,
                        len(case_rows),
                        sum(row["status"] == "pass" for row in case_rows),
                        fmt(current_bus),
                        fmt(mean(busbw)),
                        fmt(median(algbw)),
                        fmt(median(times)),
                        fmt(median(background)),
                        fmt(float(slowdown) if slowdown != "" else None),
                        fmt(float(recovery) if recovery != "" else None),
                    ]
                )


def write_markdown(root: Path, rows: List[Dict[str, Any]], output: Path) -> None:
    grouped: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["collective"]),
            str(row["size"]),
            int(row["warmup"]),
            int(row["iterations"]),
            str(row["scenario"]),
        )
        grouped.setdefault(key, []).append(row)

    base_keys = sorted(
        {(key[0], key[1], key[2], key[3]) for key in grouped},
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    failed = [row for row in rows if row["status"] != "pass"]
    lines = [
        "# NCCL regime sweep summary",
        "",
        f"- Input root: {root}",
        f"- Parsed cases: {len(rows)}; failed or incomplete: {len(failed)}",
        "- busbw is the nccl-tests bus bandwidth; slowdown is relative to the "
        "same collective/size/warmup/iteration clean-before median.",
        "",
        "| Collective | Size | Warmup | Iters | Scenario | N | BusBW median (GB/s) | AlgBW median (GB/s) | Time median (ms) | Background slowdown | Clean-after recovery | Background (GB/s) |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for base_key in base_keys:
        clean = grouped.get((*base_key, "clean-before"), [])
        clean_bus = median(numeric(clean, "busbw_gbps"))
        for scenario in scenario_order(grouped, base_key):
            case_rows = grouped.get((*base_key, scenario), [])
            busbw = median(numeric(case_rows, "busbw_gbps"))
            algbw = median(numeric(case_rows, "algbw_gbps"))
            elapsed = median(numeric(case_rows, "time_ms"))
            background = median(numeric(case_rows, "background_aggregate_gbps"))
            slowdown = None
            recovery = None
            if clean_bus not in (None, 0) and busbw is not None:
                if scenario in BACKGROUND_SCENARIOS:
                    slowdown = (1.0 - busbw / clean_bus) * 100.0
                elif scenario == "clean-after":
                    recovery = busbw / clean_bus * 100.0
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(base_key[0]),
                        str(base_key[1]),
                        str(base_key[2]),
                        str(base_key[3]),
                        scenario,
                        str(len(case_rows)),
                        fmt(busbw),
                        fmt(algbw),
                        fmt(elapsed),
                        f"{fmt(slowdown)}%" if slowdown is not None else "n/a",
                        f"{fmt(recovery)}%" if recovery is not None else "n/a",
                        fmt(background),
                    ]
                )
                + " |"
            )

    if failed:
        lines.extend(["", "## Failed or incomplete cases", ""])
        for row in failed:
            lines.append(
                f"- {row['case_dir']}: {row.get('parse_note') or row['status']}"
            )
    else:
        lines.extend(["", "All discovered cases passed parsing and status checks."])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve()
    if not root.is_dir():
        parser.error(f"input root is not a directory: {root}")

    rows = []
    for log_path in sorted(root.rglob("nccl-tests.log")):
        row = parse_case(log_path, root)
        if row:
            rows.append(row)
    if not rows:
        parser.error(f"no recognized nccl-tests.log files under {root}")

    csv_path = (args.csv or root / "summary.csv").resolve()
    markdown_path = (args.markdown or root / "summary.md").resolve()
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(root, rows, csv_path.with_name("summary-by-regime.csv"))
    write_markdown(root, rows, markdown_path)
    print(f"parsed_cases={len(rows)} failed={sum(row['status'] != 'pass' for row in rows)}")
    print(f"case_csv={csv_path}")
    print(f"regime_csv={csv_path.with_name('summary-by-regime.csv')}")
    print(f"markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
