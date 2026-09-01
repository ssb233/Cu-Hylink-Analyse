#!/usr/bin/env python3
"""Analyze Stage I directional DMA measurements.

The runner records one clean row for each background-size/topology/edge-order/
repetition tuple. Treatment rows are paired with that clean row before the
group medians are computed, so a change in victim and background payload cannot
be hidden in a single overloaded size column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SIZE_RE = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMG](?:I?[Bb])?)?$", re.IGNORECASE)
REQUIRED_COLUMNS = {
    "backgroundSize",
    "victimSize",
    "direction",
    "dutyCycle",
    "backgroundSet",
    "edgeOrder",
    "topology",
    "repetition",
    "victimAggregateGBps",
    "backgroundAggregateGBps",
    "status",
}


def parse_size(value: str) -> float:
    match = SIZE_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid size: {value}")
    multiplier = {"K": 1024.0, "M": 1024.0**2, "G": 1024.0**3}
    unit = match.group("unit")
    suffix = unit[0].upper() if unit else ""
    return float(match.group("value")) * multiplier.get(suffix, 1.0)


def float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() in {"", "NA", "null", "None"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.mean(materialized) if materialized else None


def pstdev_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.pstdev(materialized) if len(materialized) > 1 else 0.0 if materialized else None


def json_path(root: Path, value: str | None) -> Path | None:
    if value is None or value.strip() in {"", "NA", "null", "None"}:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def nested_p50(data: dict[str, Any], field: str) -> list[float]:
    result: list[float] = []
    for item in data.get(field, []):
        if isinstance(item, dict):
            value = float_or_none(str(item.get("p50", "")))
        else:
            value = float_or_none(str(item))
        if value is not None:
            result.append(value)
    return result


def load_background_metrics(root: Path, value: str | None) -> dict[str, float | None]:
    path = json_path(root, value)
    empty = {
        "background_per_device_gbps_median": None,
        "background_duration_p50_ms": None,
        "background_submit_interval_p50_ms": None,
        "background_idle_gap_p50_ms": None,
        "background_wall_active_duty_median": None,
    }
    if path is None or not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    per_device_gbps = [
        value
        for value in (float_or_none(str(item)) for item in data.get("perDeviceGBps", []))
        if value is not None
    ]
    metrics = {
        "background_per_device_gbps_median": median(per_device_gbps),
        "background_duration_p50_ms": median(nested_p50(data, "perDeviceOperationDurationMs")),
        "background_submit_interval_p50_ms": median(nested_p50(data, "perDeviceSubmitIntervalMs")),
        "background_idle_gap_p50_ms": median(nested_p50(data, "perDeviceIdleGapMs")),
        "background_wall_active_duty_median": median(
            value
            for value in (
                float_or_none(str(item)) for item in data.get("perDeviceWallActiveDuty", [])
            )
            if value is not None
        ),
    }
    return metrics


def clean_key(row: dict[str, str], include_repetition: bool = True) -> tuple[str, ...]:
    fields = ["backgroundSize", "victimSize", "edgeOrder", "topology"]
    if include_repetition:
        fields.append("repetition")
    return tuple(row.get(field, "") for field in fields)


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row.get(field, "")
        for field in (
            "backgroundSize",
            "victimSize",
            "direction",
            "dutyCycle",
            "backgroundSet",
            "edgeOrder",
            "topology",
        )
    )


def normalized_duty(value: str) -> float | None:
    return float_or_none(value) if value != "NA" else None


def is_pass(row: dict[str, str]) -> bool:
    return row.get("status", "").lower() == "pass" and float_or_none(row.get("victimAggregateGBps")) is not None


def analyze(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"summary.csv is missing columns: {', '.join(missing)}")
        rows = list(reader)

    parsed_rows = len(rows)
    failed_rows = sum(1 for row in rows if not is_pass(row))
    treatment_rows = sum(1 for row in rows if row.get("backgroundSet") != "none")

    clean_by_exact: dict[tuple[str, ...], list[float]] = defaultdict(list)
    clean_by_base: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get("victimAggregateGBps"))
        if row.get("backgroundSet") == "none" and is_pass(row) and value is not None:
            clean_by_exact[clean_key(row)].append(value)
            clean_by_base[clean_key(row, include_repetition=False)].append(value)

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    unmatched: list[dict[str, str]] = []
    for row in rows:
        if row.get("backgroundSet") == "none" or not is_pass(row):
            continue
        treatment = float_or_none(row.get("victimAggregateGBps"))
        if treatment is None:
            continue
        clean_values = clean_by_exact.get(clean_key(row))
        if not clean_values:
            clean_values = clean_by_base.get(clean_key(row, include_repetition=False))
        if not clean_values:
            unmatched.append(row)
            continue
        clean_value = statistics.median(clean_values)
        drop_pct = (1.0 - treatment / clean_value) * 100.0 if clean_value else None
        record: dict[str, Any] = {
            "backgroundSize": row.get("backgroundSize", ""),
            "victimSize": row.get("victimSize", ""),
            "direction": row.get("direction", ""),
            "dutyCycle": row.get("dutyCycle", ""),
            "backgroundSet": row.get("backgroundSet", ""),
            "edgeOrder": row.get("edgeOrder", ""),
            "topology": row.get("topology", ""),
            "repetition": row.get("repetition", ""),
            "clean_gbps": clean_value,
            "treatment_gbps": treatment,
            "paired_drop_pct": drop_pct,
            "background_gbps": float_or_none(row.get("backgroundAggregateGBps")),
        }
        record.update(load_background_metrics(root, row.get("backgroundJson")))
        grouped[group_key(row)].append(record)

    group_results: list[dict[str, Any]] = []
    for key, records in grouped.items():
        drops = [record["paired_drop_pct"] for record in records if record["paired_drop_pct"] is not None]
        clean_values = [record["clean_gbps"] for record in records]
        treatment_values = [record["treatment_gbps"] for record in records]

        def metric(name: str) -> list[float]:
            return [record[name] for record in records if record.get(name) is not None]

        result = {
            "backgroundSize": key[0],
            "victimSize": key[1],
            "direction": key[2],
            "dutyCycle": key[3],
            "backgroundSet": key[4],
            "edgeOrder": key[5],
            "topology": key[6],
            "n": len(records),
            "repetitions": sorted(record["repetition"] for record in records),
            "clean_gbps_median": median(clean_values),
            "treatment_gbps_median": median(treatment_values),
            "paired_drop_pct_median": median(drops),
            "paired_drop_pct_mean": mean_or_none(drops),
            "paired_drop_pct_pstdev": pstdev_or_none(drops),
            "background_gbps_median": median(metric("background_gbps")),
            "background_per_device_gbps_median": median(metric("background_per_device_gbps_median")),
            "background_duration_p50_ms": median(metric("background_duration_p50_ms")),
            "background_submit_interval_p50_ms": median(metric("background_submit_interval_p50_ms")),
            "background_idle_gap_p50_ms": median(metric("background_idle_gap_p50_ms")),
            "background_wall_active_duty_median": median(metric("background_wall_active_duty_median")),
            "paired_drop_pct_values": drops,
        }
        group_results.append(result)

    group_results.sort(
        key=lambda result: (
            result["direction"],
            result["backgroundSet"],
            result["topology"],
            result["edgeOrder"],
            parse_size(result["backgroundSize"]),
        )
    )

    threshold_candidates: list[dict[str, Any]] = []
    series: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in group_results:
        if (
            result["direction"] == "d2h"
            and result["backgroundSet"] == "0"
            and normalized_duty(result["dutyCycle"]) == 1.0
        ):
            series[
                (
                    result["victimSize"],
                    result["edgeOrder"],
                    result["topology"],
                )
            ].append(result)

    for series_key, values in series.items():
        values.sort(key=lambda result: parse_size(result["backgroundSize"]))
        for first, second in zip(values, values[1:]):
            first_drops = first["paired_drop_pct_values"]
            second_drops = second["paired_drop_pct_values"]
            if (
                len(first_drops) >= 3
                and len(second_drops) >= 3
                and all(drop > 5.0 for drop in first_drops)
                and all(drop > 5.0 for drop in second_drops)
            ):
                immune = [
                    candidate
                    for candidate in values
                    if parse_size(candidate["backgroundSize"]) < parse_size(first["backgroundSize"])
                    and len(candidate["paired_drop_pct_values"]) >= 3
                    and all(drop <= 5.0 for drop in candidate["paired_drop_pct_values"])
                ]
                threshold_candidates.append(
                    {
                        "victimSize": series_key[0],
                        "edgeOrder": series_key[1],
                        "topology": series_key[2],
                        "backgroundSet": "0",
                        "direction": "d2h",
                        "dutyCycle": "1.0",
                        "last_immune_size": immune[-1]["backgroundSize"] if immune else None,
                        "first_drop_size": first["backgroundSize"],
                        "next_drop_size": second["backgroundSize"],
                        "first_drop_pct": first["paired_drop_pct_median"],
                        "next_drop_pct": second["paired_drop_pct_median"],
                    }
                )
                # A later adjacent pair is part of the already-affected
                # regime, not a new onset threshold for this series.
                break

    return {
        "input_root": str(root),
        "parsed_rows": parsed_rows,
        "failed_rows": failed_rows,
        "treatment_rows": treatment_rows,
        "valid_treatment_rows": sum(len(records) for records in grouped.values()),
        "group_count": len(group_results),
        "unmatched_treatment_rows": len(unmatched),
        "threshold_candidates": threshold_candidates,
        "groups": group_results,
    }


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(root: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Stage I directional DMA diagnostic summary",
        "",
        f"- Input root: `{data['input_root']}`",
        f"- Parsed rows: {data['parsed_rows']}; failed/incomplete: {data['failed_rows']}",
        f"- Treatment rows: {data['treatment_rows']}; paired valid rows: {data['valid_treatment_rows']}",
        f"- Grouped points: {data['group_count']}; unmatched treatment rows: {data['unmatched_treatment_rows']}",
        "- `paired drop` is `1 - treatment victim GB/s / paired clean victim GB/s`.",
        "",
        "## Grouped points",
        "",
        "| Bg size | Victim | Direction | Duty | Bg set | Edge | Topology | N | Clean GB/s | Treatment GB/s | Drop | Bg GB/s | Duration p50 ms | Submit p50 ms | Idle p50 ms | Wall duty |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in data["groups"]:
        lines.append(
            "| {backgroundSize} | {victimSize} | {direction} | {dutyCycle} | {backgroundSet} | {edgeOrder} | {topology} | {n} | {clean} | {treatment} | {drop}% | {bg} | {duration} | {submit} | {idle} | {wall} |".format(
                backgroundSize=result["backgroundSize"],
                victimSize=result["victimSize"],
                direction=result["direction"],
                dutyCycle=result["dutyCycle"],
                backgroundSet=result["backgroundSet"],
                edgeOrder=result["edgeOrder"],
                topology=result["topology"],
                n=result["n"],
                clean=format_number(result["clean_gbps_median"]),
                treatment=format_number(result["treatment_gbps_median"]),
                drop=format_number(result["paired_drop_pct_median"]),
                bg=format_number(result["background_gbps_median"]),
                duration=format_number(result["background_duration_p50_ms"]),
                submit=format_number(result["background_submit_interval_p50_ms"]),
                idle=format_number(result["background_idle_gap_p50_ms"]),
                wall=format_number(result["background_wall_active_duty_median"]),
            )
        )

    lines.extend(["", "## I1 threshold candidates", ""])
    if data["threshold_candidates"]:
        lines.extend(
            [
                "| Topology | Edge | Last immune | First drop | Next drop | First drop | Next drop |",
                "|---|---|---|---|---|---:|---:|",
            ]
        )
        for candidate in data["threshold_candidates"]:
            lines.append(
                "| {topology} | {edgeOrder} | {last_immune_size} | {first_drop_size} | {next_drop_size} | {first_drop_pct:.2f}% | {next_drop_pct:.2f}% |".format(
                    **{key: ("n/a" if value is None else value) for key, value in candidate.items()}
                )
            )
    else:
        lines.append("No candidate met the definition: two consecutive sizes, each with at least three repetitions and every repetition above 5% drop.")

    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Stage I result root containing summary.csv")
    args = parser.parse_args()
    root = args.root.resolve()
    data = analyze(root)
    (root / "analysis.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(root, data)
    print(
        f"parsed_rows={data['parsed_rows']} failed={data['failed_rows']} "
        f"groups={data['group_count']} threshold_candidates={len(data['threshold_candidates'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
