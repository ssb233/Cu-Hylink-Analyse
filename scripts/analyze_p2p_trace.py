#!/usr/bin/env python3
"""Read-only analysis of Nsight Systems CUDA peer-copy activity."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


P2P_COPY_KIND = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CUDA P2P memcpy rows in an Nsight Systems SQLite export."
    )
    parser.add_argument(
        "--sqlite", required=True, help="Nsight Systems SQLite export to read"
    )
    parser.add_argument(
        "--output", default="-", help="JSON output path, or - for stdout"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="warmup iterations per logical edge (default: 10)",
    )
    parser.add_argument(
        "--edges-per-source",
        type=int,
        default=3,
        help="minimum queue-position range to report (default: 3)",
    )
    parser.add_argument(
        "--slow-threshold-ms",
        type=float,
        default=8.0,
        help="duration strictly above this threshold is slow (default: 8.0)",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.edges_per_source <= 0:
        parser.error("--edges-per-source must be positive")
    if args.slow_threshold_ms <= 0:
        parser.error("--slow-threshold-ms must be positive")
    return args


def open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite file does not exist: {resolved}")
    uri = f"file:{quote(str(resolved))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = int((len(values) - 1) * quantile)
    return values[index]


def summarize_values(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": nearest_rank(ordered, 0.50),
        "p90": nearest_rank(ordered, 0.90),
        "p99": nearest_rank(ordered, 0.99),
        "max": ordered[-1] if ordered else 0.0,
    }


def analyze(args: argparse.Namespace) -> dict:
    path = Path(args.sqlite)
    with open_read_only(path) as connection:
        rows = connection.execute(
            """
            SELECT start, end, streamId, srcDeviceId, dstDeviceId, bytes, globalPid
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind = ?
            ORDER BY streamId, start, end
            """,
            (P2P_COPY_KIND,),
        ).fetchall()
        try:
            delay_kernel_rows = connection.execute(
                """
                SELECT k.start, k.end, k.deviceId, k.streamId, s.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN StringIds AS s ON s.id = k.demangledName
                WHERE lower(coalesce(s.value, '')) LIKE '%sourceoffsetdelay%'
                ORDER BY k.start
                """
            ).fetchall()
        except sqlite3.Error:
            delay_kernel_rows = []

    source_offset_delay_kernels = []
    for start, end, device, stream_id, kernel_name in delay_kernel_rows:
        if start is None or end is None:
            continue
        source_offset_delay_kernels.append(
            {
                "source": int(device) if device is not None else None,
                "streamId": int(stream_id) if stream_id is not None else None,
                "start": int(start),
                "end": int(end),
                "durationMs": (int(end) - int(start)) / 1_000_000.0,
                "kernelName": kernel_name,
            }
        )

    by_stream: dict[int, list[dict]] = defaultdict(list)
    for start, end, stream_id, source, destination, bytes_count, global_pid in rows:
        if start is None or end is None or stream_id is None:
            raise RuntimeError("P2P memcpy row is missing start, end, or streamId")
        if source is None or destination is None:
            raise RuntimeError("P2P memcpy row is missing source or destination GPU")
        duration_ns = int(end) - int(start)
        if duration_ns < 0:
            raise RuntimeError("P2P memcpy row has negative duration")
        by_stream[int(stream_id)].append(
            {
                "start": int(start),
                "end": int(end),
                "source": int(source),
                "destination": int(destination),
                "bytes": int(bytes_count) if bytes_count is not None else None,
                "globalPid": int(global_pid) if global_pid is not None else None,
                "durationMs": duration_ns / 1_000_000.0,
            }
        )

    annotated: list[dict] = []
    stream_details: list[dict] = []
    max_edge_count = 1
    for stream_id, stream_rows in sorted(by_stream.items()):
        edge_positions: dict[tuple[int, int], int] = {}
        for row in stream_rows:
            edge = (row["source"], row["destination"])
            if edge not in edge_positions:
                edge_positions[edge] = len(edge_positions)

        edge_count = len(edge_positions)
        for ordinal, row in enumerate(stream_rows):
            edge = (row["source"], row["destination"])
            row["streamId"] = stream_id
            row["ordinalInStream"] = ordinal
            row["queuePosition"] = edge_positions[edge]
            row["round"] = ordinal // edge_count
            row["isWarmup"] = row["round"] < args.warmup
            annotated.append(row)

        max_edge_count = max(max_edge_count, edge_count)
        stream_details.append(
            {
                "streamId": stream_id,
                "source": stream_rows[0]["source"] if stream_rows else None,
                "rows": len(stream_rows),
                "edgeCount": edge_count,
                "warmupRows": args.warmup * edge_count,
                "edges": [
                    {"source": source, "destination": destination, "queuePosition": position}
                    for (source, destination), position in sorted(
                        edge_positions.items(), key=lambda item: item[1]
                    )
                ],
            }
        )

    durations = sorted(row["durationMs"] for row in annotated)
    slow_rows = [row for row in annotated if row["durationMs"] > args.slow_threshold_ms]
    measured_rows = [row for row in annotated if not row["isWarmup"]]
    measured_slow_rows = [
        row for row in measured_rows if row["durationMs"] > args.slow_threshold_ms
    ]

    first_measured_start_by_source: dict[int, int] = {}
    for row in measured_rows:
        source = row["source"]
        first_measured_start_by_source[source] = min(
            first_measured_start_by_source.get(source, row["start"]),
            row["start"],
        )
    first_measured_starts = sorted(first_measured_start_by_source.items())
    first_measured_baseline = (
        min(start for _, start in first_measured_starts)
        if first_measured_starts
        else None
    )
    first_measured_source_starts = [
        {
            "source": source,
            "start": start,
            "relativeStartMs": (
                (start - first_measured_baseline) / 1_000_000.0
                if first_measured_baseline is not None
                else 0.0
            ),
        }
        for source, start in first_measured_starts
    ]

    rows_by_round: dict[int, list[dict]] = defaultdict(list)
    for row in annotated:
        rows_by_round[row["round"]].append(row)

    round_summaries: list[dict] = []
    slow_waves: list[dict] = []
    for round_index, round_rows in sorted(rows_by_round.items()):
        slow_round_rows = [
            row
            for row in round_rows
            if row["durationMs"] > args.slow_threshold_ms
        ]
        start_times = [row["start"] for row in round_rows]
        slow_start_times = [row["start"] for row in slow_round_rows]
        round_summary = {
            "round": round_index,
            "p2pCount": len(round_rows),
            "sourceGpuCount": len({row["source"] for row in round_rows}),
            "startSpanMs": (max(start_times) - min(start_times)) / 1_000_000.0,
            "slowCount": len(slow_round_rows),
            "slowSourceGpuCount": len(
                {row["source"] for row in slow_round_rows}
            ),
            "slowStartSpanMs": (
                (max(slow_start_times) - min(slow_start_times)) / 1_000_000.0
                if slow_start_times
                else 0.0
            ),
        }
        round_summaries.append(round_summary)
        if slow_round_rows:
            slow_waves.append(
                {
                    "round": round_index,
                    "slowCount": len(slow_round_rows),
                    "gpuCount": len({row["source"] for row in slow_round_rows}),
                    "sourceGpus": sorted({row["source"] for row in slow_round_rows}),
                    "startSpanMs": round_summary["slowStartSpanMs"],
                }
            )

    slow_wave_gpu_counts = [wave["gpuCount"] for wave in slow_waves]
    slow_wave_start_spans = [wave["startSpanMs"] for wave in slow_waves]

    max_position = max(args.edges_per_source - 1, max_edge_count - 1)
    slow_by_position = []
    for position in range(max_position + 1):
        count = sum(row["queuePosition"] == position for row in slow_rows)
        slow_by_position.append(
            {
                "position": position,
                "count": count,
                "fraction": count / len(slow_rows) if slow_rows else 0.0,
            }
        )

    edges = sorted({(row["source"], row["destination"]) for row in annotated})
    slow_by_edge = []
    for source, destination in edges:
        count = sum(
            row["source"] == source
            and row["destination"] == destination
            for row in slow_rows
        )
        slow_by_edge.append(
            {
                "source": source,
                "destination": destination,
                "count": count,
                "fraction": count / len(slow_rows) if slow_rows else 0.0,
            }
        )

    return {
        "input": str(path.expanduser().resolve()),
        "p2pCopyKind": P2P_COPY_KIND,
        "slowThresholdMs": args.slow_threshold_ms,
        "warmupIterations": args.warmup,
        "p2pCount": len(annotated),
        "streamCount": len(by_stream),
        "measuredP2pCount": len(measured_rows),
        "firstMeasuredStartBySource": first_measured_source_starts,
        "sourceOffsetDelayKernels": source_offset_delay_kernels,
        "slowCount": len(slow_rows),
        "slowCountScope": "allP2PRowsIncludingWarmup",
        "measuredSlowCount": len(measured_slow_rows),
        "durationMs": summarize_values(durations),
        "slowByQueuePosition": slow_by_position,
        "slowByEdge": slow_by_edge,
        "roundCount": len(rows_by_round),
        "p2pRows": annotated,
        "rounds": round_summaries,
        "slowWaveCount": len(slow_waves),
        "slowWaveGpuCountStats": summarize_values(slow_wave_gpu_counts),
        "slowWaveStartSpanMsStats": summarize_values(slow_wave_start_spans),
        "slowWaves": slow_waves,
        "streams": stream_details,
    }


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args)
    except (FileNotFoundError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
