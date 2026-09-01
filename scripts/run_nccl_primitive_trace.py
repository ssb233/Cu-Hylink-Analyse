#!/usr/bin/env python3
"""Run clean/concurrent NCCL cases and retain device primitive trace dumps.

The NCCL primitive-trace build reads ``NCCL_PRIMITIVE_TRACE_FILE`` at
communicator destruction and replaces ``%r`` with the communicator rank.  The
environment is changed between the three stages so each stage has its own
trace files and can be compared with the exact NCCL measured window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from run_nccl_p0_overlap import (
    COLLECTIVE_BINARIES,
    SCENARIOS,
    run_treatment,
    run_victim,
    parse_csv,
)


# nccl-tests runs the out-of-place and in-place collective for each benchmark
# iteration.  The device channel workCounter therefore advances by up to two
# work units per requested iteration on this Stage 9 path.
NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER = 2


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] + weight * (ordered[right] - ordered[left])


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def trace_summary(case_dir: Path, label: str) -> Dict[str, Any]:
    files = sorted(
        path
        for path in case_dir.glob(f"{label}.*")
        if path.is_file() and path.name[len(label) + 1 :].isdigit()
    )
    records: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("type") == "meta":
                meta.append(value)
            elif value.get("type") == "record":
                records.append(value)
            elif value.get("type") == "overflow":
                overflow.append(value)
    durations = [float(record["durationNs"]) for record in records]
    by_primitive: Dict[str, int] = {}
    per_work: Dict[str, float] = {}
    for record in records:
        key = str(record.get("primitive"))
        by_primitive[key] = by_primitive.get(key, 0) + 1
        work_key = ":".join(
            str(record.get(field))
            for field in ("channel", "group", "role", "primitive", "sequence")
        )
        per_work[work_key] = per_work.get(work_key, 0.0) + float(record["durationNs"])
    work_durations = list(per_work.values())
    dropped_records = sum(int(item.get("droppedRecords", 0)) for item in overflow)
    event_counts = [int(record.get("eventCount", 0)) for record in records]
    storage = sorted({str(item.get("storage")) for item in meta if item.get("storage") is not None})
    return {
        "label": label,
        "files": [str(path) for path in files],
        "fileCount": len(files),
        "metaCount": len(meta),
        "recordCount": len(records),
        "overflowCount": len(overflow),
        "droppedRecords": dropped_records,
        "samplePeriod": meta[0].get("samplePeriod") if meta else None,
        "storage": storage[0] if len(storage) == 1 else storage,
        "maxSampledWorks": meta[0].get("maxSampledWorks") if meta else None,
        "eventsPerRole": meta[0].get("eventsPerRole") if meta else None,
        "recordsPerChannel": meta[0].get("recordsPerChannel") if meta else None,
        "eventCount": summarize([float(value) for value in event_counts]),
        "recordsByPrimitive": by_primitive,
        "perEventDurationNs": summarize(durations),
        "perWorkCount": len(work_durations),
        "perWorkDurationNs": summarize(work_durations),
    }


def parse_positive_list(value: str, name: str) -> List[int]:
    result: List[int] = []
    for item in parse_csv(value):
        try:
            parsed = int(item)
        except ValueError as error:
            raise ValueError(f"{name} must contain positive integers") from error
        if parsed <= 0:
            raise ValueError(f"{name} must contain positive integers")
        result.append(parsed)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repoRoot", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--collectives", default="allgather")
    parser.add_argument("--sizes", default="64M")
    parser.add_argument("--scenarios", default="disjoint")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--victimIterations", type=int, default=5000)
    parser.add_argument("--victimWarmup", type=int, default=100)
    parser.add_argument("--relaySize", default="255M")
    parser.add_argument("--relayWarmup", type=int, default=20)
    parser.add_argument("--relayReportMs", type=int, default=100)
    parser.add_argument("--settleMs", type=int, default=250)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--relayCpus", default="0,2,4,6")
    parser.add_argument("--victimCpus", default="8,10")
    parser.add_argument("--randomSeed", type=int, default=20260831)
    parser.add_argument("--timeoutSec", type=float, default=180.0)
    parser.add_argument("--samplePeriods", default="512,256,128")
    parser.add_argument("--primitiveTraceKind", default="post")
    parser.add_argument("--runtimeDisabled", action="store_true")
    parser.add_argument("--maxSampledWorks", type=int, default=128)
    parser.add_argument("--ncclBuild", type=Path, required=True)
    parser.add_argument("--testsBuild", type=Path)
    parser.add_argument("--relayBin", type=Path)
    parser.add_argument("--outputRoot", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo_root = args.repoRoot.resolve()
    nccl_build = args.ncclBuild.resolve()
    tests_build = (args.testsBuild or repo_root / "build/nccl-tests-p0-sm70").resolve()
    relay_bin = (args.relayBin or repo_root / "build/cuda_copy/host_relay").resolve()
    output_root = args.outputRoot.resolve()
    collectives = parse_csv(args.collectives)
    sizes = parse_csv(args.sizes)
    scenario_names = parse_csv(args.scenarios)
    sample_periods = [0] if args.runtimeDisabled else parse_positive_list(args.samplePeriods, "samplePeriods")

    if args.primitiveTraceKind not in {"wait", "fence", "store", "post", "copy"}:
        raise ValueError("primitiveTraceKind must be one of wait, fence, store, post, copy")
    if any(item not in COLLECTIVE_BINARIES for item in collectives):
        raise ValueError(f"unknown collective; choose from {sorted(COLLECTIVE_BINARIES)}")
    if any(item not in SCENARIOS for item in scenario_names):
        raise ValueError(f"unknown scenario; choose from {sorted(SCENARIOS)}")
    if args.repetitions <= 0 or args.victimIterations <= 0 or args.victimWarmup < 0:
        raise ValueError("repetitions/victimIterations must be positive and victimWarmup non-negative")
    if args.relayWarmup < 0 or args.relayReportMs <= 0 or args.settleMs < 0:
        raise ValueError("relayWarmup must be non-negative; relayReportMs must be positive; settleMs non-negative")
    if args.maxSampledWorks <= 0:
        raise ValueError("maxSampledWorks must be positive")
    if not args.runtimeDisabled:
        work_counter_upper_bound = (
            args.victimWarmup + args.victimIterations
        ) * NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER
        for period in sample_periods:
            sampled_works = work_counter_upper_bound // period + 1
            if sampled_works > args.maxSampledWorks:
                raise ValueError(
                    f"samplePeriod={period} needs {sampled_works} sampled works, "
                    f"exceeding maxSampledWorks={args.maxSampledWorks}"
                )
    if not relay_bin.is_file():
        raise FileNotFoundError(relay_bin)
    if not (nccl_build / "lib/libnccl.so.2.31.2").is_file():
        raise FileNotFoundError(nccl_build / "lib/libnccl.so.2.31.2")
    for collective in collectives:
        if not (tests_build / COLLECTIVE_BINARIES[collective]).is_file():
            raise FileNotFoundError(tests_build / COLLECTIVE_BINARIES[collective])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "schemaVersion": 1,
        "clock": "steady_clock/time.monotonic_ns; device globaltimer for primitive records",
        "createdAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "collectives": collectives,
            "sizes": sizes,
            "scenarios": scenario_names,
            "repetitions": args.repetitions,
            "victimIterations": args.victimIterations,
            "victimWarmup": args.victimWarmup,
            "relaySize": args.relaySize,
            "relayWarmup": args.relayWarmup,
            "relayReportMs": args.relayReportMs,
            "settleMs": args.settleMs,
            "devices": args.devices,
            "relayCpus": args.relayCpus,
            "victimCpus": args.victimCpus,
            "randomSeed": args.randomSeed,
            "samplePeriods": sample_periods,
            "primitiveTraceKind": args.primitiveTraceKind,
            "runtimeDisabled": args.runtimeDisabled,
            "maxSampledWorks": args.maxSampledWorks,
            "capacityProof": {
                "victimWorkItemsUpperBound": args.victimWarmup + args.victimIterations,
                "workCounterMultiplier": NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER,
                "workCounterUpperBound": (
                    args.victimWarmup + args.victimIterations
                ) * NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER,
                "sampledWorksUpperBound": {
                    str(period): (
                        (
                            (
                                args.victimWarmup + args.victimIterations
                            )
                            * NCCL_PRIMITIVE_TRACE_WORK_COUNTER_MULTIPLIER
                        )
                        // period
                        + 1
                    )
                    for period in sample_periods
                    if period > 0
                },
            },
            "ncclBuild": str(nccl_build),
            "testsBuild": str(tests_build),
            "relayBin": str(relay_bin),
        },
        "cases": [],
    }

    case_specs = [
        (period, collective, size, repetition, scenario_name)
        for period in sample_periods
        for collective in collectives
        for size in sizes
        for repetition in range(1, args.repetitions + 1)
        for scenario_name in scenario_names
    ]
    rng = random.Random(args.randomSeed)
    rng.shuffle(case_specs)
    manifest["caseOrder"] = [
        {
            "samplePeriod": period,
            "collective": collective,
            "size": size,
            "repetition": repetition,
            "scenario": scenario_name,
        }
        for period, collective, size, repetition, scenario_name in case_specs
    ]

    old_trace_file = os.environ.get("NCCL_PRIMITIVE_TRACE_FILE")
    old_sample_period = os.environ.get("NCCL_PRIMITIVE_TRACE_SAMPLE_PERIOD")
    try:
        for period, collective, size, repetition, scenario_name in case_specs:
                            scenario = SCENARIOS[scenario_name]
                            case_dir = (
                                output_root
                                / args.primitiveTraceKind
                                / f"sample-{period}"
                                / collective
                                / size
                                / f"rep-{repetition}"
                                / scenario_name
                            )
                            case_dir.mkdir(parents=True, exist_ok=True)
                            case: Dict[str, Any] = {
                                "collective": collective,
                                "size": size,
                                "repetition": repetition,
                                "scenario": scenario_name,
                                "primitiveTraceKind": args.primitiveTraceKind,
                                "samplePeriod": period,
                                "traceEnabled": not args.runtimeDisabled,
                                "caseDir": str(case_dir),
                            }

                            print(
                                f"[{args.primitiveTraceKind} N={period} {collective} {size} "
                                f"rep-{repetition} {scenario_name}] clean-before -> treatment -> clean-after",
                                flush=True,
                            )

                            if args.runtimeDisabled:
                                os.environ.pop("NCCL_PRIMITIVE_TRACE_SAMPLE_PERIOD", None)
                                os.environ.pop("NCCL_PRIMITIVE_TRACE_FILE", None)
                            else:
                                os.environ["NCCL_PRIMITIVE_TRACE_SAMPLE_PERIOD"] = str(period)
                                os.environ["NCCL_PRIMITIVE_TRACE_FILE"] = str(case_dir / "clean-before.%r")
                            before = run_victim(
                                case_dir=case_dir,
                                tests_build=tests_build,
                                nccl_build=nccl_build,
                                collective=collective,
                                size=size,
                                devices=args.devices,
                                warmup=args.victimWarmup,
                                iterations=args.victimIterations,
                                victim_cpus=args.victimCpus,
                                label="clean-before",
                            )
                            before["primitiveTrace"] = trace_summary(case_dir, "clean-before")

                            if not args.runtimeDisabled:
                                os.environ["NCCL_PRIMITIVE_TRACE_FILE"] = str(case_dir / "concurrent.%r")
                            treatment = run_treatment(
                                case_dir=case_dir,
                                relay_bin=relay_bin,
                                tests_build=tests_build,
                                nccl_build=nccl_build,
                                scenario=scenario,
                                collective=collective,
                                victim_size=size,
                                relay_size=args.relaySize,
                                devices=args.devices,
                                victim_warmup=args.victimWarmup,
                                victim_iterations=args.victimIterations,
                                relay_warmup=args.relayWarmup,
                                relay_report_ms=args.relayReportMs,
                                relay_cpus=args.relayCpus,
                                victim_cpus=args.victimCpus,
                                settle_ms=args.settleMs,
                                timeout_sec=args.timeoutSec,
                            )
                            treatment["primitiveTrace"] = trace_summary(case_dir, "concurrent")

                            if not args.runtimeDisabled:
                                os.environ["NCCL_PRIMITIVE_TRACE_FILE"] = str(case_dir / "clean-after.%r")
                            after = run_victim(
                                case_dir=case_dir,
                                tests_build=tests_build,
                                nccl_build=nccl_build,
                                collective=collective,
                                size=size,
                                devices=args.devices,
                                warmup=args.victimWarmup,
                                iterations=args.victimIterations,
                                victim_cpus=args.victimCpus,
                                label="clean-after",
                            )
                            after["primitiveTrace"] = trace_summary(case_dir, "clean-after")

                            case.update(
                                {
                                    "cleanBefore": before,
                                    "treatment": treatment,
                                    "cleanAfter": after,
                                }
                            )
                            (case_dir / "case.json").write_text(
                                json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                            )
                            manifest["cases"].append(case)
                            (output_root / "manifest.json").write_text(
                                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                            )
    finally:
        if old_trace_file is None:
            os.environ.pop("NCCL_PRIMITIVE_TRACE_FILE", None)
        else:
            os.environ["NCCL_PRIMITIVE_TRACE_FILE"] = old_trace_file
        if old_sample_period is None:
            os.environ.pop("NCCL_PRIMITIVE_TRACE_SAMPLE_PERIOD", None)
        else:
            os.environ["NCCL_PRIMITIVE_TRACE_SAMPLE_PERIOD"] = old_sample_period

    print(f"Primitive trace results: {output_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
