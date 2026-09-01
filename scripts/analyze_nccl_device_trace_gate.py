#!/usr/bin/env python3
"""Analyze the Stage 9 NCCL device-only primitive-trace gate.

The analyzer compares release trace-off, instrumented runtime-disabled, and
sampled trace roots by repetition.  It also validates the correctness/window/
relay invariants before reporting the observer-effect thresholds from the
experiment plan.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BaseKey = Tuple[str, str, int, str]
FullKey = Tuple[int, str, str, int, str]


T95_BY_N = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
}


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def finite(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def case_key(case: Dict[str, Any]) -> FullKey:
    return (
        int(case["samplePeriod"]),
        str(case["collective"]),
        str(case["size"]),
        int(case["repetition"]),
        str(case["scenario"]),
    )


def base_key(case: Dict[str, Any]) -> BaseKey:
    _, collective, size, repetition, scenario = case_key(case)
    return (collective, size, repetition, scenario)


def load_root(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    cases: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("case.json")):
        value = load_json(path)
        if isinstance(value, dict) and "samplePeriod" in value:
            cases.append(value)
    if not cases:
        manifest_cases = manifest.get("cases")
        if isinstance(manifest_cases, list):
            cases = [value for value in manifest_cases if isinstance(value, dict)]
    index: Dict[FullKey, Dict[str, Any]] = {}
    duplicate_keys: List[str] = []
    for case in cases:
        key = case_key(case)
        if key in index:
            duplicate_keys.append(str(key))
        index[key] = case
    return {
        "root": str(root),
        "manifest": manifest,
        "cases": cases,
        "index": index,
        "duplicateKeys": duplicate_keys,
    }


def find_case(root_data: Dict[str, Any], base: BaseKey, period: int) -> Optional[Dict[str, Any]]:
    index: Dict[FullKey, Dict[str, Any]] = root_data["index"]
    collective, size, repetition, scenario = base
    exact = index.get((period, collective, size, repetition, scenario))
    if exact is not None:
        return exact
    candidates = [
        case
        for (candidate_period, candidate_collective, candidate_size, candidate_rep, candidate_scenario), case in index.items()
        if candidate_collective == collective
        and candidate_size == size
        and candidate_rep == repetition
        and candidate_scenario == scenario
    ]
    return candidates[0] if len(candidates) == 1 else None


def t95(n: int) -> float:
    if n < 2:
        return math.inf
    return T95_BY_N.get(n, 1.96 if n >= 30 else 2.0)


def confidence_interval(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "lower95": None,
            "upper95": None,
            "includesZero": False,
        }
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    half_width = t95(len(values)) * sd / math.sqrt(len(values)) if sd is not None else None
    lower = mean - half_width if half_width is not None else None
    upper = mean + half_width if half_width is not None else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "lower95": lower,
        "upper95": upper,
        "includesZero": lower is not None and lower <= 0.0 <= upper,
    }


def measurement(case: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    value = case.get(label)
    if not isinstance(value, dict):
        return None
    result = value.get("measurement")
    return result if isinstance(result, dict) else None


def busbw(case: Dict[str, Any], label: str) -> Optional[float]:
    result = measurement(case, label)
    return finite(result.get("busbwGBps")) if result else None


def slowdown(case: Dict[str, Any]) -> Optional[float]:
    clean = busbw(case, "cleanBefore")
    concurrent = busbw(case, "treatment")
    if clean in (None, 0.0) or concurrent is None:
        return None
    return (1.0 - concurrent / clean) * 100.0


def stage_labels() -> Tuple[str, str, str]:
    return ("cleanBefore", "treatment", "cleanAfter")


def validate_case(case: Dict[str, Any], trace_required: bool) -> List[str]:
    failures: List[str] = []
    path = str(case.get("caseDir", "<unknown-case>"))
    for label in stage_labels():
        value = case.get(label)
        if not isinstance(value, dict):
            failures.append(f"{path}:{label}:missing-stage")
            continue
        if label == "treatment":
            if value.get("victimReturnCode") != 0:
                failures.append(f"{path}:{label}:victimReturnCode={value.get('victimReturnCode')}")
            if value.get("relayReturnCode") != 0:
                failures.append(f"{path}:{label}:relayReturnCode={value.get('relayReturnCode')}")
            if not isinstance(value.get("telemetryRecordCount"), int) or value["telemetryRecordCount"] <= 0:
                failures.append(f"{path}:{label}:incomplete-telemetry-records")
            overlap = value.get("relayOverlap")
            if not isinstance(overlap, dict) or overlap.get("coveredByTelemetry") is not True:
                failures.append(f"{path}:{label}:telemetry-not-covering-window")
        elif value.get("returnCode") != 0:
            failures.append(f"{path}:{label}:returnCode={value.get('returnCode')}")

        result = measurement(case, label)
        if result is None or finite(result.get("busbwGBps")) is None:
            failures.append(f"{path}:{label}:missing-or-nonfinite-measurement")
        elif result.get("wrong") != 0:
            failures.append(f"{path}:{label}:wrong={result.get('wrong')}")
        window = value.get("window")
        if not isinstance(window, dict):
            failures.append(f"{path}:{label}:missing-window")
        else:
            if window.get("markerCountBegin") != 1:
                failures.append(f"{path}:{label}:markerCountBegin={window.get('markerCountBegin')}")
            if window.get("markerCountEnd") != 1:
                failures.append(f"{path}:{label}:markerCountEnd={window.get('markerCountEnd')}")

        if trace_required:
            trace = value.get("primitiveTrace")
            if not isinstance(trace, dict):
                failures.append(f"{path}:{label}:missing-primitive-trace")
                continue
            if trace.get("storage") != "device":
                failures.append(f"{path}:{label}:storage={trace.get('storage')}")
            if trace.get("overflowCount") != 0:
                failures.append(f"{path}:{label}:overflowCount={trace.get('overflowCount')}")
            if trace.get("droppedRecords") != 0:
                failures.append(f"{path}:{label}:droppedRecords={trace.get('droppedRecords')}")
            if not isinstance(trace.get("recordCount"), int) or trace["recordCount"] <= 0:
                failures.append(f"{path}:{label}:empty-primitive-trace")
    return failures


def validate_root(name: str, root_data: Dict[str, Any], trace_required: bool) -> Dict[str, Any]:
    failures: List[str] = []
    failures.extend(f"{name}:duplicate-key:{key}" for key in root_data["duplicateKeys"])
    for case in root_data["cases"]:
        failures.extend(validate_case(case, trace_required))
    return {
        "name": name,
        "root": root_data["root"],
        "caseCount": len(root_data["cases"]),
        "baseCount": len({base_key(case) for case in root_data["cases"]}),
        "periods": sorted({case_key(case)[0] for case in root_data["cases"]}),
        "pass": not failures,
        "failures": failures,
    }


def paired_values(
    left_root: Dict[str, Any],
    right_root: Dict[str, Any],
    bases: Iterable[BaseKey],
    left_period: int,
    right_period: int,
    value_fn,
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    missing: List[str] = []
    for base in sorted(bases):
        left = find_case(left_root, base, left_period)
        right = find_case(right_root, base, right_period)
        if left is None or right is None:
            missing.append(str(base))
            continue
        left_value = value_fn(left)
        right_value = value_fn(right)
        if left_value is None or right_value is None:
            missing.append(str(base))
            continue
        values.append(float(value_fn(left, right)))
    return values, missing


def relative_clean(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[float]:
    left_value = busbw(left, "cleanBefore")
    right_value = busbw(right, "cleanBefore")
    if left_value is None or right_value in (None, 0.0):
        return None
    return (left_value / right_value - 1.0) * 100.0


def slowdown_difference(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[float]:
    left_value = slowdown(left)
    right_value = slowdown(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def compare_metric(
    name: str,
    left_root: Dict[str, Any],
    right_root: Dict[str, Any],
    bases: Iterable[BaseKey],
    left_period: int,
    right_period: int,
    value_fn,
) -> Dict[str, Any]:
    values: List[float] = []
    missing: List[str] = []
    for base in sorted(set(bases)):
        left = find_case(left_root, base, left_period)
        right = find_case(right_root, base, right_period)
        if left is None or right is None:
            missing.append(str(base))
            continue
        value = value_fn(left, right)
        if value is None or not math.isfinite(float(value)):
            missing.append(str(base))
            continue
        values.append(float(value))
    ci = confidence_interval(values)
    ci.update({"name": name, "missing": missing})
    return ci


def read_trace_durations(root_data: Dict[str, Any], periods: Sequence[int]) -> Dict[str, Any]:
    durations: List[float] = []
    bad_event_counts = 0
    missing_files: List[str] = []
    for case in root_data["cases"]:
        if case_key(case)[0] not in periods:
            continue
        for label in stage_labels():
            trace = case.get(label, {}).get("primitiveTrace", {})
            files = trace.get("files", []) if isinstance(trace, dict) else []
            for file_name in files:
                path = Path(str(file_name))
                if not path.is_file():
                    missing_files.append(str(path))
                    continue
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if value.get("type") != "record":
                        continue
                    event_count = value.get("eventCount")
                    if event_count != 1:
                        bad_event_counts += 1
                    duration = finite(value.get("durationNs"))
                    if duration is not None:
                        durations.append(duration)
    ordered = sorted(durations)
    return {
        "periods": list(periods),
        "eventCount": len(durations),
        "badEventCount": bad_event_counts,
        "missingFiles": missing_files,
        "p50Ns": percentile(ordered, 0.50),
        "p99Ns": percentile(ordered, 0.99),
    }


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def event_stability(root_data: Dict[str, Any]) -> Dict[str, Any]:
    n128 = read_trace_durations(root_data, [128])
    n256 = read_trace_durations(root_data, [256])
    checks: Dict[str, Any] = {"n128": n128, "n256": n256}
    for label, limit, field in (
        ("p50", 0.10, "p50Ns"),
        ("p99", 0.20, "p99Ns"),
    ):
        left = n128[field]
        right = n256[field]
        if left is None or right is None:
            relative = None
            passed = False
        elif right == 0:
            relative = 0.0 if left == 0 else math.inf
            passed = left == 0
        else:
            relative = abs(left - right) / abs(right)
            passed = relative <= limit
        checks[label] = {
            "relativeDifference": relative,
            "limit": limit,
            "pass": passed,
        }
    checks["pass"] = (
        checks["p50"]["pass"]
        and checks["p99"]["pass"]
        and n128["badEventCount"] == 0
        and n256["badEventCount"] == 0
        and not n128["missingFiles"]
        and not n256["missingFiles"]
    )
    return checks


def parse_trace_root(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"trace root must be KIND=PATH: {spec}")
    kind, path = spec.split("=", 1)
    if not kind or not path:
        raise ValueError(f"trace root must be KIND=PATH: {spec}")
    return kind, Path(path)


def check_result(name: str, passed: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), **details}


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    release = load_root(args.releaseRoot)
    runtime = load_root(args.runtimeDisabledRoot)
    traces = {kind: load_root(path) for kind, path in (parse_trace_root(spec) for spec in args.traceRoot)}
    checks: List[Dict[str, Any]] = []

    release_validation = validate_root("release", release, trace_required=False)
    runtime_validation = validate_root("runtime-disabled", runtime, trace_required=False)
    checks.append(check_result("release correctness/window/telemetry", release_validation["pass"], release_validation))
    checks.append(check_result("runtime-disabled correctness/window/telemetry", runtime_validation["pass"], runtime_validation))
    expected_bases = {base_key(case) for case in release["cases"]}

    for kind, root_data in traces.items():
        validation = validate_root(kind, root_data, trace_required=True)
        checks.append(check_result(f"{kind} correctness/window/telemetry/trace", validation["pass"], validation))
        for period in (128, 256, 512):
            present = {
                base_key(case)
                for case in root_data["cases"]
                if case_key(case)[0] == period
            }
            checks.append(
                check_result(
                    f"{kind} coverage N={period}",
                    present == expected_bases,
                    {"expected": len(expected_bases), "present": len(present), "missing": sorted(str(item) for item in expected_bases - present)},
                )
            )

    runtime_clean = compare_metric(
        "runtime-disabled clean difference vs release",
        runtime,
        release,
        expected_bases,
        0,
        0,
        relative_clean,
    )
    checks.append(
        check_result(
            "runtime-disabled clean difference 95% CI includes 0 and abs mean <= 1%",
            runtime_clean["includesZero"] and runtime_clean["mean"] is not None and abs(runtime_clean["mean"]) <= 1.0,
            {"metric": runtime_clean, "limitPct": 1.0, "confidence": "95% t interval"},
        )
    )

    for kind, root_data in traces.items():
        clean_n128 = compare_metric(
            f"{kind} N=128 clean overhead vs release",
            root_data,
            release,
            expected_bases,
            128,
            0,
            relative_clean,
        )
        checks.append(
            check_result(
                f"{kind} N=128 clean overhead <= 2%",
                clean_n128["mean"] is not None and abs(clean_n128["mean"]) <= 2.0,
                {"metric": clean_n128, "limitPct": 2.0},
            )
        )

        slowdown_n128 = compare_metric(
            f"{kind} N=128 concurrent slowdown difference vs release",
            root_data,
            release,
            expected_bases,
            128,
            0,
            slowdown_difference,
        )
        release_slowdowns: List[float] = []
        for base in sorted(expected_bases):
            release_case = find_case(release, base, 0)
            if release_case is not None and slowdown(release_case) is not None:
                release_slowdowns.append(float(slowdown(release_case)))
        release_mean = statistics.fmean(release_slowdowns) if release_slowdowns else None
        limit = max(1.0, abs(release_mean) * 0.20) if release_mean is not None else None
        checks.append(
            check_result(
                f"{kind} N=128 slowdown difference <= max(1pp, 20% of release)",
                limit is not None and slowdown_n128["mean"] is not None and abs(slowdown_n128["mean"]) <= limit,
                {"metric": slowdown_n128, "releaseSlowdownMeanPct": release_mean, "limitPp": limit},
            )
        )
        stability = event_stability(root_data)
        checks.append(check_result(f"{kind} N=128/N=256 per-event p50/p99 stability", stability["pass"], stability))

    passed = all(check["pass"] for check in checks)
    return {
        "schemaVersion": 1,
        "plan": "Stage 9 device-only primitive trace observer-effect gate",
        "passed": passed,
        "releaseRoot": release["root"],
        "runtimeDisabledRoot": runtime["root"],
        "traceRoots": {kind: value["root"] for kind, value in traces.items()},
        "checks": checks,
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# NCCL device-only primitive trace gate",
        "",
        f"- Overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Release root: `{result['releaseRoot']}`",
        f"- Runtime-disabled root: `{result['runtimeDisabledRoot']}`",
        "",
        "| Check | Result | Key details |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        details = []
        if "metric" in check:
            metric = check["metric"]
            details.append(f"n={metric.get('n')}, mean={metric.get('mean')}")
            if metric.get("lower95") is not None:
                details.append(f"95% CI=[{metric['lower95']}, {metric['upper95']}]")
        if "limitPct" in check:
            details.append(f"limit={check['limitPct']}%")
        if "limitPp" in check:
            details.append(f"limit={check['limitPp']}pp")
        if "p50" in check:
            details.append(f"p50={check['p50'].get('relativeDifference')}, p99={check['p99'].get('relativeDifference')}")
        lines.append(f"| {check['name']} | {'PASS' if check['pass'] else 'FAIL'} | {'; '.join(details)} |")
        failures = check.get("failures", [])
        for failure in failures[:10]:
            lines.append(f"|  |  | `{failure}` |")
        if len(failures) > 10:
            lines.append(f"|  |  | ... {len(failures) - 10} more failures |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releaseRoot", type=Path, required=True)
    parser.add_argument("--runtimeDisabledRoot", type=Path, required=True)
    parser.add_argument("--traceRoot", action="append", required=True, metavar="KIND=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = analyze(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
        print(json.dumps({"passed": result["passed"], "output": str(args.output)}, sort_keys=True))
        return 0 if result["passed"] else 1
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
