#!/usr/bin/env python3
"""Run the four SHM-control cases required by the updated experiment plan.

The victim is a P2P NCCL collective selected by ``--collectives``.  The three
treatment controls are: no aggressor, a second P2P NCCL collective, a second
SHM NCCL collective, and an explicit four-edge CE relay.  NCCL aggressors use
the instrumented READY/STOP files and write their transport-selection logs for
later auditing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts import run_nccl_p0_overlap as p0  # noqa: E402


CONTROL_NAMES = ("alone", "p2p-nccl", "shm-nccl", "ce-relay")
COLLECTIVE_BINARIES = p0.COLLECTIVE_BINARIES


def classify_transport(log_text: str, p2p_disabled: Optional[bool]) -> Dict[str, Any]:
    shm_direct = len(re.findall(r"via SHM/direct", log_text))
    direct = len(re.findall(r"via P2P/direct|via direct", log_text))
    p2p_disable_lines = len(re.findall(r"NCCL_P2P_DISABLE set by environment", log_text))
    if shm_direct:
        classification = "shm-direct"
    elif direct:
        classification = "p2p-direct"
    else:
        classification = "unknown"
    return {
        "classification": classification,
        "shmDirectLines": shm_direct,
        "directLines": direct,
        "p2pDisableSetLines": p2p_disable_lines,
        "expectedP2PDisabled": p2p_disabled,
    }


def parse_aggressor_window(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return p0.marker_window(path)
    except (OSError, RuntimeError, json.JSONDecodeError):
        return None


def aggressor_command(
    tests_build: Path,
    collective: str,
    size: str,
    devices: str,
    warmup: int,
    cpus: str,
) -> List[str]:
    command = p0.victim_command(
        tests_build,
        collective,
        size,
        devices,
        warmup,
        1,
        cpus,
    )
    command.extend(["-N", "0"])
    return command


def aggressor_environment(
    devices: str,
    nccl_build: Path,
    marker_file: Path,
    ready_file: Path,
    stop_file: Path,
    p2p_disabled: bool,
) -> Dict[str, str]:
    environment = p0.command_environment(devices, nccl_build, marker_file)
    environment.update(
        {
            "NCCL_TESTS_READY_FILE": str(ready_file),
            "NCCL_TESTS_STOP_FILE": str(stop_file),
            "NCCL_P2P_DISABLE": "1" if p2p_disabled else "0",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET,COLL",
        }
    )
    return environment


def start_aggressor(
    *,
    case_dir: Path,
    tests_build: Path,
    nccl_build: Path,
    devices: str,
    collective: str,
    size: str,
    warmup: int,
    cpus: str,
    p2p_disabled: bool,
    timeout_sec: float,
) -> Tuple[subprocess.Popen[Any], Any, Dict[str, Any]]:
    marker_file = case_dir / "aggressor.window.jsonl"
    ready_file = case_dir / "aggressor.ready"
    stop_file = case_dir / "aggressor.stop"
    log_file = case_dir / "aggressor.log"
    command = aggressor_command(tests_build, collective, size, devices, warmup, cpus)
    environment = aggressor_environment(
        devices,
        nccl_build,
        marker_file,
        ready_file,
        stop_file,
        p2p_disabled,
    )
    stream = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    try:
        p0.wait_for_file(ready_file, process, timeout_sec)
    except Exception:
        p0.stop_relay(process, stop_file, 10.0)
        stream.close()
        raise
    return process, stream, {
        "command": command,
        "markerFile": str(marker_file),
        "readyFile": str(ready_file),
        "stopFile": str(stop_file),
        "log": str(log_file),
        "p2pDisabled": p2p_disabled,
    }


def run_nccl_aggressor_treatment(
    *,
    case_dir: Path,
    tests_build: Path,
    nccl_build: Path,
    devices: str,
    collective: str,
    victim_size: str,
    aggressor_size: str,
    victim_warmup: int,
    victim_iterations: int,
    aggressor_warmup: int,
    aggressor_cpus: str,
    victim_cpus: str,
    p2p_disabled: bool,
    timeout_sec: float,
    label: str,
) -> Dict[str, Any]:
    process: Optional[subprocess.Popen[Any]] = None
    stream: Any = None
    metadata: Dict[str, Any] = {}
    try:
        process, stream, metadata = start_aggressor(
            case_dir=case_dir,
            tests_build=tests_build,
            nccl_build=nccl_build,
            devices=devices,
            collective=collective,
            size=aggressor_size,
            warmup=aggressor_warmup,
            cpus=aggressor_cpus,
            p2p_disabled=p2p_disabled,
            timeout_sec=timeout_sec,
        )
        victim = p0.run_victim(
            case_dir=case_dir,
            tests_build=tests_build,
            nccl_build=nccl_build,
            collective=collective,
            size=victim_size,
            devices=devices,
            warmup=victim_warmup,
            iterations=victim_iterations,
            victim_cpus=victim_cpus,
            label=label,
        )
    finally:
        if process is not None:
            metadata["returnCode"] = p0.stop_relay(
                process, case_dir / "aggressor.stop", timeout_sec
            )
        if stream is not None:
            stream.close()

    log_path = Path(metadata["log"])
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    marker_path = Path(metadata["markerFile"])
    p2p_disabled_expected = bool(metadata["p2pDisabled"])
    result: Dict[str, Any] = {
        "label": label,
        "control": "shm-nccl" if p2p_disabled_expected else "p2p-nccl",
        "returnCode": victim.get("returnCode"),
        "measurement": victim.get("measurement"),
        "measurementError": victim.get("measurementError"),
        "window": victim.get("window"),
        "windowError": victim.get("windowError"),
        "log": victim.get("log"),
        "windowFile": victim.get("windowFile"),
        "gpuBefore": victim.get("gpuBefore"),
        "gpuAfter": victim.get("gpuAfter"),
        "aggressor": {
            **metadata,
            "window": parse_aggressor_window(marker_path),
            "transport": classify_transport(log_text, p2p_disabled_expected),
        },
    }
    result = {key: value for key, value in result.items() if value is not None}
    return result


def normalize_treatment(treatment: Dict[str, Any]) -> Dict[str, Any]:
    if "returnCode" not in treatment and "victimReturnCode" in treatment:
        treatment["returnCode"] = treatment["victimReturnCode"]
    return treatment


def self_test() -> None:
    assert CONTROL_NAMES == ("alone", "p2p-nccl", "shm-nccl", "ce-relay")
    assert set(COLLECTIVE_BINARIES) == {"allgather", "allreduce", "reducescatter"}
    assert classify_transport("Channel 00 via SHM/direct", True)["classification"] == "shm-direct"
    assert classify_transport("Channel 00 via P2P/direct pointer", False)["classification"] == "p2p-direct"
    unknown = classify_transport("no transport line", None)
    assert unknown["classification"] == "unknown"
    assert "-N" in aggressor_command(Path("/tmp/tests"), "allgather", "64M", "0,1,2,3", 20, "0")
    assert any(item.endswith("all_reduce_perf") for item in aggressor_command(Path("/tmp/tests"), "allreduce", "64M", "0,1,2,3", 20, "0"))
    assert any(item.endswith("reduce_scatter_perf") for item in aggressor_command(Path("/tmp/tests"), "reducescatter", "64M", "0,1,2,3", 20, "0"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repoRoot", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--victimSize", default="64M")
    parser.add_argument("--aggressorSize", default="255M")
    parser.add_argument("--victimIterations", type=int, default=5000)
    parser.add_argument("--victimWarmup", type=int, default=100)
    parser.add_argument("--aggressorWarmup", type=int, default=20)
    parser.add_argument("--collectives", default="allgather")
    parser.add_argument("--controls", default=",".join(CONTROL_NAMES))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--aggressorCpus", default="0,2,4,6")
    parser.add_argument("--victimCpus", default="8,10")
    parser.add_argument("--settleMs", type=int, default=250)
    parser.add_argument("--randomSeed", type=int, default=20260831)
    parser.add_argument("--timeoutSec", type=float, default=180.0)
    parser.add_argument("--ncclBuild", type=Path)
    parser.add_argument("--testsBuild", type=Path)
    parser.add_argument("--relayBin", type=Path)
    parser.add_argument("--outputRoot", type=Path, required=False)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo_root = args.repoRoot.resolve()
    nccl_build = (args.ncclBuild or repo_root / "build/nccl-v2.31.2-sm70-sys").resolve()
    tests_build = (args.testsBuild or repo_root / "build/nccl-tests-p0-sm70").resolve()
    relay_bin = (args.relayBin or repo_root / "build/cuda_copy/host_relay").resolve()
    output_root = args.outputRoot
    if output_root is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_root = repo_root / "doc/results/nccl-hybrid-path/shm-control-p0" / stamp
    output_root = output_root.resolve()
    collectives = p0.parse_csv(args.collectives)
    if any(collective not in COLLECTIVE_BINARIES for collective in collectives):
        raise ValueError(f"unknown collective; choose from {tuple(COLLECTIVE_BINARIES)}")
    controls = p0.parse_csv(args.controls)
    if any(control not in CONTROL_NAMES for control in controls):
        raise ValueError(f"unknown control; choose from {CONTROL_NAMES}")
    if args.repetitions <= 0 or args.victimIterations <= 0 or args.victimWarmup < 0:
        raise ValueError("repetitions/victimIterations must be positive and victimWarmup non-negative")
    if args.aggressorWarmup < 0 or args.settleMs < 0:
        raise ValueError("aggressorWarmup and settleMs must be non-negative")
    if not relay_bin.is_file():
        raise FileNotFoundError(relay_bin)
    for collective in collectives:
        binary = tests_build / COLLECTIVE_BINARIES[collective]
        if not binary.is_file():
            raise FileNotFoundError(binary)
    if not (nccl_build / "lib/libnccl.so.2.31.2").is_file():
        raise FileNotFoundError(nccl_build / "lib/libnccl.so.2.31.2")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.randomSeed)
    manifest: Dict[str, Any] = {
        "schemaVersion": 1,
        "clock": "steady_clock/time.monotonic_ns",
        "createdAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "victim": {
            "collectives": collectives,
            "size": args.victimSize,
            "iterations": args.victimIterations,
            "warmup": args.victimWarmup,
            "devices": args.devices,
            "cpus": args.victimCpus,
        },
        "aggressor": {
            "size": args.aggressorSize,
            "warmup": args.aggressorWarmup,
            "cpus": args.aggressorCpus,
        },
        "controls": controls,
        "randomSeed": args.randomSeed,
        "ncclBuild": str(nccl_build),
        "testsBuild": str(tests_build),
        "relayBin": str(relay_bin),
        "cases": [],
    }

    for collective in collectives:
        for repetition in range(1, args.repetitions + 1):
            order = list(controls)
            rng.shuffle(order)
            for control in order:
                case_dir = output_root / collective / f"rep-{repetition}" / control
                case_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[{collective} {args.victimSize} rep-{repetition} {control}] "
                    "clean-before -> treatment -> clean-after",
                    flush=True,
                )
                before = p0.run_victim(
                    case_dir=case_dir,
                    tests_build=tests_build,
                    nccl_build=nccl_build,
                    collective=collective,
                    size=args.victimSize,
                    devices=args.devices,
                    warmup=args.victimWarmup,
                    iterations=args.victimIterations,
                    victim_cpus=args.victimCpus,
                    label="clean-before",
                )
                if control == "alone":
                    treatment = p0.run_victim(
                        case_dir=case_dir,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        collective=collective,
                        size=args.victimSize,
                        devices=args.devices,
                        warmup=args.victimWarmup,
                        iterations=args.victimIterations,
                        victim_cpus=args.victimCpus,
                        label="alone",
                    )
                    treatment["control"] = control
                elif control in ("p2p-nccl", "shm-nccl"):
                    treatment = run_nccl_aggressor_treatment(
                        case_dir=case_dir,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        devices=args.devices,
                        collective=collective,
                        victim_size=args.victimSize,
                        aggressor_size=args.aggressorSize,
                        victim_warmup=args.victimWarmup,
                        victim_iterations=args.victimIterations,
                        aggressor_warmup=args.aggressorWarmup,
                        aggressor_cpus=args.aggressorCpus,
                        victim_cpus=args.victimCpus,
                        p2p_disabled=control == "shm-nccl",
                        timeout_sec=args.timeoutSec,
                        label=control,
                    )
                else:
                    relay_scenario = p0.Scenario(
                        "ring", "relay", "0:1,1:2,2:3,3:0"
                    )
                    treatment = p0.run_treatment(
                        case_dir=case_dir,
                        relay_bin=relay_bin,
                        tests_build=tests_build,
                        nccl_build=nccl_build,
                        scenario=relay_scenario,
                        collective=collective,
                        victim_size=args.victimSize,
                        relay_size=args.aggressorSize,
                        devices=args.devices,
                        victim_warmup=args.victimWarmup,
                        victim_iterations=args.victimIterations,
                        relay_warmup=args.aggressorWarmup,
                        relay_report_ms=100,
                        relay_cpus=args.aggressorCpus,
                        victim_cpus=args.victimCpus,
                        settle_ms=args.settleMs,
                        timeout_sec=args.timeoutSec,
                    )
                    treatment["control"] = control
                treatment = normalize_treatment(treatment)
                after = p0.run_victim(
                    case_dir=case_dir,
                    tests_build=tests_build,
                    nccl_build=nccl_build,
                    collective=collective,
                    size=args.victimSize,
                    devices=args.devices,
                    warmup=args.victimWarmup,
                    iterations=args.victimIterations,
                    victim_cpus=args.victimCpus,
                    label="clean-after",
                )
                case = {
                    "collective": collective,
                    "repetition": repetition,
                    "control": control,
                    "randomizedOrder": order,
                    "caseDir": str(case_dir),
                    "cleanBefore": before,
                    "treatment": treatment,
                    "cleanAfter": after,
                }
                (case_dir / "case.json").write_text(
                    json.dumps(case, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                manifest["cases"].append(case)
                (output_root / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    print(f"SHM control results: {output_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        return run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
