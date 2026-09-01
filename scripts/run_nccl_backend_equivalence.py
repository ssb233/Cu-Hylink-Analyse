#!/usr/bin/env python3
"""Run clean NCCL measurements for named backend equivalence gates.

The test binary is shared by all backends and receives the selected NCCL
library through the runner's environment.  Backends are deliberately named
instead of accepting an arbitrary library path so that the experiment matrix
cannot silently mix builds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts import run_nccl_p0_overlap as p0  # noqa: E402


BACKEND_BUILD_DIRS = {
    "official-sys": Path("build/nccl-official-v2.31.2-sm70-sys"),
    "experiment-sys": Path("build/nccl-experiment-v2.31.2-sm70-sys-equivalence"),
    "experiment-gpu": Path("build/nccl-experiment-v2.31.2-sm70-gpu"),
}
SUPPORTED_SIZES = ("64M", "256M")


def parse_csv(value: str) -> List[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("comma-separated value cannot be empty")
    return result


def backend_build(repo_root: Path, backend: str) -> Path:
    try:
        relative = BACKEND_BUILD_DIRS[backend]
    except KeyError as error:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {tuple(BACKEND_BUILD_DIRS)}"
        ) from error
    return (repo_root / relative).resolve()


def iterations_for_size(size: str, iterations_64m: int, iterations_256m: int) -> int:
    normalized = size.upper()
    if normalized == "64M":
        return iterations_64m
    if normalized == "256M":
        return iterations_256m
    raise ValueError(f"unsupported size {size!r}; choose from {SUPPORTED_SIZES}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(path: Path) -> Dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return completed.stdout.strip()

    return {
        "describe": git("describe", "--tags", "--always", "--dirty"),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
    }


def build_identity(repo_root: Path, backend: str) -> Dict[str, Any]:
    build = backend_build(repo_root, backend)
    library = build / "lib/libnccl.so.2.31.2"
    if not library.is_file():
        raise FileNotFoundError(f"{backend} library is missing: {library}")
    return {
        "backend": backend,
        "buildDir": str(build),
        "library": str(library),
        "libraryRealpath": str(library.resolve()),
        "librarySha256": sha256_file(library),
    }


def self_test() -> None:
    assert tuple(BACKEND_BUILD_DIRS) == (
        "official-sys",
        "experiment-sys",
        "experiment-gpu",
    )
    root = Path("/tmp/nccl-backend-equivalence-self-test")
    assert backend_build(root, "official-sys") == (
        root / BACKEND_BUILD_DIRS["official-sys"]
    ).resolve()
    assert iterations_for_size("64M", 5000, 1300) == 5000
    assert iterations_for_size("256M", 5000, 1300) == 1300
    try:
        backend_build(root, "/tmp/arbitrary-library")
    except ValueError:
        pass
    else:
        raise AssertionError("arbitrary backend path was accepted")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--backends", default="official-sys,experiment-sys",
        help="named backends; supported: official-sys, experiment-sys, experiment-gpu",
    )
    parser.add_argument("--collectives", default="allgather,allreduce,reducescatter")
    parser.add_argument("--sizes", default="64M,256M")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--iterations64M", type=int, default=5000)
    parser.add_argument("--iterations256M", type=int, default=1300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--victimCpus", default="8,10")
    parser.add_argument("--randomSeed", type=int, default=20260831)
    parser.add_argument("--repoRoot", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--testsBuild", type=Path)
    parser.add_argument("--outputRoot", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    repo_root = args.repoRoot.resolve()
    backends = parse_csv(args.backends)
    collectives = parse_csv(args.collectives)
    sizes = [item.upper() for item in parse_csv(args.sizes)]
    for backend in backends:
        backend_build(repo_root, backend)
    if any(item not in p0.COLLECTIVE_BINARIES for item in collectives):
        raise ValueError(
            f"unknown collective; choose from {tuple(p0.COLLECTIVE_BINARIES)}"
        )
    if any(item not in SUPPORTED_SIZES for item in sizes):
        raise ValueError(f"unknown size; choose from {SUPPORTED_SIZES}")
    if args.repetitions <= 0 or args.iterations64M <= 0 or args.iterations256M <= 0:
        raise ValueError("repetitions and iteration counts must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    tests_build = (args.testsBuild or repo_root / "build/nccl-tests-p0-sm70").resolve()
    for collective in collectives:
        binary = p0.COLLECTIVE_BINARIES[collective]
        if not (tests_build / binary).is_file():
            raise FileNotFoundError(tests_build / binary)

    output_root = args.outputRoot
    if output_root is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_root = repo_root / "doc/results/nccl-hybrid-path/backend-equivalence" / stamp
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise ValueError(f"outputRoot already contains a manifest: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    identities = {
        backend: build_identity(repo_root, backend) for backend in backends
    }
    test_identity = {
        "buildDir": str(tests_build),
        "binaries": {collective: str(tests_build / p0.COLLECTIVE_BINARIES[collective]) for collective in collectives},
    }
    manifest: Dict[str, Any] = {
        "schemaVersion": 1,
        "clock": "steady_clock/time.monotonic_ns",
        "createdAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "clean-only-equivalence-gate",
        "backends": identities,
        "tests": test_identity,
        "source": {
            "official": git_metadata(Path("/home/songxb26/HyLink/nccl")),
            "experiment": git_metadata(repo_root / "third_party/nccl"),
        },
        "config": {
            "backends": backends,
            "collectives": collectives,
            "sizes": sizes,
            "repetitions": args.repetitions,
            "iterations64M": args.iterations64M,
            "iterations256M": args.iterations256M,
            "warmup": args.warmup,
            "devices": args.devices,
            "victimCpus": args.victimCpus,
            "randomSeed": args.randomSeed,
        },
        "cases": [],
    }
    rng = random.Random(args.randomSeed)
    jobs: List[Tuple[str, str, str]] = [
        (backend, collective, size)
        for backend in backends
        for collective in collectives
        for size in sizes
    ]

    for repetition in range(1, args.repetitions + 1):
        order = list(jobs)
        rng.shuffle(order)
        for backend, collective, size in order:
            case_dir = output_root / f"rep-{repetition}" / backend / f"{collective}-{size}"
            case_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{backend} rep-{repetition} {collective} {size}] clean",
                flush=True,
            )
            result = p0.run_victim(
                case_dir=case_dir,
                tests_build=tests_build,
                nccl_build=backend_build(repo_root, backend),
                collective=collective,
                size=size,
                devices=args.devices,
                warmup=args.warmup,
                iterations=iterations_for_size(
                    size, args.iterations64M, args.iterations256M
                ),
                victim_cpus=args.victimCpus,
                label="clean",
            )
            case = {
                "repetition": repetition,
                "backend": backend,
                "collective": collective,
                "size": size,
                "orderIndex": order.index((backend, collective, size)),
                "caseDir": str(case_dir),
                **result,
            }
            (case_dir / "case.json").write_text(
                json.dumps(case, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["cases"].append(case)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    print(f"NCCL backend equivalence results: {output_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
