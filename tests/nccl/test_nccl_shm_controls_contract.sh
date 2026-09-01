#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd -P)
python3 "$repo_root/scripts/run_nccl_shm_controls.py" --self-test
help_text=$(python3 "$repo_root/scripts/run_nccl_shm_controls.py" --help)
grep -q -- '--collectives COLLECTIVES' <<<"$help_text"
python3 - "$repo_root" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
path = root / "scripts/run_nccl_shm_controls.py"
spec = importlib.util.spec_from_file_location("shm_controls", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

args = module.parse_args(["--collectives=allreduce,reducescatter"])
assert args.collectives == "allreduce,reducescatter"
assert any(item.endswith("all_reduce_perf") for item in module.aggressor_command(pathlib.Path("/tmp/tests"), "allreduce", "64M", "0,1,2,3", 20, "0"))
assert any(item.endswith("reduce_scatter_perf") for item in module.aggressor_command(pathlib.Path("/tmp/tests"), "reducescatter", "64M", "0,1,2,3", 20, "0"))
PY
echo "PASS: NCCL SHM control configuration"
