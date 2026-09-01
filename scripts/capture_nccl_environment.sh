#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-doc/results/nccl-contention/environment/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output_root"

capture() {
  local name="$1"
  shift
  {
    echo "command: $*"
    echo
    if "$@"; then
      rc=0
    else
      rc=$?
    fi
    echo
    echo "exit_code=$rc"
  } > "$output_root/$name.txt" 2>&1
}

capture gpu-list nvidia-smi -L
capture nvidia-smi nvidia-smi
capture topology nvidia-smi topo -m
capture nvcc-version nvcc --version
capture nsys-version nsys --version
capture ncu-version ncu --version
capture official-status git -C /home/songxb26/HyLink/nccl status --short
capture official-describe git -C /home/songxb26/HyLink/nccl describe --tags --always --dirty
capture official-dirty-diff git -C /home/songxb26/HyLink/nccl diff -- src/device/prims_simple.h
capture third-party-status git -C third_party/nccl status --short
capture third-party-describe git -C third_party/nccl describe --tags --always --dirty
capture third-party-dirty-diff git -C third_party/nccl diff -- makefiles/common.mk
capture nccl-tests-describe git -C third_party/nccl-tests describe --tags --always --dirty
capture nccl-tests-status git -C third_party/nccl-tests status --short
capture third-party-sys-fence rg -n 'fence_acq_rel_(sys|gpu)' third_party/nccl/src/device/prims_simple.h
capture official-build-ldd ldd /home/songxb26/HyLink/nccl-tests/build/all_gather_perf
capture current-environment env

{
  echo "NCCL_HOME=${NCCL_HOME:-unset}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "repo_root=$(pwd)"
  echo "repo_revision=$(git rev-parse HEAD)"
  echo "deviceList=0,1,2"
  echo "three_gpu_adaptation=true"
  echo "third_party_nccl_revision=$(git -C third_party/nccl rev-parse HEAD)"
  echo "third_party_nccl_tests_revision=$(git -C third_party/nccl-tests rev-parse HEAD)"
  echo "third_party_lib=$(readlink -f third_party/nccl/build/lib/libnccl.so.2)"
  echo "third_party_lib_sha256=$(sha256sum third_party/nccl/build/lib/libnccl.so.2.31.2 | awk '{print $1}')"
  echo "official_lib=$(readlink -f /home/songxb26/HyLink/nccl/build/lib/libnccl.so.2)"
  echo "official_lib_sha256=$(sha256sum /home/songxb26/HyLink/nccl/build/lib/libnccl.so.2.29.2 | awk '{print $1}')"
  echo "official_prims_diff_sha256=$(git -C /home/songxb26/HyLink/nccl diff -- src/device/prims_simple.h | sha256sum | awk '{print $1}')"
  echo "third_party_common_diff_sha256=$(git -C third_party/nccl diff -- makefiles/common.mk | sha256sum | awk '{print $1}')"
} > "$output_root/identity.txt"

echo "$output_root"
