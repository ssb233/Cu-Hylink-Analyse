#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
src_dir="${repo_root}/src/cuda_copy"
build_dir="${repo_root}/build/cuda_copy"

if [[ -n "${NVCC_BIN:-}" ]]; then
  nvcc_bin="${NVCC_BIN}"
elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
  nvcc_bin="/usr/local/cuda/bin/nvcc"
else
  nvcc_bin="$(command -v nvcc)"
fi

mkdir -p "${build_dir}"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/d2d_peer_bw.cu" \
  -o "${build_dir}/d2d_peer_bw"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/host_copy_background.cu" \
  -o "${build_dir}/host_copy_background"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/d2d_multi_peer_bw.cu" \
  -o "${build_dir}/d2d_multi_peer_bw"

echo "Built CUDA copy benchmarks in ${build_dir}"
