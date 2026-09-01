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

nvcc_real="$(readlink -f "${nvcc_bin}")"
cuda_root="$(cd "$(dirname "${nvcc_real}")/.." && pwd -P)"
cuda_target="${cuda_root}/targets/x86_64-linux"
cxx_bin="$(command -v g++)"

mkdir -p "${build_dir}"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/d2d_peer_bw.cu" \
  -o "${build_dir}/d2d_peer_bw"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/host_copy_background.cu" \
  -o "${build_dir}/host_copy_background"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -gencode=arch=compute_70,code=sm_70 \
  -I"${src_dir}" "${src_dir}/host_relay.cu" \
  -o "${build_dir}/host_relay"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/d2d_multi_peer_bw.cu" \
  -o "${build_dir}/d2d_multi_peer_bw"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/d2d_with_background_bw.cu" \
  -o "${build_dir}/d2d_with_background_bw"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/copy_path_background.cu" \
  -o "${build_dir}/copy_path_background"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/minimal_source_pair_bw.cu" \
  -o "${build_dir}/minimal_source_pair_bw"

"${nvcc_bin}" -std=c++17 -O2 -lineinfo \
  -I"${src_dir}" "${src_dir}/p2p_kernel_bw.cu" \
  -o "${build_dir}/p2p_kernel_bw"

"${cxx_bin}" -std=c++17 -O2 -fPIC -shared \
  -I"${cuda_target}/include" "${src_dir}/cupti_memcpy_channel_trace.cpp" \
  -L"${cuda_target}/lib" -Wl,-rpath,"${cuda_target}/lib" \
  -lcupti -ldl -pthread \
  -o "${build_dir}/libcupti_memcpy_channel_trace.so"

echo "Built CUDA copy benchmarks in ${build_dir}"
