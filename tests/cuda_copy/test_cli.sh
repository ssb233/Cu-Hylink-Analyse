#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
d2d_bin="${repo_root}/build/cuda_copy/d2d_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${d2d_bin}" ]] || fail "missing executable: ${d2d_bin}"
[[ -x "${background_bin}" ]] || fail "missing executable: ${background_bin}"

d2d_help="$(${d2d_bin} --help 2>&1)" || fail "d2d --help failed"
grep -F -- "--mode=unidirectional|bidirectional" <<<"${d2d_help}" >/dev/null || fail "d2d mode help missing"
grep -F -- "--repeats=20" <<<"${d2d_help}" >/dev/null || fail "d2d default repeats missing"
grep -F -- "warmup=10 (fixed)" <<<"${d2d_help}" >/dev/null || fail "d2d fixed warmup missing"
grep -F -- "size=255M" <<<"${d2d_help}" >/dev/null || fail "d2d default size missing"
grep -F -- "--srcDev=0" <<<"${d2d_help}" >/dev/null || fail "d2d srcDev default missing"
grep -F -- "--dstDev=1" <<<"${d2d_help}" >/dev/null || fail "d2d dstDev default missing"

background_help="$(${background_bin} --help 2>&1)" || fail "background --help failed"
grep -F -- "--devList=0,1" <<<"${background_help}" >/dev/null || fail "background default devList missing"
grep -F -- "--direction=d2h|h2d" <<<"${background_help}" >/dev/null || fail "background direction help missing"
grep -F -- "size=255M" <<<"${background_help}" >/dev/null || fail "background default size missing"

if "${d2d_bin}" --mode=unidirectional --srcDev=0 --dstDev=0 --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "d2d accepted identical source and destination devices"
fi

if "${background_bin}" --direction=invalid --devList=0 --size=1M >/dev/null 2>&1; then
  fail "background accepted an invalid direction"
fi

echo "PASS: CLI defaults and validation"
