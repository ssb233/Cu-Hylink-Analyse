#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
runner="${repo_root}/scripts/run_4gpu_copy_matrix.sh"
phase_runner="${repo_root}/scripts/run_phase_lock_diagnostic.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${multi_bin}" ]] || fail "missing executable: ${multi_bin}"
[[ -x "${runner}" ]] || fail "missing script: ${runner}"
[[ -x "${phase_runner}" ]] || fail "missing phase-lock diagnostic script: ${phase_runner}"

multi_help="$(${multi_bin} --help 2>&1)" || fail "multi-GPU --help failed"
grep -F -- "--pattern=ring|allpairs" <<<"${multi_help}" >/dev/null || fail "pattern help missing"
grep -F -- "--deviceList=0,1,2,3" <<<"${multi_help}" >/dev/null || fail "device list default missing"
grep -F -- "--devices=4" <<<"${multi_help}" >/dev/null || fail "devices help missing"
grep -F -- "--repeats=20" <<<"${multi_help}" >/dev/null || fail "default repeats missing"
grep -F -- "--chunkRepeats=0" <<<"${multi_help}" >/dev/null || fail "chunk repeats help missing"
grep -F -- "--syncEachIteration=0|1" <<<"${multi_help}" >/dev/null || fail "sync-each-iteration help missing"
grep -F -- "--edgeOrder=source-major|destination-major" <<<"${multi_help}" >/dev/null || fail "edge order help missing"
grep -F -- "--edgePermutation=0,1,2" <<<"${multi_help}" >/dev/null || fail "edge permutation help missing"
grep -F -- "--sourceOffsetsUs=0,250,500,750" <<<"${multi_help}" >/dev/null || fail "source offset help missing"
grep -F -- "--streamMode=per-source|per-edge" <<<"${multi_help}" >/dev/null || fail "stream mode help missing"
grep -F -- "--streamDependency=none|source-chain" <<<"${multi_help}" >/dev/null || fail "stream dependency help missing"
grep -F -- "--streamsPerSource=1|2|3" <<<"${multi_help}" >/dev/null || fail "streams-per-source help missing"
grep -F -- "warmup=10 (fixed)" <<<"${multi_help}" >/dev/null || fail "fixed warmup missing"
grep -F -- "size=255M" <<<"${multi_help}" >/dev/null || fail "default size missing"

runner_help="$(${runner} --help 2>&1)" || fail "4-GPU runner --help failed"
grep -F -- "--patterns=ring,allpairs" <<<"${runner_help}" >/dev/null || fail "runner pattern help missing"
grep -F -- "--directions=none,d2h,h2d" <<<"${runner_help}" >/dev/null || fail "runner direction help missing"
grep -F -- "--deviceList=0,1,2,3" <<<"${runner_help}" >/dev/null || fail "runner device list help missing"

phase_help="$(${phase_runner} --help 2>&1)" || fail "phase-lock runner --help failed"
grep -F -- "--edgeOrders=source-major,destination-major" <<<"${phase_help}" >/dev/null || fail "phase runner edge order help missing"
grep -F -- "--streamModes=per-source,per-edge" <<<"${phase_help}" >/dev/null || fail "phase runner stream mode help missing"
grep -F -- "--repeatsList=20,300" <<<"${phase_help}" >/dev/null || fail "phase runner repeats help missing"
grep -F -- "--scenarios=none,d2h-all" <<<"${phase_help}" >/dev/null || fail "phase runner scenarios help missing"

if "${phase_runner}" --runs=0 --outputRoot="$(mktemp -d -t phase-lock-invalid.XXXXXX)" \
  >/dev/null 2>&1; then
  fail "phase-lock runner accepted zero runs"
fi

if "${multi_bin}" --pattern=invalid --deviceList=0,1,2,3 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted an invalid pattern"
fi

if "${multi_bin}" --chunkRepeats=-1 --deviceList=0,1,2,3 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted negative chunk repeats"
fi

if "${multi_bin}" --syncEachIteration=2 --deviceList=0,1,2,3 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted invalid sync-each-iteration value"
fi

if "${multi_bin}" --edgeOrder=invalid --deviceList=0,1,2,3 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted an invalid edge order"
fi

if "${multi_bin}" --edgePermutation=0,0,1 --deviceList=0,1,2,3 \
  --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted a duplicate edge permutation"
fi

if "${multi_bin}" --edgePermutation=0,1 --deviceList=0,1,2,3 \
  --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted a short edge permutation"
fi

if "${multi_bin}" --sourceOffsetsUs=0,1,2 --deviceList=0,1,2,3 \
  --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted a short source offset list"
fi

if "${multi_bin}" --streamsPerSource=2 --sourceOffsetsUs=0,1,2,3 \
  --deviceList=0,1,2,3 --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted source offsets with multiple streams"
fi

if "${multi_bin}" --streamMode=invalid --deviceList=0,1,2,3 --repeats=1 --size=1M \
  >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted an invalid stream mode"
fi

if "${multi_bin}" --streamDependency=invalid --deviceList=0,1,2,3 \
  --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted an invalid stream dependency"
fi

if "${multi_bin}" --streamsPerSource=4 --deviceList=0,1,2,3 \
  --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted an invalid streams-per-source value"
fi

if "${multi_bin}" --streamMode=per-edge --streamsPerSource=2 \
  --deviceList=0,1,2,3 --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted a conflicting stream configuration"
fi

if "${multi_bin}" --streamMode=per-source --streamDependency=source-chain \
  --deviceList=0,1,2,3 --repeats=1 --size=1M >/dev/null 2>&1; then
  fail "multi-GPU benchmark accepted source-chain with per-source streams"
fi

echo "PASS: four-GPU CLI contracts"
