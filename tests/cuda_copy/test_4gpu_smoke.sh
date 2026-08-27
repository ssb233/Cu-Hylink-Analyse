#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
multi_bin="${repo_root}/build/cuda_copy/d2d_multi_peer_bw"
background_bin="${repo_root}/build/cuda_copy/host_copy_background"
tmp_dir="$(mktemp -d -t cuda-copy-4gpu-smoke.XXXXXX)"
background_pid=""

cleanup() {
  if [[ -n "${background_pid}" ]] && kill -0 "${background_pid}" 2>/dev/null; then
    touch "${tmp_dir}/stop"
    kill -INT "${background_pid}" 2>/dev/null || true
    wait "${background_pid}" 2>/dev/null || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

[[ -x "${multi_bin}" ]] || { echo "FAIL: missing ${multi_bin}" >&2; exit 1; }
[[ -x "${background_bin}" ]] || { echo "FAIL: missing ${background_bin}" >&2; exit 1; }

ring_log="${tmp_dir}/ring.log"
"${multi_bin}" --pattern=ring --deviceList=0,1,2,3 \
  --repeats=1 --size=1M --output="${tmp_dir}/ring.json" >"${ring_log}"
grep -F -- "pattern=ring" "${ring_log}" >/dev/null
grep -F -- "devices=0,1,2,3" "${ring_log}" >/dev/null
grep -F -- "directions=4" "${ring_log}" >/dev/null
grep -F -- "warmup=10" "${ring_log}" >/dev/null

allpairs_log="${tmp_dir}/allpairs.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --repeats=1 --size=1M --output="${tmp_dir}/allpairs.json" >"${allpairs_log}"
grep -F -- "pattern=allpairs" "${allpairs_log}" >/dev/null
grep -F -- "directions=12" "${allpairs_log}" >/dev/null
! grep -F -- '"syncEachIteration"' "${tmp_dir}/allpairs.json" >/dev/null

sync_log="${tmp_dir}/sync-each.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --repeats=3 --syncEachIteration=1 --size=1M \
  --output="${tmp_dir}/sync-each.json" >"${sync_log}"
grep -F -- "syncEachIteration=1" "${sync_log}" >/dev/null
grep -F -- '"syncEachIteration": true' "${tmp_dir}/sync-each.json" >/dev/null

grep -F -- '"streamDependency": "none"' "${tmp_dir}/allpairs.json" >/dev/null

edge_stream_log="${tmp_dir}/edge-stream.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --repeats=2 --edgeOrder=destination-major --streamMode=per-edge \
  --size=1M --output="${tmp_dir}/edge-stream.json" >"${edge_stream_log}"
grep -F -- "edgeOrder=destination-major" "${edge_stream_log}" >/dev/null
grep -F -- "streamMode=per-edge" "${edge_stream_log}" >/dev/null
grep -F -- '"edgeOrder": "destination-major"' "${tmp_dir}/edge-stream.json" >/dev/null
grep -F -- '"streamMode": "per-edge"' "${tmp_dir}/edge-stream.json" >/dev/null
grep -F -- '"streamDependency": "none"' "${tmp_dir}/edge-stream.json" >/dev/null
grep -F -- '"edgeResults": [' "${tmp_dir}/edge-stream.json" >/dev/null
grep -F -- '"source": 0, "destination": 1' "${tmp_dir}/edge-stream.json" >/dev/null

permutation_log="${tmp_dir}/permutation.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --edgePermutation=2,1,0 --repeats=2 --size=1M \
  --output="${tmp_dir}/permutation.json" >"${permutation_log}"
grep -F -- "edgePermutation=2,1,0" "${permutation_log}" >/dev/null
grep -F -- '"edgePermutation": [2, 1, 0]' "${tmp_dir}/permutation.json" >/dev/null
grep -F -- '"source": 0, "destinations": [3, 2, 1]' "${tmp_dir}/permutation.json" >/dev/null

offset_log="${tmp_dir}/offset.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --streamsPerSource=1 --sourceOffsetsUs=0,1,2,3 --repeats=2 --size=1M \
  --output="${tmp_dir}/offset.json" >"${offset_log}"
grep -F -- "sourceOffsetsUs=0,1,2,3" "${offset_log}" >/dev/null
grep -F -- '"sourceOffsetsUs": [0, 1, 2, 3]' "${tmp_dir}/offset.json" >/dev/null

chain_log="${tmp_dir}/edge-chain.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --repeats=2 --edgeOrder=source-major --streamMode=per-edge \
  --streamDependency=source-chain --size=1M \
  --output="${tmp_dir}/edge-chain.json" >"${chain_log}"
grep -F -- "streamDependency=source-chain" "${chain_log}" >/dev/null
grep -F -- '"streamDependency": "source-chain"' "${tmp_dir}/edge-chain.json" >/dev/null

stream_count_log="${tmp_dir}/stream-count-2.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --streamsPerSource=2 --streamDependency=none --repeats=2 --size=1M \
  --output="${tmp_dir}/stream-count-2.json" >"${stream_count_log}"
grep -F -- "streamsPerSource=2" "${stream_count_log}" >/dev/null
grep -F -- '"streamsPerSource": 2' "${tmp_dir}/stream-count-2.json" >/dev/null
grep -F -- '"streamMode": "per-source"' "${tmp_dir}/stream-count-2.json" >/dev/null
! grep -F -- '"edgeResults": [' "${tmp_dir}/stream-count-2.json" >/dev/null

for assignment in 0,1,1 1,0,1 1,1,0; do
  assignment_id="${assignment//,/}"
  assignment_log="${tmp_dir}/assignment-${assignment_id}.log"
  "${multi_bin}" --pattern=allpairs --devices=4 \
    --streamsPerSource=2 --streamAssignment="${assignment}" \
    --repeats=2 --size=1M \
    --output="${tmp_dir}/assignment-${assignment_id}.json" >"${assignment_log}"
  grep -F -- "streamAssignment=${assignment}" "${assignment_log}" >/dev/null
  grep -F -- "\"streamAssignment\": [" "${tmp_dir}/assignment-${assignment_id}.json" >/dev/null
done

chunk_log="${tmp_dir}/chunked.log"
"${multi_bin}" --pattern=allpairs --devices=4 \
  --repeats=5 --chunkRepeats=2 --size=1M \
  --output="${tmp_dir}/chunked.json" >"${chunk_log}"
grep -F -- "chunkRepeats=2" "${chunk_log}" >/dev/null
grep -F -- '"chunkRepeats": 2' "${tmp_dir}/chunked.json" >/dev/null
grep -F -- '"chunkCount": 3' "${tmp_dir}/chunked.json" >/dev/null
grep -F -- '"index": 2, "repeats": 1' "${tmp_dir}/chunked.json" >/dev/null

ready_file="${tmp_dir}/ready"
stop_file="${tmp_dir}/stop"
background_log="${tmp_dir}/background.log"
"${background_bin}" --devList=0,1,2,3 --direction=d2h --size=1M \
  --readyFile="${ready_file}" --stopFile="${stop_file}" \
  --reportSec=1 --output="${tmp_dir}/background.json" >"${background_log}" 2>&1 &
background_pid=$!

for _ in $(seq 1 100); do
  [[ -f "${ready_file}" ]] && break
  if ! kill -0 "${background_pid}" 2>/dev/null; then
    cat "${background_log}" >&2
    exit 1
  fi
  sleep 0.1
done
[[ -f "${ready_file}" ]] || { cat "${background_log}" >&2; exit 1; }
touch "${stop_file}"
wait "${background_pid}"
background_pid=""
grep -F -- "direction=d2h" "${background_log}" >/dev/null
grep -F -- "devices=0,1,2,3" "${background_log}" >/dev/null

echo "PASS: four-GPU CUDA smoke tests"
