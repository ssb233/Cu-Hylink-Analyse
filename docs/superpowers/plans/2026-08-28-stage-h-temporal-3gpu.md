# Stage H Temporal Pressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the report-defined Stage H on the currently available three V100 GPUs by matching background active duty/cadence, tracing representative cases, and executing the minimal source-local diagnostic before making the final public-tool attribution.

**Architecture:** Extend both background generators with per-device wall-time operation statistics while preserving their ready/stop protocols. Add one temporal-ablation runner/analyzer for H0/H1/H2, and a separate two-edge source-local benchmark/runner for H3. Keep `single-two-copy` as one source stream carrying two consecutive copies; use `edge-independent` as the stream-topology control. Use CUPTI/Nsight Systems only for selected temporal points and Nsight Compute only for standalone kernel-property validation.

**Tech Stack:** CUDA 12.6, nvcc, CUDA Runtime API, CUPTI activity API, Nsight Systems, Nsight Compute when available, Bash, Python 3, CSV/JSON artifacts, Tesla V100-SXM2-32GB.

**Spec:** `doc/reports/gpu-internal-copy-contention-results.md`, section 8.10 Stage H.

## Global Constraints

- Available hardware is exactly GPU 0, GPU 1, and GPU 2; do not run or report a four-GPU Stage H result.
- All victim cases use `255M`, fixed warmup `10`, measured repeats `300`, and `single-two-copy`/`edge-independent` unless a task explicitly defines the H3 two-edge workload.
- `single-two-copy` means one source stream with two consecutive copies; it is not a two-stream topology and is not equivalent to four-GPU `assignment=0,1,0`.
- Replacement background target and duty are independent controls; every result records target, actual bytes/rates, operation duration, submit interval, idle gap, wall active duty, and whether it is `bandwidth-matched`, `duty-matched`, or `saturated`.
- Original D2H is the positive control and clean original P2P is the negative control; do not infer path equivalence from aggregate GB/s alone.
- All raw results go below `doc/results/gpu-contention/copy-path-temporal/` or `doc/results/gpu-contention/minimal-source-diagnostic/`; traces go below the matching `doc/traces/gpu-contention/` directories.
- Do not call a CUPTI `channelID` a physical Copy Engine number and do not name an HBM partition, L2 slice, FIFO, credit mechanism, or NVLink scheduler without a directly supported counter.
- H0/H1 must produce a usable time-pressure comparison before HBM/L2/local-D2D negative results are used to narrow the attribution.

---

### Task 1: Add per-device temporal metrics to background generators

**Files:**
- Modify: `src/cuda_copy/host_copy_background.cu`
- Modify: `src/cuda_copy/copy_path_background.cu`
- Modify: `scripts/build_cuda_copy.sh`
- Create: `tests/cuda_copy/test_temporal_metrics_json.sh`

**Interfaces:**
- Consumes: each worker's measured operation begin/end timestamps, bytes, and operation count.
- Produces: both background JSON files contain aligned per-device arrays for `perDeviceBytes`, `perDeviceOperations`, `perDeviceWallActiveSec`, `perDeviceWallActiveDuty`, `perDeviceGpuActivitySec`, `perDeviceGpuActivityDuty`, `perDeviceOperationDurationMs`, `perDeviceSubmitIntervalMs`, and `perDeviceIdleGapMs`; the last two arrays contain `{count,p50,p90,p99,max}` objects. `perDeviceGpuActivity*` is `null` in an unprofiled process and is filled by the temporal analyzer when a CUPTI trace is available.

- [x] **Step 1: Write the failing JSON contract test.**

  Run each real background binary for a short externally controlled three-GPU interval with `--readyFile`, `--stopFile`, and `--output`. Parse both JSON files and assert three positive byte/operation entries, positive duration/interval statistics, `sum(perDeviceBytes)==totalBytes`, aligned device order, and the presence of the wall-duty and GPU-activity fields.

- [x] **Step 2: Run the test and record the expected failure.**

  Run `bash tests/cuda_copy/test_temporal_metrics_json.sh`; it must fail because the current JSON does not expose operation duration, cadence, idle-gap, and wall-duty statistics.

- [x] **Step 3: Implement timestamp collection and percentile serialization.**

  In each worker, record a monotonic `begin` immediately before issuing the operation and `end` after the stream completion. After the worker joins, compute duration `end-begin`, submit interval `begin[i]-begin[i-1]`, idle gap `max(0, begin[i]-end[i-1])`, wall active seconds as the duration sum, and wall active duty as wall active seconds divided by the common measured elapsed interval. Serialize p50/p90/p99/max with deterministic nearest-rank selection and keep all existing aggregate fields unchanged. Add a bounded `--iterations=N` option to `copy_path_background` for standalone profiler runs; `0` retains the existing forever-until-stop behavior.

- [x] **Step 4: Build and rerun the real JSON test.**

  Run `bash scripts/build_cuda_copy.sh` and `bash tests/cuda_copy/test_temporal_metrics_json.sh`; inspect both JSON files and verify the new fields are numeric or explicitly `null` only for unprofiled GPU activity.

### Task 2: Implement the temporal analyzer and schema test

**Files:**
- Create: `scripts/analyze_copy_path_temporal_ablation.py`
- Create: `tests/cuda_copy/test_copy_path_temporal_analysis.sh`

**Interfaces:**
- Consumes: a temporal runner `summary.csv`, referenced victim/background JSON, and optional CUPTI CSV plus `.meta.json`.
- Produces: `analysis.json` with per-case `bandwidthClass`, wall/gpu activity duty, duration/cadence/gap summaries, P2P slow counts when present, P2P/background overlap seconds/ratio when trace records exist, and grouped clean-to-treatment drop statistics.

- [x] **Step 1: Write the failing synthetic analysis test.**

  Create a temporary two-case summary fixture with one clean row and one D2H row, JSON metric objects, and a three-row CUPTI fixture containing a P2P interval overlapping a D2H interval. Assert the analyzer emits the correct duty, p50/p99, overlap, slow count, and clean-to-D2H drop.

- [x] **Step 2: Run the test and record the expected failure.**

  Run `bash tests/cuda_copy/test_copy_path_temporal_analysis.sh`; it must fail because the temporal analyzer is absent.

- [x] **Step 3: Implement read-only CSV/JSON aggregation.**

  Require explicit columns for pressure level, background path, victim mode, topology, target, duty, status, JSON paths, and optional trace paths. Validate three devices and all per-device array lengths. Compute overlap only within the same process/device/context scope and return `null` when no GPU activity record exists; never merge channel IDs across processes or contexts.

- [x] **Step 4: Run the analyzer contract test.**

  Run `bash tests/cuda_copy/test_copy_path_temporal_analysis.sh` and verify the output contains the literal `bandwidth-matched`, `duty-matched`, and `saturated` classes and the three-GPU topology adaptation note.

### Task 3: Implement the H0/H1 temporal ablation runner

**Files:**
- Create: `scripts/run_copy_path_temporal_ablation.sh`
- Create: `tests/cuda_copy/test_copy_path_temporal_cli.sh`

**Interfaces:**
- Consumes: `d2d_multi_peer_bw`, `host_copy_background`, `copy_path_background`, and optionally `libcupti_memcpy_channel_trace.so`.
- Produces: a fresh result root with `environment.txt`, `summary.csv`, per-case environment/command/log/JSON files, optional CUPTI CSV/meta files, and rows with `pressureLevel`, `backgroundPath`, `victimMode`, `topology`, `targetGBps`, `dutyCycle`, `bandwidthClass`, exits, and artifact paths.

- [x] **Step 1: Write the failing CLI/smoke contract test.**

  Assert `--help`, exactly-three-device validation, pressure levels `current-pulsed`, `duty-0.01`, `duty-0.1`, `duty-0.5`, `duty-1.0`, and `saturated`, and a two-repeat three-GPU smoke case. The smoke summary must contain clean, original D2H, and one replacement path, record `single-two-copy` as `--streamsPerSource=1`, and record target/duty/class fields.

- [x] **Step 2: Run the test and record the expected failure.**

  Run `bash tests/cuda_copy/test_copy_path_temporal_cli.sh`; it must fail because the temporal runner is absent.

- [x] **Step 3: Implement the runner and ready/stop lifecycle.**

  Use `deviceList=0,1,2` by default and reject other device counts. Run clean without a background, original D2H through `host_copy_background`, and replacement paths through `copy_path_background`. Map `current-pulsed` to target `4.0`, duty `1.0`; map duty levels to target `4.0` and their requested duty; map `saturated` to target `0` and duty `1.0`. Set `bandwidthClass` from the requested protocol initially and let the analyzer refine it from measured metrics. Run topologies sequentially and terminate every background with its stop marker even on errors.

- [x] **Step 4: Run the CLI smoke test and inspect artifacts.**

  Run `bash tests/cuda_copy/test_copy_path_temporal_cli.sh`; inspect `summary.csv`, one replacement JSON, and the command/environment record before starting the matrix.

### Task 4: Execute H0 and H1 and choose representative points

**Files:**
- Create by runner: `doc/results/gpu-contention/copy-path-temporal/<timestamp>/`
- Modify: `doc/reports/gpu-internal-copy-contention-results.md`

**Interfaces:**
- Consumes: the H0 metrics schema and H1 runner.
- Produces: complete three-GPU current-pulsed/duty/saturated results and an analysis JSON identifying the last immune point, first visible drop, and any pressure class that is genuinely time-comparable to original D2H.

- [x] **Step 1: Run the H0 current-pulsed controls.**

  Run clean, original D2H, and all four replacement backgrounds for both topologies, `repeats=300`, `runs=3`, with no profiler. Verify original D2H reproduces the Stage G positive control before interpreting replacement paths.

- [x] **Step 2: Run the H1 duty sweep.**

  Run replacement paths at requested duty `0.01,0.1,0.5,1.0` with target `4.0 GB/s/GPU`, both topologies, three repetitions. Run local D2D CE, streaming HBM read/write, and L2 candidate at `targetGBps=0` saturated pressure; retain clean and original D2H controls in the same result family.

- [x] **Step 3: Analyze and classify pressure matching.**

  Run `python3 scripts/analyze_copy_path_temporal_ablation.py --summary <root>/summary.csv --output <root>/analysis.json`. A row may be called `bandwidth-matched` only when mean background rate is within ±10% of D2H; `duty-matched` only when per-device wall/gpu duty is within ±10 percentage points and duration or submit cadence is within ±20%; otherwise retain the most specific class or `saturated`.

- [x] **Step 4: Select H2/H3 points from measured data.**

  Select original D2H positive control, current-pulsed replacement controls, the highest-duty replacement point, the first drop point if one exists, and their `edge-independent` controls. For H3 select the last immune and first drop D2H sizes after the 64K/4M bracket; do not choose points from aggregate drop alone when per-device timing contradicts it.

### Task 5: Collect H2 representative CUPTI and Nsight Systems traces

**Files:**
- Modify: `scripts/run_copy_path_temporal_ablation.sh`
- Create by runner: `doc/traces/gpu-contention/copy-path-temporal/<timestamp>/`

**Interfaces:**
- Consumes: selected H1 cases.
- Produces: separate D2D/background CUPTI CSV/meta files and `.nsys-rep` files for representative cases, with trace analysis including P2P activity duration, previous-activity gaps, scoped channel key, and overlap with background memcpy records.

- [x] **Step 1: Run CUPTI traces for selected points.**

  Enable the existing preload tracer only for the selected original D2H and P2P cases. Verify each three-GPU allpairs trace has `6 × (10+300)=1860` P2P records when the victim is the standard workload, zero dropped records, and channel keys scoped by PID/device/context/type/ID. Kernel replacement backgrounds may have no memcpy activity; record that limitation rather than treating it as zero GPU activity.

- [x] **Step 2: Run Nsight Systems traces for the same representative points.**

  Capture D2D/background process timelines without applying Nsight Systems to the complete matrix. Record the exact command, CUDA connection environment, GPU set, and output path; compare activity duration against previous-activity gaps.

- [x] **Step 3: Analyze and verify trace artifacts.**

  Run the temporal analyzer with trace paths and confirm overlap is reported only for records in the matching device/process scope, dropped-record metadata is zero, and no `channelID` is promoted to a physical CE name.

### Task 6: Implement and execute H3 minimal source-local diagnostic

**Files:**
- Create: `src/cuda_copy/minimal_source_pair_bw.cu`
- Create: `scripts/run_minimal_source_diagnostic.sh`
- Create: `scripts/analyze_minimal_source_diagnostic.py`
- Create: `tests/cuda_copy/test_minimal_source_diagnostic_cli.sh`
- Modify: `scripts/build_cuda_copy.sh`
- Create by runner: `doc/results/gpu-contention/minimal-source-diagnostic/<timestamp>/`

**Interfaces:**
- Consumes: two directed edges `GPU0->GPU1` and `GPU0->GPU2`, `255M` victim payload, edge order, shared/independent stream mode, and background device subsets.
- Produces: per-size/per-background/per-topology summaries with source and edge timing, clean/D2H bandwidth, D2H per-device metrics, and optional CUPTI/NSys traces.

- [x] **Step 1: Write the failing CLI/smoke test.**

  Assert the benchmark accepts `--edgeOrder=0->1,0->2|0->2,0->1`, `--streamMode=shared|independent`, `--streamDependency=none|source-chain`, `--repeats`, `--size`, and the runner accepts background sets `none`, `0`, `1`, `2`, and `all`. Run a two-repeat smoke case and assert two edge results plus one source aggregate.

- [x] **Step 2: Run the test and record the expected failure.**

  Run `bash tests/cuda_copy/test_minimal_source_diagnostic_cli.sh`; it must fail because the two-edge benchmark and runner are absent.

- [x] **Step 3: Implement the two-edge benchmark.**

  Allocate source buffer on GPU0 and destination buffers on GPU1/GPU2, enable peer access, use one stream for `shared`, one stream per edge for `independent`, and event dependencies for `independent + source-chain`. Keep warmup fixed at 10, measure per-edge events and source aggregate, and serialize edge order, queue position, stream identity, and device list.

- [x] **Step 4: Implement the runner and analyzer.**

  Run clean and D2H background subsets sequentially for `64K,128K,256K,512K,1M,2M,4M`, both stream modes, both edge orders, and three repetitions. Use 4M/64K as the existing bracket and explicitly label the last immune and first drop sizes from measured results.

- [x] **Step 5: Run the H3 matrix and selected traces.**

  Execute the full minimal matrix without profiling, then collect H2 traces at the last immune and first drop sizes (or 64K and 4M if no interior pair exists). Record whether D2H must share GPU0 with the P2P source and whether only the second shared-stream position is affected.

### Task 7: Run H4 Nsight Compute property checks

**Files:**
- Create by command: `doc/traces/gpu-contention/copy-path-temporal/ncu/`
- Modify: `doc/reports/gpu-internal-copy-contention-results.md`

**Interfaces:**
- Consumes: bounded `copy_path_background --iterations=N` runs and the installed NCU metric query.
- Produces: standalone NCU reports for streaming HBM read/write and 4 MiB L2 candidate, or an exact environment record if NCU is unavailable or cannot collect counters.

- [x] **Step 1: Query the installed NCU version and metrics.**

  Run `which ncu`, `ncu --version`, and `ncu --query-metrics`. Select only metrics actually listed by this installation; do not hard-code metrics from another CUDA release.

- [x] **Step 2: Profile bounded kernel cases.**

  Run one bounded streaming-read, streaming-write, and L2-candidate case with the smallest metric set that reports DRAM bytes/L2 traffic and kernel duration. Keep NCU separate from formal victim performance runs.

- [x] **Step 3: Validate the intended path distinction.**

> H4 的 metric query 和 bounded profile 均被 `ERR_NVGPUCTRPERM` 阻断；已记录精确
> 工具版本、命令和错误，未将不可观测的 DRAM/L2 counter 当作实验结论。

  Confirm whether the 4 MiB candidate has materially lower DRAM traffic than streaming read and whether continuous/throttled kernels have long active duration. If counters are unavailable, record that H4 is unobservable rather than inferring L2 residency from working-set size.

### Task 8: Update the Stage H decision document and verify

**Files:**
- Create: `doc/results/gpu-contention/copy-path-temporal/stage-h-decision.md`
- Modify: `doc/reports/gpu-internal-copy-contention-results.md`
- Modify: `docs/superpowers/plans/2026-08-28-stage-h-temporal-3gpu.md`

**Interfaces:**
- Consumes: H0/H1 analysis, H2 traces, H3 results, and H4 NCU evidence or its recorded limitation.
- Produces: the final public-tool boundary in section 8.10/8.11 and a linked decision table.

- [x] **Step 1: Generate the literal H0/H1/H2/H3/H4 decision table.**

  Include per-case background rate, wall/gpu duty, duration/cadence/gap, overlap, P2P slow count, topology, and data-quality status. Keep `bandwidth-matched`, `duty-matched`, and `saturated` distinct.

- [x] **Step 2: Apply the report’s Stage H decision rules.**

  If only time-comparable original D2H + P2P CE reproduces the effect, stop at the D2H host/PCIe and P2P CE remote-copy admission/path combination. If a continuous replacement reproduces it, retain the corresponding broader source path. Do not use low-duty Stage G negatives to eliminate a path.

- [x] **Step 3: Update the report, index, and plan checkboxes.**

  Add raw-result links, exact commands, limitations, and the three-GPU one-stream/two-copy definition. Mark only executed and verified plan steps complete.

- [ ] **Step 4: Run final verification.**

  Run `bash scripts/build_cuda_copy.sh`, all Stage H synthetic/CLI tests, `python3 -m py_compile` on new analyzers, `bash -n` on new runners, `git diff --check`, and validate every formal summary has the expected row count and zero non-pass rows before claiming completion.
