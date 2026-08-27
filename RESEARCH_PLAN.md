# CUDA Copy Interference Research Plan

> **For agentic workers:** This is the living research specification and execution plan for the first experiment stage. Keep the checkboxes updated as experiments are completed.

**Goal:** First establish the two-GPU control, then measure whether concurrent pinned-memory D2H or H2D traffic interferes with CUDA peer D2D traffic across the available four Tesla V100-SXM2 GPUs, with ring and allpairs traffic as separate cases.

**Architecture:** Use CUDA Runtime microbenchmarks for peer D2D traffic and sustained host-device traffic. The two-GPU executable remains the control; a multi-GPU executable adds ring/allpairs traffic. Shell runners launch controlled scenarios, capture the environment and raw JSON/CSV output, and save profiler traces separately from summary reports.

**Tech Stack:** CUDA 12.6.85, CUDA Runtime API, `nvcc`, Nsight Systems 2024.7.1, Nsight Compute 2024.3.2, `nvidia-smi`, Bash, and Python only for result parsing if a parser is needed.

**Spec:** This file is both the approved research design and the living execution plan, following the repository convention requested by the user.

## Global Constraints

- The first stage targets only GPU 0 and GPU 1: Tesla V100-SXM2-32GB connected by NV2 NVLink.
- The current host uses Driver 580.178.04, `nvcc` 12.6.85, and NCCL 2.29.2 at `/home/songxb26/HyLink/nccl/build`.
- The four-GPU full-mesh result from the referenced forum cannot be reproduced directly with two GPUs; two-way D2D is an explicit lower-pressure negative control, not evidence about four-GPU behavior.
- The initial copy size is 255 MiB to match the reference experiment. Size and stream sweeps come only after the fixed-size baseline is stable.
- Host transfers use pinned memory and `cudaMemcpyAsync`; peer transfers use `cudaMemcpyPeerAsync` with peer access enabled in both directions.
- Every run records its command line, git revision, device/topology information, CUDA/Driver versions, synchronization mode, and raw measurements.
- Source belongs under `src/`, scripts under `scripts/`, build outputs under `build/`, and reports/raw results/traces under `doc/`.
- The initial stage does not make a causal claim about NCCL. NCCL collectives are a later, separate stage because their kernels and synchronization paths add independent variables.

## Current Four-GPU CUDA Extension

The available machine now exposes four Tesla V100-SXM2-32GB GPUs. The current
snapshot is Driver 580.178.04, CUDA 12.6.85, and an NV2 link between every GPU
pair; all four GPUs are on NUMA node 0. NCCL remains reserved for a later stage.

The four-GPU D2D executable extends the same `cudaMemcpyPeerAsync` model:

- `--pattern=ring` issues four directed copies: `0→1→2→3→0` for the default device list.
- `--pattern=allpairs` issues all twelve ordered pairs, excluding self-copies.
- `--deviceList=0,1,2,3` selects visible devices; `--devices=4` expands to `0..3`.
- `--repeats=20` is measured by default, while warmup is fixed at 10 iterations.
- `--size=255M` is the default payload. One source buffer is allocated per GPU and one destination buffer per ordered edge, so concurrent allpairs copies do not race on a destination allocation.
- A source GPU owns one CUDA stream. All of its outgoing copies are queued on that stream, and aggregate bandwidth is computed with the maximum elapsed time across source streams.

The matrix runner covers `ring/allpairs × none/d2h/h2d`, records environment and
commands, and stops/reaps the four-GPU background process through ready/stop
markers. The first quick matrix is a code-path validation; it is not yet the
fixed-size, repeated experiment report.

The hardware-specification baseline, allpairs byte accounting, and the working
data paths for D2D/D2H/H2D are documented in
[`doc/reports/v100-32gb-memory-hierarchy-and-copy-path-analysis.md`](doc/reports/v100-32gb-memory-hierarchy-and-copy-path-analysis.md).

## Research Questions and Hypotheses

1. Does D2H background traffic reduce one-way D2D bandwidth between the two V100s?
2. Does bidirectional D2D traffic show a larger effect than one-way D2D traffic?
3. Is the effect different when background traffic runs on GPU 0, GPU 1, or both GPUs?
4. Is D2H behavior different from H2D behavior at the same transfer size and copy cadence?
5. Do Copy Engine activity, NVLink traffic, stream synchronization time, or host NUMA placement change together with the measured D2D bandwidth?

The working hypothesis is that the two-GPU experiment may show little or no degradation, matching the two-GPU control in the reference report. If a degradation appears, its GPU-local directionality will be tested before attributing it to NVLink or memory arbitration.

## Experiment Matrix

### Fixed first-pass configuration

- D2D payload: 255 MiB
- D2D warmup: 10 iterations
- D2D measurement: 20 repetitions by default; copies are queued without host synchronization after each repetition
- Background payload: 255 MiB
- Background transfer: one pinned host buffer and one device buffer per participating GPU
- Background loop: `cudaMemcpyAsync` followed by `cudaStreamSynchronize`, matching the reference semantics
- Repetitions: 5 independent runs per scenario after one dry run
- Primary result: aggregate D2D GB/s and per-direction D2D GB/s

### Scenarios

| ID | D2D traffic | Background traffic | Purpose |
| --- | --- | --- | --- |
| B0 | GPU0 → GPU1 | none | One-way baseline |
| B1 | GPU1 → GPU0 | none | Reverse one-way baseline |
| B2 | GPU0 ↔ GPU1 | none | Bidirectional baseline |
| D0 | GPU0 → GPU1 | D2H on GPU0 and GPU1 | Main D2H comparison |
| D1 | GPU1 → GPU0 | D2H on GPU0 and GPU1 | Reverse-direction D2H comparison |
| D2 | GPU0 ↔ GPU1 | D2H on GPU0 and GPU1 | Bidirectional D2H comparison |
| H0 | GPU0 → GPU1 | H2D on GPU0 and GPU1 | Main H2D comparison |
| H1 | GPU1 → GPU0 | H2D on GPU0 and GPU1 | Reverse-direction H2D comparison |
| H2 | GPU0 ↔ GPU1 | H2D on GPU0 and GPU1 | Bidirectional H2D comparison |

The next matrix, only if the first matrix produces a stable difference or an interesting asymmetry, adds background traffic on GPU 0 only and GPU 1 only. It then adds 4 MiB, 64 MiB, and 1 GiB payloads while keeping the D2D payload fixed.

## Planned Program Interfaces

### `src/cuda_copy/d2d_peer_bw.cu`

The executable will expose the following stable interface:

```text
./d2d_peer_bw \
  --mode=unidirectional|bidirectional \
  --srcDev=0 --dstDev=1 \
  --size=255M --repeats=20 --output=<json>
```

- `unidirectional` issues one peer copy per repetition.
- `bidirectional` issues GPU0→GPU1 and GPU1→GPU0 on one stream per direction.
- `srcDev` defaults to 0, `dstDev` defaults to 1, and `repeats` defaults to 20.
- Warmup is fixed at 10 iterations and is not a user-settable option.
- Peer access is checked and enabled before allocation; unsupported access fails before the measurement.
- CUDA events measure each logical direction. Aggregate bandwidth uses total transferred bytes divided by the maximum elapsed time across participating streams.
- The output includes bytes, iterations, elapsed milliseconds, aggregate bandwidth, per-direction bandwidth, device names, and peer-access status.

### `src/cuda_copy/host_copy_background.cu`

The executable will expose the following stable interface:

```text
./host_copy_background \
  --devList=0,1 --direction=d2h|h2d \
  --size=255M --readyFile=<path> --stopFile=<path> \
  --output=<json>
```

- `devList` defaults to `0,1` and accepts any comma-separated list of visible GPU indices.
- Direction defaults to `d2h`; `h2d` uses the same allocation and synchronization path in reverse.
- Each device gets one pinned host allocation, one device allocation, and one CUDA stream.
- The process writes the ready marker only after allocation and warmup succeed.
- The process loops forever until SIGINT/SIGTERM or a stop marker, and reports completed bytes and aggregate host-transfer bandwidth.
- Signal handling and a shell `trap` must release background processes so interrupted experiments do not leave GPU allocations behind.

## Repository Layout

```text
RESEARCH_PLAN.md
doc/
  reports/                 # Human-readable experiment reports
  results/                 # Per-run JSON and matrix CSV files
  traces/                  # Nsight Systems and other profiler outputs
  environment/             # Captured nvidia-smi/toolchain snapshots
src/
  cuda_copy/
    copy_common.cuh
    d2d_peer_bw.cu
    d2d_multi_peer_bw.cu
    host_copy_background.cu
  nccl_collectives/        # Reserved for the later NCCL stage
scripts/
  collect_environment.sh
  build_cuda_copy.sh
  run_4gpu_copy_matrix.sh
  run_copy_matrix.sh
  profile_copy.sh
tests/
  cuda_copy/
    test_cli.sh
    test_smoke.sh
    test_4gpu_cli.sh
    test_4gpu_smoke.sh
build/
  cuda_copy/               # Generated executables and objects
  nccl_collectives/        # Reserved for later NCCL executables
```

## Execution Tasks

### Task 1: Capture and freeze the experiment environment

**Files:**

- Create: `scripts/collect_environment.sh`
- Create: `doc/environment/README.md`

- [ ] Record `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi nvlink -s`, `nvcc --version`, `nsys --version`, `ncu --version`, `nvprof --version`, `lscpu`, and `numactl --hardware`.
- [ ] Record `CUDA_VISIBLE_DEVICES`, `NCCL_HOME`, the repository git revision, and the NCCL header version.
- [ ] Verify both GPUs are idle before each matrix run; do not change persistence, application clocks, or system-wide settings.

**Acceptance:** A fresh environment snapshot can be regenerated with one script and identifies the exact GPU order used by every result.

### Task 2: Build the D2D benchmark

**Files:**

- Create: `src/cuda_copy/copy_common.cuh`
- Create: `src/cuda_copy/d2d_peer_bw.cu`
- Create: `scripts/build_cuda_copy.sh`
- Create: `build/cuda_copy/d2d_peer_bw`

- [x] Add checked CUDA error handling, argument validation, peer-access validation, device allocation, warmup, event timing, and JSON output.
- [x] Build with `nvcc -std=c++17 -O2` and place all generated output under `build/cuda_copy/`.
- [x] Run `--help`, invalid-argument checks, one-way GPU0→GPU1, reverse one-way, and bidirectional smoke tests.
- [x] Run `compute-sanitizer` if available before accepting the first result set.

**Acceptance:** All three D2D modes complete without CUDA errors and report non-zero, internally consistent byte counts and elapsed times.

### Task 2A: Extend D2D traffic to four GPUs

**Files:**

- Create: `src/cuda_copy/d2d_multi_peer_bw.cu`
- Modify: `scripts/build_cuda_copy.sh`
- Create: `build/cuda_copy/d2d_multi_peer_bw`

- [x] Add ring and allpairs patterns with explicit directed-copy counts.
- [x] Add device-list and `--devices=N` selection while keeping warmup fixed at 10 and the default payload/repetition values unchanged.
- [x] Allocate independent destination buffers for all ordered edges and report per-source plus aggregate D2D bandwidth.
- [x] Build and run the four-GPU CLI, real-device smoke test, and `compute-sanitizer` memcheck.

**Acceptance:** On the four V100s, ring reports four directions and allpairs reports twelve directions without CUDA errors or memory-checker findings.

### Task 2B: Automate four-GPU copy/background cases

**Files:**

- Create: `scripts/run_4gpu_copy_matrix.sh`
- Create: `doc/results/README.md`
- Create: `tests/cuda_copy/test_4gpu_cli.sh`
- Create: `tests/cuda_copy/test_4gpu_smoke.sh`

- [x] Cover `none`, `d2h`, and `h2d` background cases for both D2D patterns.
- [x] Save an environment snapshot, exact commands, raw logs, JSON measurements, and `summary.csv` below a unique result directory.
- [x] Stop and reap background workers after each case and on script exit or interruption.
- [x] Run a 1 MiB/one-repeat six-case validation matrix and confirm numeric D2D/background results.
- [ ] Run the default 255 MiB/20-repeat matrix repeatedly enough to write the four-GPU comparison report.

**Acceptance:** One command can reproduce the six-case four-GPU measurement matrix; scientific conclusions remain pending repeated fixed-size runs and profiling.

### Task 2C: Establish the V100 theoretical bandwidth baseline

**Files:**

- Create: `doc/reports/v100-32gb-memory-hierarchy-and-copy-path-analysis.md`
- Reference: `doc/results/4gpu/default-4gpu-validation/`

- [x] Record the official V100 32GB HBM2, HBM interface, L2, SM, L1/shared-memory,
  and NVLink specifications with primary NVIDIA sources.
- [x] Convert the current 255 MiB four-GPU allpairs result into per-GPU logical
  D2D traffic and estimated local HBM read/write traffic.
- [x] Describe the expected D2D/NVLink, D2H/PCIe, and H2D/PCIe paths, including
  the distinction between SM work and Copy Engine work.
- [x] Mark cache policy, exact CE count, and internal arbitration details as
  runtime/profiler questions rather than undocumented fixed specifications.

**Acceptance:** The theory document distinguishes official peak specifications,
measured results, derived estimates, and hypotheses that still require counters or
timeline evidence.

### Task 3: Build the D2H/H2D background benchmark

**Files:**

- Create: `src/cuda_copy/host_copy_background.cu`
- Modify: `src/cuda_copy/copy_common.cuh`

- [x] Implement pinned host allocation and one stream per GPU.
- [x] Implement both `d2h` and `h2d` using the same allocation size and synchronization cadence.
- [x] Add ready/stop markers, an infinite loop with signal-based shutdown, per-device byte counters, and clean shutdown.
- [x] Verify independently that D2H and H2D sustain traffic on both GPUs before combining them with D2D.

**Acceptance:** Each direction runs for a bounded duration, exits cleanly, and reports a positive aggregate transfer rate without memory leaks or orphaned processes.

### Task 4: Automate the fixed-size experiment matrix

**Files:**

- Create: `scripts/run_copy_matrix.sh`
- Create: `doc/results/README.md`

- [ ] Run B0–B2 without background traffic.
- [ ] Run D0–D2 with the D2H process synchronized through its ready marker.
- [ ] Run H0–H2 with the H2D process synchronized through its ready marker.
- [ ] Use a unique run directory per repetition containing command lines, environment snapshot reference, D2D JSON, background JSON, and process logs.
- [ ] Use a shell trap to stop and reap background processes on success, failure, or interruption.
- [ ] Generate one matrix CSV containing scenario, repetition, D2D aggregate GB/s, per-direction GB/s, background GB/s, and exit status.

**Acceptance:** The complete first-pass matrix can be rerun from one command and produces no result file without the corresponding command and environment metadata.

### Task 5: Add profiler and hardware-counter runs

**Files:**

- Create: `scripts/profile_copy.sh`
- Create: `doc/reports/profiling-notes.md`

- [ ] Profile one baseline and one D2H scenario with Nsight Systems, tracing CUDA API and memory-copy activity.
- [ ] Capture concurrent `nvidia-smi dmon -s putc --gpm-metrics=2,3,20,21,60,61` output for SM activity, SM occupancy, PCIe RX/TX, and NVLink RX/TX; preserve `-` values as an explicit unsupported-metric result.
- [ ] Capture `nvidia-smi nvlink -s` and `nvidia-smi nvlink -gt d` before and after the run; record `N/A` counters rather than treating them as zero.
- [ ] Use Nsight Compute only for metrics that are meaningful for any helper kernels; do not label a memcpy timeline as SM occupancy.
- [ ] Record whether the observed difference is in D2D copy duration, stream synchronization, NVLink activity, or host-transfer throughput.

**Acceptance:** Each profile has a matching scenario ID and can be compared against the corresponding unprofiled run without changing the primary benchmark configuration.

### Task 6: Write the first result report

**Files:**

- Create: `doc/reports/two-gpu-cuda-copy-interference.md`

- [ ] Report median, minimum, maximum, and relative change against the matching no-background baseline.
- [ ] Separate one-way, bidirectional, D2H, and H2D conclusions.
- [ ] Explicitly state whether the two-GPU result is a positive effect, a negative control, or inconclusive because of variance.
- [ ] Do not generalize a two-GPU result to the four-GPU allpairs case.
- [ ] List the next experiment only after identifying which observation justifies it: per-GPU background placement, payload sweep, stream sweep, NUMA placement, or profiling.

**Acceptance:** The report can be read without terminal access and still contains enough metadata to reproduce every plotted or tabulated number.

## Measurement and Interpretation Rules

- Use the same D2D payload, warmup, iteration count, and synchronization mode for each compared pair.
- Compare each background scenario only with its matching D2D baseline and report both absolute GB/s and relative change.
- Treat a result as stable only after five repetitions show a consistent direction; a single outlier is not a mechanism.
- A lack of degradation on two GPUs is a valid result and should be recorded as a boundary condition.
- Copy Engine and NVLink activity are evidence about resource overlap, not by themselves proof of causality.
- For pure memcpy, report SM activity as an auxiliary observation. SM occupancy becomes a primary metric in the later NCCL-kernel stage.

## Later Directions

After the two-GPU CUDA report is complete, extend in this order:

1. Background placement: GPU0 only versus GPU1 only.
2. Payload, stream, synchronization, and NUMA-placement sweeps.
3. Additional Copy Engine/NVLink counter correlation.
4. NCCL allgather, reducescatter, and allreduce using the local NCCL build.
5. NCCL-specific SM occupancy, synchronization, Copy Engine, and protocol analysis.
6. Repeat the fixed-size four-GPU ring/allpairs matrix and correlate it with
   Copy Engine/NVLink traces before drawing conclusions about the forum case.
