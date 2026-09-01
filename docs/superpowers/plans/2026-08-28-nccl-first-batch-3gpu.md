# NCCL First-Batch 3-GPU Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the NCCL experiment design through version/build validation and the first clean/D2H/H2D macro-regime measurements on the currently available three V100 GPUs.

**Architecture:** Keep `/home/songxb26/HyLink/nccl` untouched because it is a dirty v2.29.2 experimental `.gpu` tree. Use the existing `third_party/nccl` v2.31.2 source as the documented `.sys` build input, build `third_party/nccl-tests` against that library, and reuse the already validated CUDA host-copy background binary through a new current-repository runner. Save all NCCL raw outputs under `doc/results/nccl-contention/` and keep clean-before/D2H/clean-after measurements separate.

**Tech Stack:** CUDA 12.6.85, nvcc, NCCL 2.31.2, nccl-tests, CUDA Runtime API, Bash, Python 3, Nsight Systems only after macro validation.

**Spec:** `doc/reports/nccl-d2h-interference-experiment-plan.md`

## Global Constraints

- Current hardware is exactly GPU 0, GPU 1, and GPU 2; all NCCL results are explicitly three-GPU adaptations of the four-GPU design.
- `third_party/nccl` source commit is `v2.31.2-1`; its only current dirty change is the V100-only `NVCC_GENCODE` build setting in `makefiles/common.mk`.
- `/home/songxb26/HyLink/nccl` is `v2.29.2-1-dirty` and has an existing `.gpu` fence patch; do not overwrite, reset, or use it as the `.sys` baseline.
- Use `NCCL_ALGO=RING`, `NCCL_PROTO=SIMPLE`, `CUDA_VISIBLE_DEVICES=0,1,2` for the first matrix.
- Keep nccl-tests profiling, Inspector, and performance truth in separate runs; never mix library builds in one result directory.
- The first running matrix is a 3-GPU L0/B-stage calibration: 64 MiB, warmup 5/20, measured 20/100/500, clean-before and D2H-all; clean-after is required for D2H cases.
- Do not claim a root cause from the first matrix; first validate library identity, correctness, algorithm/protocol/channel stability, and recovery.

---

### Task 1: Capture build identities and preserve dirty external state

**Files:**
- Create: `doc/results/nccl-contention/environment/20260828T.../`
- Create: `doc/results/nccl-contention/environment/official-nccl-dirty-diff.patch`
- Create: `doc/results/nccl-contention/environment/third-party-nccl-diff.patch`

- [ ] **Step 1:** Record GPU count/topology, CUDA, driver, NCCL_HOME, source git descriptions, status, and existing dirty diffs.
- [ ] **Step 2:** Verify the dirty official v2.29.2 `.gpu` tree is not selected by the new runner.
- [ ] **Step 3:** Record the three-GPU adaptation and all executable/library paths before building.

### Task 2: Build and validate the v2.31.2 V100 `.sys` stack

**Files:**
- Create by command: `build/nccl-v2.31.2-sm70-sys/`
- Create by command: `build/nccl-tests-v2.31.2-sm70/`
- Create: `scripts/build_nccl_v100_sys.sh`
- Create: `tests/nccl/test_nccl_build_contract.sh`

- [ ] **Step 1:** Build NCCL v2.31.2 with only `compute_70 -> sm_70`, preserving `.sys` `fence_acq_rel_sys()` in `prims_simple.h`.
- [ ] **Step 2:** Build third_party nccl-tests AG/AR/RS binaries against that library.
- [ ] **Step 3:** Inspect `ldd`, NCCL runtime version, and cubin/SASS for the expected sm70/system-fence build identity.
- [ ] **Step 4:** Run 3-GPU correctness smoke tests for AG, AR, RS at 4 MiB and 64 MiB.

### Task 3: Add the controlled NCCL macro runner/analyzer

**Files:**
- Create: `scripts/run_nccl_first_batch.sh`
- Create: `scripts/analyze_nccl_first_batch.py`
- Create: `tests/nccl/test_nccl_runner_contract.sh`
- Create: `tests/nccl/test_nccl_analysis_contract.sh`

- [ ] **Step 1:** Add help/argument validation for `--devices`, collective list, sizes, warmups, iterations, repetitions, and output root.
- [ ] **Step 2:** Implement ready/stop background lifecycle using `host_copy_background`, with exact PID cleanup and clean-after for D2H cases.
- [ ] **Step 3:** Record `time`, `algbw`, `busbw`, status, build identity, background JSON, and command/environment per case.
- [ ] **Step 4:** Analyze paired clean-before/clean-after, recovery error, median/mean/min/max, and D2H slowdown without merging regimes.

### Task 4: Execute L0 and B-stage three-GPU calibration

**Files:**
- Create by runner: `doc/results/nccl-contention/regime-sweep/<timestamp>/`
- Modify: `doc/reports/nccl-d2h-interference-experiment-plan.md`
- Create: `doc/results/nccl-contention/regime-sweep/<timestamp>/README.md`

- [ ] **Step 1:** Run AG/AR/RS at 64 MiB across warmup 5/20 and iterations 20/100/500, clean and D2H-all, with at least 3 independent repetitions.
- [ ] **Step 2:** Verify clean-after recovery and that algorithm/protocol/channel configuration is unchanged.
- [ ] **Step 3:** Select short-burst and steady-state regimes from measured window and operation timing, not aggregate bandwidth alone.
- [ ] **Step 4:** Write the first NCCL result summary, exact limitations, and next macro matrix without claiming fence causality.

### Task 5: Run final verification

- [ ] `bash scripts/build_nccl_v100_sys.sh`
- [ ] `bash tests/nccl/test_nccl_build_contract.sh`
- [ ] `bash tests/nccl/test_nccl_runner_contract.sh`
- [ ] `bash tests/nccl/test_nccl_analysis_contract.sh`
- [ ] `python3 -m py_compile scripts/analyze_nccl_first_batch.py`
- [ ] `bash -n scripts/build_nccl_v100_sys.sh scripts/run_nccl_first_batch.sh`
- [ ] `git diff --check`
- [ ] Validate all formal summary rows have expected counts and zero non-pass rows.
