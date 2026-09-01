# Stage F/G Three-GPU Copy-Contention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the Stage F work-queue/HW-channel investigation and, only when its decision table provides a usable boundary, execute the minimum Stage G copy-path ablation on the currently available three V100 GPUs.

**Architecture:** Keep the historical four-GPU experiments unchanged and add a dedicated Stage F runner with a default `deviceList=0,1,2`. Because three-GPU allpairs has two outgoing edges per source, collapse the old two-stream one-vs-two topology into one source stream carrying two consecutive P2P copies; use two independent edge streams and the same two streams with source-chain dependencies as the comparison topologies. Record the adaptation explicitly so no three-GPU result is presented as a four-GPU equivalent.

**Tech Stack:** CUDA 12.6, nvcc, CUDA Runtime API, CUPTI Activity API (`CUpti_ActivityMemcpy5` and `CUpti_ActivityMemcpyPtoP4`), Bash, Python 3, SQLite/CSV-compatible JSON artifacts, Tesla V100-SXM2-32GB.

**Spec:** `doc/reports/gpu-internal-copy-contention-results.md`, sections 8.8 Stage F and 8.9 Stage G.

## Global Constraints

- Available hardware is exactly `GPU 0, GPU 1, GPU 2`, all Tesla V100-SXM2-32GB, compute capability 7.0.
- Stage F D2D workload uses three-GPU allpairs, D2D size `255M`, warmup fixed at `10`, and measured repeats `300`; 20 repeats are a short-batch regression only.
- Stage F connection values are `unset,1,2,4,8,16,32`; `CUDA_DEVICE_MAX_CONNECTIONS` is set before either CUDA process/context initializes.
- Do not use `CUDA_DEVICE_MAX_COPY_CONNECTIONS`, because the target is Volta compute capability 7.0.
- Every Stage F case records D2D aggregate/per-source throughput, background aggregate/per-GPU throughput, command, environment, exit status, raw JSON, and logs.
- `channelID` and `channelType` are reported as CUPTI HW-channel fields only; never label them as physical CE instance numbers.
- The three-GPU topology adaptation is: `single-two-copy` = one source stream with two consecutive allpairs P2P copies; `edge-independent` = one active stream per outgoing edge; `edge-source-chain` = the same independent edge streams with source-chain event dependencies.
- Stage G is gated on the Stage F decision table and uses the same three-GPU victim/controls; it stops after the minimum replacement matrix in the report has been collected.
- Results go below `doc/results/gpu-contention/`; source goes below `src/cuda_copy/`; build products go below `build/cuda_copy/`; scripts go below `scripts/`; tests go below `tests/cuda_copy/`.

---

### Task 1: Define the three-GPU Stage F runner contract

**Files:**
- Create: `scripts/run_work_queue_channel_diagnostic.sh`
- Test: `tests/cuda_copy/test_work_queue_channel_cli.sh`
- Modify: `doc/reports/gpu-internal-copy-contention-results.md` only after the runner contract is verified

**Interfaces:**
- Consumes: `build/cuda_copy/d2d_multi_peer_bw`, `build/cuda_copy/host_copy_background`.
- Produces: a runner accepting `--deviceList`, `--connectionValues`, `--topologies`, `--scenarios`, `--runs`, `--repeats`, `--size`, `--trace`, and `--outputRoot`; one CSV row per case with `topology`, `connectionValue`, `scenario`, `devices`, `d2dSize`, `backgroundSize`, throughput, exit status, and artifact paths.

- [x] **Step 1: Write the failing CLI contract test.**

  The test must execute the real runner with `--help`, a one-case 2-repeat three-GPU smoke matrix, and an invalid four-GPU device list. It must assert the generated environment contains `deviceList=0,1,2`, the child command contains `--deviceList=0,1,2`, `single-two-copy` maps to `--streamsPerSource=1`, `edge-independent` maps to `--streamMode=per-edge --streamsPerSource=3`, and the summary has one header plus one `pass` row.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_work_queue_channel_cli.sh`. It must fail because `scripts/run_work_queue_channel_diagnostic.sh` does not yet exist or does not expose the contract.

- [x] **Step 3: Implement the minimal runner.**

  Validate exactly three unique device indices, default to `0,1,2`, validate topology names and connection values, launch background and D2D as separate processes, set/unset `CUDA_DEVICE_MAX_CONNECTIONS` before each child, use ready/stop marker files, and write a fresh result directory with `environment.txt`, `summary.csv`, one case directory, `command.txt`, `d2d.json`, `background.json`, and logs. For the three-GPU mapping use these exact child arguments:

  ```text
  single-two-copy   --pattern=allpairs --deviceList=0,1,2 --streamsPerSource=1
  edge-independent  --pattern=allpairs --deviceList=0,1,2 --streamMode=per-edge --streamsPerSource=3
  edge-source-chain --pattern=allpairs --deviceList=0,1,2 --streamMode=per-edge --streamsPerSource=3 --streamDependency=source-chain
  ```

  For `none`, run only D2D. For `d2h-all`, start `host_copy_background --devList=0,1,2 --direction=d2h --size=255M` and wait for its ready marker before launching D2D. Use `env -u CUDA_DEVICE_MAX_CONNECTIONS` for `unset`; otherwise use `env CUDA_DEVICE_MAX_CONNECTIONS=<value>`.

- [x] **Step 4: Run the CLI contract test and the three-GPU 2-repeat smoke case.**

  Run `bash tests/cuda_copy/test_work_queue_channel_cli.sh`. Expected output includes `PASS: three-GPU Stage F runner CLI contract`; inspect the smoke JSON and summary before continuing.

### Task 2: Add per-GPU background accounting

**Files:**
- Modify: `src/cuda_copy/host_copy_background.cu`
- Modify: `scripts/build_cuda_copy.sh`
- Test: `tests/cuda_copy/test_host_copy_per_device_json.sh`

**Interfaces:**
- Consumes: the existing per-worker atomic byte counters and measured elapsed interval.
- Produces: `host_copy_background` JSON fields `perDeviceBytes` and `perDeviceGBps`, each aligned with the requested device list, while preserving `aggregateGBps` and all existing fields.

- [x] **Step 1: Write the failing JSON behavior test.**

  Run a bounded three-GPU background process with `--readyFile`, `--stopFile`, `--output`, and a short externally controlled interval. Parse the real JSON and assert `perDeviceBytes` has three positive entries, `perDeviceGBps` has three numeric entries, and their byte sum equals `totalBytes`.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_host_copy_per_device_json.sh`; it must fail because the current JSON has aggregate bytes only.

- [x] **Step 3: Implement per-device byte/rate serialization.**

  Keep the measured start/stop protocol unchanged. Read each `Counter` once at shutdown, serialize the aligned arrays in device-list order, calculate each rate using the same elapsed seconds used for aggregate rate, and keep `totalBytes` equal to the sum of per-device bytes.

- [x] **Step 4: Build and rerun the test.**

  Run `bash scripts/build_cuda_copy.sh` followed by `bash tests/cuda_copy/test_host_copy_per_device_json.sh`; both must exit zero.

### Task 3: Execute and summarize Stage F1 connection scans

**Files:**
- Create: `scripts/analyze_work_queue_channel.py`
- Create: `tests/cuda_copy/test_work_queue_channel_analysis.sh`
- Create by runner: `doc/results/gpu-contention/work-queue-channel/<timestamp>/`

**Interfaces:**
- Consumes: Stage F `summary.csv` and per-case D2D/background JSON files.
- Produces: a read-only JSON/Markdown summary grouped by topology, connection value, and scenario, including mean/min/max aggregate D2D, per-source D2D, aggregate/per-GPU background, and the three-GPU adaptation note.

- [x] **Step 1: Write the failing synthetic analysis test.**

  Create a temporary two-case CSV/JSON fixture in the test itself: one clean row and one D2H row with three per-device background rates. Assert the analyzer emits the literal topology name, connection value, aggregate means, and a `threeGpuAdaptation` field explaining that the single-stream topology contains two consecutive copies per source.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_work_queue_channel_analysis.sh`; it must fail because the analyzer is absent.

- [x] **Step 3: Implement the analyzer.**

  Parse CSV with Python’s `csv.DictReader`, load referenced JSON read-only, aggregate only `status=pass` rows, preserve `unset` ordering before numeric connection values, and write JSON to stdout or `--output`. Reject missing required columns and mismatched per-device array lengths.

- [x] **Step 4: Run the analyzer test and the complete F1 matrix.**

  Run `bash tests/cuda_copy/test_work_queue_channel_analysis.sh`, then:

  ```bash
  bash scripts/run_work_queue_channel_diagnostic.sh \
    --deviceList=0,1,2 \
    --connectionValues=unset,1,2,4,8,16,32 \
    --topologies=single-two-copy,edge-independent,edge-source-chain \
    --scenarios=none,d2h-all --repeats=300 --runs=3
  ```

  Verify the expected matrix size is `7 × 3 × 2 × 3 = 126` case rows and every row is `pass` before interpreting the results.

### Task 4: Implement CUPTI memcpy HW-channel tracing and F2 analysis

**Files:**
- Create: `src/cuda_copy/cupti_memcpy_channel_trace.cpp`
- Create: `scripts/analyze_copy_channel_trace.py`
- Create: `tests/cuda_copy/test_copy_channel_analysis.sh`
- Modify: `scripts/build_cuda_copy.sh`
- Modify: `scripts/run_work_queue_channel_diagnostic.sh`
- Create by runner: `doc/traces/gpu-contention/work-queue-channel/<timestamp>/`

**Interfaces:**
- Consumes: `CUPTI_ACTIVITY_KIND_MEMCPY` and `CUPTI_ACTIVITY_KIND_MEMCPY2` activity records from the D2D or background child process.
- Produces: a preloadable `libcupti_memcpy_channel_trace.so` and CSV columns `pid,deviceId,contextId,streamId,channelID,channelType,copyKind,srcDeviceId,dstDeviceId,bytes,startNs,endNs,durationMs,correlationId,activityKind`; the analyzer emits channel/stream counts, slow-by-channel probabilities, dropped-record count, and per-kind totals.

- [x] **Step 1: Write the failing synthetic CSV analysis test.**

  Use a literal CSV fixture with one P2P fast row, one P2P slow row, and one D2H row on distinct channels. Assert the analyzer reports one slow P2P, one D2H, the correct slow channel, and no dropped records when the metadata says zero.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_copy_channel_analysis.sh`; it must fail because the analyzer is absent.

- [x] **Step 3: Implement the CUPTI preload library.**

  Register request/completion callbacks in a constructor, allocate callback buffers, enable both memcpy activity kinds, cast `CUPTI_ACTIVITY_KIND_MEMCPY` to `CUpti_ActivityMemcpy5` and `CUPTI_ACTIVITY_KIND_MEMCPY2` to `CUpti_ActivityMemcpyPtoP4`, and write one CSV row per activity under a mutex. Take the output path from `COPYBENCH_CUPTI_MEMCPY_OUTPUT`; flush with `cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)` in the destructor. Store the process ID in every row and set unavailable source/destination IDs to `-1` for non-P2P records.

- [x] **Step 4: Build, run the analyzer test, and run one 3-GPU CUPTI smoke trace.**

  Run `bash scripts/build_cuda_copy.sh`, build the shared tracer, run a 2-repeat `single-two-copy` clean and D2H case with `LD_PRELOAD`, and verify six P2P rows per measured topology plus D2H rows, valid numeric channel fields, and no CSV loss.

- [x] **Step 5: Run the F2 key cases.**

  Collect clean/D2H traces for `single-two-copy`, `edge-independent`, and `edge-source-chain` at connection values `unset`, `1`, and `8`. Analyze each trace with the analyzer, report CUPTI overhead against the corresponding no-tracer JSON, and keep channel identity scoped by `(pid,deviceId,contextId,channelType,channelID)`.

### Task 5: Execute F3 context-boundary controls

**Files:**
- Create: `src/cuda_copy/d2d_with_background_bw.cu`
- Create: `scripts/run_context_boundary_diagnostic.sh`
- Create: `tests/cuda_copy/test_context_boundary_cli.sh`
- Modify: `scripts/build_cuda_copy.sh`
- Create by runner: `doc/results/gpu-contention/work-queue-channel/context-boundary/<timestamp>/`

**Interfaces:**
- Consumes: the same three-GPU D2D allpairs and D2H buffer/cadence parameters as F1.
- Produces: a same-process/same-primary-context benchmark with independent host worker threads, plus a two-process baseline; output includes D2D aggregate/per-source and background aggregate/per-device rates.

- [x] **Step 1: Write the failing context-mode CLI test.**

  Assert `--contextMode=two-process|same-process` is accepted by the runner, rejects unknown modes, defaults to the two-process baseline, and records the mode in the environment and JSON.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_context_boundary_cli.sh`; it must fail because the same-process benchmark/runner is absent.

- [x] **Step 3: Implement same-process execution.**

  Reuse the D2D allocation/stream topology and launch one D2H worker per device as host threads in the same process. Create the D2H streams after `cudaSetDevice` in those workers, keep one stream per worker, synchronize each background memcpy, and start/stop the D2D measured range only after all workers are ready. Serialize the mode, background per-device bytes/rates, D2D source results, and exit status.

- [x] **Step 4: Run F3 with matched background pressure.**

  Run `single-two-copy` and `edge-independent`, clean/D2H, for two-process and same-process modes, three repetitions each. If Volta MPS is operational and can be started/stopped without changing the host state, run MPS as a separately labeled optional case; otherwise record the exact inability and do not treat it as a missing result.

### Task 6: Build the Stage F decision table and gate Stage G

**Files:**
- Modify: `doc/reports/gpu-internal-copy-contention-results.md`
- Create: `doc/results/gpu-contention/work-queue-channel/stage-f-decision.md`

**Interfaces:**
- Consumes: F1 summary, F2 channel analyses/traces, F3 context results, and the three-GPU adaptation statement.
- Produces: a per-case decision table classified as `work-queue mapping confirmed`, `same HW channel/downstream stall more likely`, or `HW channel not observable`, without naming physical CE instances.

- [x] **Step 1: Generate literal per-case comparisons.**

  For each topology/connection/scenario, record D2D aggregate/per-source mean and range, background aggregate/per-device mean and range, P2P count, measured slow count when CUPTI is enabled, stream count, channel keys, and profiler overhead.

- [x] **Step 2: Apply the report’s Stage F decision rules.**

  Classify connection-dependent topology changes, same-channel fast/slow behavior, context-mode differences, or CUPTI unavailability exactly according to section 8.8.5. If the three-GPU topology collapse prevents a four-GPU claim, state that limitation in the decision table.

- [x] **Step 3: Update the report and decide the Stage G gate.**

  Append Stage F commands, result links, data-quality failures, and the narrowest supported conclusion to sections 8.8 and 9. Proceed to Task 7 only if F1/F2/F3 are complete or their unavailable portions are explicitly recorded and the remaining evidence distinguishes a CE/work-queue hypothesis from a memory/NVLink hypothesis enough to choose a minimum Stage G matrix.

### Task 7: Execute the gated minimum Stage G copy-path ablation

**Files:**
- Create: `src/cuda_copy/copy_path_background.cu`
- Create: `src/cuda_copy/p2p_kernel_bw.cu`
- Create: `scripts/run_copy_path_ablation.sh`
- Create: `scripts/analyze_copy_path_ablation.py`
- Create: `tests/cuda_copy/test_copy_path_cli.sh`
- Modify: `scripts/build_cuda_copy.sh`
- Modify: `doc/reports/gpu-internal-copy-contention-results.md`

**Interfaces:**
- Consumes: the Stage F victim topology and matched D2H source-read rate.
- Produces: `doc/results/gpu-contention/copy-path-ablation/` and optional traces under `doc/traces/gpu-contention/copy-path-ablation/`, with clean, original D2H positive control, replacement background, and replacement victim rows.

- [x] **Step 1: Write the failing CLI/measurement test.**

  Assert the real runner accepts `local-d2d-ce`, `streaming-hbm-read`, `streaming-hbm-write`, `l2-resident-read`, `peer-read`, `peer-write`, and `original-d2h`, rejects unknown paths, and records path, device list, bytes, cadence, and duty cycle.

- [x] **Step 2: Run the test and observe the expected failure.**

  Run `bash tests/cuda_copy/test_copy_path_cli.sh`; it must fail because the Stage G binaries and runner are absent.

- [x] **Step 3: Implement the minimum pressure/victim kernels.**

  Implement local D2D CE memcpy, streaming HBM read/write kernels with a working set larger than 6 MiB L2 and checksum writes, an L2-resident read with a working set below 6 MiB, destination peer-read and source peer-write kernels using peer pointers, and a local D2D CE victim. Keep independent buffers, prevent compiler elimination, expose actual bytes and elapsed time, and allow a duty-cycle/throttle parameter so replacement source-read rate is within ±10% before saturation points.

- [x] **Step 4: Build, smoke test, calibrate, and run the minimum matrix.**

  Build all binaries, run 2-repeat three-GPU smoke tests, calibrate replacement backgrounds to original D2H source-read rate, then run the `single-two-copy` one-stream/two-consecutive-copy victim and `edge-independent` controls for every replacement background and victim mode, three repetitions each, with original D2H positive controls.

- [x] **Step 5: Analyze and update the final report.**

  Report P2P count, measured slow count, second-operation slowdown, background per-device rate, HBM/L2 validation evidence, and the narrowest conclusion permitted by section 8.9.4. Stop after the minimum matrix and explicitly list which physical CE/L2/HBM/NVLink details remain unobservable.
