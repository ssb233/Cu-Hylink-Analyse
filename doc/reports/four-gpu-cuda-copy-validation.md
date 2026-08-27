# Four-GPU CUDA copy validation

This is a single default-configuration validation run for the four-GPU CUDA
copy path. It verifies that the implementation can allocate and exercise the
full matrix; it is not a repeated experiment and does not establish a causal
performance conclusion.

## Environment

- GPUs: 4 × Tesla V100-SXM2-32GB
- Topology: every GPU pair reports `NV2`; all GPUs are on NUMA node 0
- Driver: 580.178.04
- CUDA toolkit: 12.6.85
- Payload: 255 MiB (`267386880` bytes)
- D2D warmup: 10 iterations
- D2D repetitions: 20
- Background devices: `0,1,2,3`
- Repository snapshot: `f67c24bfe82675a111bb1044ed15cc529f13a8d2`

The complete raw run is under
[`doc/results/4gpu/default-4gpu-validation/`](../results/4gpu/default-4gpu-validation/),
with the matrix table in
[`summary.csv`](../results/4gpu/default-4gpu-validation/summary.csv) and the
captured environment in
[`environment.txt`](../results/4gpu/default-4gpu-validation/environment.txt).

## Observed single-run values

| Pattern | Background | Directed copies | D2D aggregate GB/s | Background aggregate GB/s |
| --- | --- | ---: | ---: | ---: |
| ring | none | 4 | 193.957 | — |
| ring | D2H | 4 | 193.319 | 15.934 |
| ring | H2D | 4 | 193.163 | 15.095 |
| allpairs | none | 12 | 179.756 | — |
| allpairs | D2H | 12 | 127.796 | 11.844 |
| allpairs | H2D | 12 | 178.159 | 11.633 |

Within this one run, the ring values were nearly unchanged by the selected
four-GPU host traffic, while allpairs with D2H was lower than its no-background
case. This observation needs repeated runs and profiler/counter correlation
before it can be interpreted as interference. The runner currently measures
the copy-engine path only; it does not yet collect SM occupancy or NVLink/PCIe
counters.

## Reproduction

```bash
./scripts/build_cuda_copy.sh
./scripts/run_4gpu_copy_matrix.sh \
  --size=255M --repeats=20 \
  --outputRoot=doc/results/4gpu/<new-run-directory>
```

For a fast code-path check, use `--size=1M --repeats=1`. The later report
should use at least five independent fixed-size repetitions per scenario and
compare each background case with its matching pattern baseline.
