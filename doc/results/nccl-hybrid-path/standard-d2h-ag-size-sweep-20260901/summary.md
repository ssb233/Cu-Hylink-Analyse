# Standard full-duty D2H vs NCCL AllGather size sweep

Date: 2026-09-01 UTC

This is a direct A/B measurement, not a repeated paired formal. It uses one clean
`nccl-tests` run and one concurrent run with a standard full-duty D2H process.

## Configuration

- Victim: 4 × Tesla V100-SXM2-32GB, `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- NCCL test: `all_gather_perf -b 16M -e 1G -f 2 -g 4 -n 30 -w 10`.
- NCCL sizes: 16M, 32M, 64M, 128M, 256M, 512M, 1G.
- Background: GPU0 only, `host_copy_background --direction=d2h --size=255M`.
- Background cadence: `dutyCycle=1.0`; one `cudaMemcpyAsync(D2H)` followed by
  `cudaStreamSynchronize`, repeated continuously.
- Correctness: both NCCL runs returned 0 and every numeric row had `#wrong=0`.
- Both valid NCCL runs reported `nccl-library=23102`.

## NCCL bus bandwidth

| NCCL size | clean busbw (GB/s) | D2H concurrent busbw (GB/s) | slowdown |
| ---: | ---: | ---: | ---: |
| 16M | 74.58 | 71.38 | 4.29% |
| 32M | 90.60 | 87.02 | 3.95% |
| 64M | 102.53 | 99.01 | 3.43% |
| 128M | 105.73 | 103.54 | 2.07% |
| 256M | 108.34 | 106.89 | 1.34% |
| 512M | 110.39 | 109.00 | 1.26% |
| 1G | 111.68 | 111.49 | 0.17% |

The corresponding out-of-place `algbw` values were:

```text
clean      = 99.44, 120.80, 136.71, 140.97, 144.45, 147.19, 148.91 GB/s
concurrent = 95.17, 116.02, 132.02, 138.05, 142.51, 145.33, 148.65 GB/s
```

## D2H rate

The background process reported steady 2-second windows of 13.092, 13.092 and
12.023 GB/s. Its total rate over the 7.167-second lifetime was 12.648 GB/s.
The lower final window is the interval affected by the NCCL run; the process was
still full-duty (`dutyCycleTarget=1`) and was terminated only after the NCCL test
completed.

Raw logs:

- [`clean`](nccl_ag_16m_1g_clean.log)
- [`D2H concurrent`](nccl_ag_16m_1g_d2h_concurrent_final.log)
- [`D2H background`](nccl_ag_16m_1g_d2h_bg_final.log)
