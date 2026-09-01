# NCCL NVLink / CE Host Relay 干扰实验阶段性结果

日期：2026-08-31（UTC）  
状态：完成 Stage 1、Stage 2、P0 overlap 修正后的 64M/256M formal、SHM 四组 7 次 paired formal、official/experiment `.sys` clean 等价性、experiment `.sys/.gpu` 四环 64M/256M scope 消融、`.gpu` 的 SHM AG/AR/RS 对照、P2 代表 profile、`.gpu` 64M 的 D2H/H2D/single/disjoint 方向/拓扑矩阵，以及 64M disjoint pair 上 system/GPU scope 的 wait/fence/store/post/copy 细粒度打点；真实双路 collective 仍未开始。

## 1. 实验配置

- GPU：4 × Tesla V100-SXM2-32GB，compute capability 7.0，GPU 间拓扑均为 NV2。
- NCCL：official v2.31.2-1，源码目录 `/home/songxb26/HyLink/nccl`。
- NCCL 编译命令：

  ```bash
  make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
  ```

- NCCL runtime：`build/nccl-official-v2.31.2-sm70-sys/lib/libnccl.so.2.31.2`。
- nccl-tests：`build/nccl-tests-hybrid-sm70`。
- victim 固定环境：`NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0`。
- relay 和 victim 均绑定 NUMA node 0；formal 中 relay 使用 CPU `0,2,4,6`，victim 使用 `8,10`。
- 用户已确认不把 NCCL 中的 `sm_52` 占位 cubin 作为实验阻塞条件。

新增的独立 relay 测试器为 [`src/cuda_copy/host_relay.cu`](../../src/cuda_copy/host_relay.cu)，测试为 [`tests/cuda_copy/test_host_relay_contract.sh`](../../tests/cuda_copy/test_host_relay_contract.sh)。它使用独立 pinned host buffer，严格执行：

```text
D2H cudaMemcpyAsync -> cudaStreamSynchronize
H2D cudaMemcpyAsync -> cudaStreamSynchronize
```

warmup 后和正式计时结束后都校验 destination；正式计时记录 D2H、transition gap、H2D、端到端耗时和 iteration interval。

## 2. Stage 1：CE Host Relay standalone

单 edge 使用 100 次正式 iteration，覆盖计划要求的全部 size；D2H-only 使用 GPU0，H2D-only 使用 GPU1，relay 使用 `0 -> 1`。steady 区间的典型结果如下，带宽单位为 GB/s：

| payload | D2H-only | H2D-only | relay useful | relay traffic |
| ---: | ---: | ---: | ---: | ---: |
| 4M | 12.752 | 11.972 | 6.183 | 12.366 |
| 16M | 13.048 | 12.275 | 6.325 | 12.651 |
| 64M | 13.124 | 12.359 | 6.365 | 12.731 |
| 255M | 13.146 | 12.379 | 6.375 | 12.751 |
| 1G | 13.151 | 12.385 | 6.378 | 12.756 |

relay useful bandwidth 按 `S / T_end_to_end`，traffic rate 按 `2*S / T_end_to_end` 计算。单 edge 在约 16M 后进入平台区；64K–1M 主要表现为启动/提交开销。

拓扑 screening（64M）结果：

| topology | aggregate useful GB/s | aggregate traffic GB/s |
| --- | ---: | ---: |
| single `0 -> 1` | 6.365 | 12.731 |
| disjoint `0 -> 1, 2 -> 3` | 6.166 | 12.332 |
| four-ring `0 -> 1 -> 2 -> 3 -> 0` | 10.154 | 20.309 |

结果目录：[`doc/results/nccl-hybrid-path/relay-standalone/`](../results/nccl-hybrid-path/relay-standalone/)。

## 3. Stage 2：official NCCL clean baseline

AG、AR、RS 均使用四卡、Ring+Simple、100 次 warmup 后 100 次正式 iteration。`1M -> 1G` 的 11 个 size 点全部完成，所有 numeric row 的 out-of-place/in-place `#wrong` 均为 0。

结果目录：[`doc/results/nccl-hybrid-path/nccl-official-baseline/`](../results/nccl-hybrid-path/nccl-official-baseline/)。

## 4. Stage 3：64M background-first formal

每个场景执行 7 次 paired repetition，顺序为 `clean-before -> ready/start/steady -> concurrent -> stop -> clean-after`。slowdown 定义为：

```text
1 - median(concurrent algbw) / median(clean-before algbw)
```

| background | AG clean → concurrent | AG slowdown | AR clean → concurrent | AR slowdown | RS clean → concurrent | RS slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2H-only | 136.650 → 131.410 | 3.83% | 75.860 → 74.550 | 1.73% | 134.760 → 131.030 | 2.77% |
| H2D-only | 134.000 → 129.810 | 3.13% | 75.780 → 73.710 | 2.73% | 132.180 → 128.230 | 2.99% |
| single relay | 133.460 → 132.080 | 1.03% | 75.880 → 74.820 | 1.40% | 132.340 → 131.100 | 0.94% |
| disjoint pair | 137.110 → 128.740 | 6.10% | 75.910 → 73.880 | 2.67% | 134.740 → 128.490 | 4.64% |
| four-ring relay | 137.050 → 120.240 | 12.27% | 75.850 → 70.060 | 7.63% | 134.810 → 119.990 | 10.99% |

clean-after 的中位恢复率约为：AG 的 D2H/H2D/single/disjoint/four-ring 分别为 96.47%、97.84%、99.89%、99.99%、99.99%；AR 分别为 100.00%、100.21%、97.25%、97.18%、97.80%；RS 分别为 96.65%、98.68%、98.48%、97.88%、97.80%。本轮没有观察到持久性错误。

结果目录：

- [`AG 64M formal`](../results/nccl-hybrid-path/ce-relay-interference/20260830-ag64m-formal-reps7/)
- [`AR 64M formal`](../results/nccl-hybrid-path/ce-relay-interference/20260830-ar64m-formal-reps7/)
- [`RS 64M formal`](../results/nccl-hybrid-path/ce-relay-interference/20260830-rs64m-formal-reps7/)

## 5. 256M screening

256M 对 AG/AR/RS 各场景做了一次 paired screening，所有结果 `#wrong=0`：

| background | AG | AR | RS |
| --- | ---: | ---: | ---: |
| D2H-only | 0.06% | 0.70% | -1.79% |
| H2D-only | 1.69% | 1.17% | 1.85% |
| single relay | 0.66% | -0.66% | -1.03% |
| disjoint pair | 1.27% | -0.48% | 1.71% |
| four-ring relay | 5.58% | 2.33% | 2.99% |

该表只有单次 screening，不能替代 7 次 formal；它显示 256M 的干扰比 64M ring formal 弱，需要后续重复确认。

## 6. Stage 4：SHM 对照先导（1 rep）

SHM aggressor 独立进程使用 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0` 循环，日志已确认连接为 `via SHM/direct`；P2P victim 为 AG 64M。

```text
clean-before : 132.88 GB/s
concurrent   :  45.87 GB/s
slowdown     : 65.48%
clean-after  : 137.14 GB/s
```

三组 victim 日志的 `#wrong` 均为 0。该结果说明本机 SHM 路径的干扰签名明显强于单 edge 显式 CE relay，后续需要用更严格的重复和 profile 解释 mapped-host/polling/fence 贡献。

结果目录：[`doc/results/nccl-hybrid-path/shm-control/20260830-ag64m-screening/`](../results/nccl-hybrid-path/shm-control/20260830-ag64m-screening/)。

## 7. 初始阶段结论（已由后续 P0 数据补充）

1. 非流水化 host relay 在 16M 以上达到稳定平台，单 edge 有效带宽约 6.3 GB/s，双向 PCIe traffic 约 12.7 GB/s。
2. 64M formal 中，四卡 ring relay 对 NCCL 的 slowdown 最大且跨 AG/AR/RS 一致；single relay 和 disjoint pair 的影响较小但可测。
3. clean-after 基本恢复，当前没有观察到 NCCL correctness failure 或持久性退化。
4. SHM 对照的单次 AG 64M slowdown 很大，不能把显式 CE relay 与 NCCL SHM 视为同一路径。
5. 本轮尚未进入真实双路 collective、Nsight/Inspector 时序归因；`.gpu` fence scope 消融已在四环 64M/256M 完成，但仍需要 profile 和更完整的背景矩阵。

所有 pinned host allocation 都记录到当前 `RLIMIT_MEMLOCK=64K` 警告，但本轮 CUDA pinned allocation 成功；后续大规模矩阵仍需持续监控内存和分配失败。

## 8. P0：精确 overlap 与 64M/256M formal

根据更新后的实验计划，victim 的正式区间由 instrumented nccl-tests 写出 `measured_begin/measured_end`，relay 每 100 ms 写出单调时间戳、每 edge 的累计完成字节数及最新延迟；slowdown 使用同一 case 的 clean-before/clean-after bracket：

本轮 P0 使用 `build/nccl-v2.31.2-sm70-sys` 和 `build/nccl-tests-p0-sm70`，均为 V100 `sm_70` 构建；official NCCL 的原始源码和库未被这些 window marker 改动。

```text
bracket base = (clean-before busbw + clean-after busbw) / 2
slowdown = 1 - treatment busbw / bracket base
```

64M ring 使用 `-n5000 -w100`，256M 使用 `-n1300 -w100`；relay payload 为 255M，victim/relay 绑定 CPU `8,10` 与 `0,2,4,6`。每个 collective/scenario 均为 7 次。下表为 bracket slowdown 的均值、Student-t 95% CI 半宽和逐次范围：

| size / background | AG | AR | RS |
| --- | ---: | ---: | ---: |
| 64M ring | 12.97% ± 2.84% [9.39%, 17.34%] | 7.78% ± 2.28% [6.38%, 13.28%] | 14.16% ± 2.45% [11.08%, 16.81%] |
| 256M single | 0.78% ± 0.05% [0.73%, 0.86%] | 0.58% ± 0.03% [0.54%, 0.61%] | 0.88% ± 0.09% [0.74%, 1.01%] |
| 256M disjoint pair | 1.22% ± 0.08% [1.11%, 1.34%] | 0.87% ± 0.08% [0.73%, 0.97%] | 1.61% ± 0.08% [1.50%, 1.76%] |
| 256M ring | 3.91% ± 1.04% [3.04%, 5.86%] | 2.18% ± 0.23% [1.85%, 2.53%] | 3.93% ± 0.82% [3.27%, 5.75%] |

P0 64M 长窗口 clean baseline 约为 AG/RS 101 GB/s、AR 114 GB/s，与早先短 formal 中约 130/137 GB/s 的 gear 不同。因此这里不跨运行直接比较绝对带宽，只使用同 case 的 before/after bracket。64M 与 256M 均显示 ring 的干扰稳定高于 single；256M 的 single 只有约 0.6–0.9%，disjoint 约 0.9–1.6%，需要避免将其解释成超出 baseline gear noise 的单 edge 因果效应。

结果目录：

- [`64M ring P0 formal`](../results/nccl-hybrid-path/p0-overlap/20260831-64m-ring-reps7/)
- [`256M single/disjoint/ring P0 formal`](../results/nccl-hybrid-path/p0-overlap/20260831-256m-relay-reps7/)

审计结果：64M 为 21 cases、256M 为 63 cases；所有 victim/relay 返回正常，所有 NCCL numeric row 的 `#wrong=0`，每个 victim window 都有唯一 begin/end marker，relay overlap 均被 100 ms telemetry 覆盖。P0 仍然是背景 relay 与 victim overlap，不是真实双路 collective。

## 9. SHM 四组控制：7 次 paired formal

本轮固定 victim 为 AG 64M（`-n5000 -w100`），每个 control 都执行 `clean-before -> treatment -> clean-after`，每个 control 7 次；四组顺序由 seed `20260831` 随机化。P2P/SHM aggressor 均为独立 NCCL all-gather（255M、warmup 20、`-N0`），CE 组为同等 255M payload 的显式四 edge relay。

| control | transport 审计 | clean-before → treatment → clean-after (GB/s) | bracket slowdown |
| --- | --- | ---: | ---: |
| alone | 无 background | 101.25 → 101.34 → 101.32 | -0.06% ± 0.19% [-0.33%, 0.24%] |
| p2p-nccl | 7/7 `p2p-direct` | 101.24 → 38.61 → 101.29 | 61.87% ± 0.43% [61.39%, 62.52%] |
| shm-nccl | 7/7 `shm-direct` | 101.39 → 30.68 → 101.16 | 69.71% ± 1.61% [66.67%, 72.58%] |
| ce-relay | explicit four-ring relay | 101.46 → 87.89 → 101.40 | 13.35% ± 2.33% [11.35%, 18.04%] |

上述 CI 是 n=7 的 Student-t 95% CI 半宽；slowdown 逐次使用各自 before/after bracket。以 paired repetition 计算，SHM 相对 P2P 的额外 slowdown 为 `+7.84% ± 1.91%`，范围 `[+4.19%, +11.04%]`；CE relay 相对 P2P 则低 `48.52% ± 2.49%`。CE relay 在 victim 实测窗口内的 useful rate 均值为 9.435 GB/s、中位数 9.579 GB/s，范围 8.323–10.227 GB/s；对应双向 traffic 均值为 18.870 GB/s。

28 个正式 case 的窗口、GPU snapshot 和 correctness 均已审计；P2P/SHM aggressor 均正常退出且 transport 分类符合环境变量。所有 snapshot 的 P-state 为 P0，原始 clocks、温度、功率和 throttle reason 均保留；8 个 pre-snapshot 出现原始 reason code `0x0000000000000001`，未在统计中静默过滤。`rep-5/p2p-nccl` 的 clean-after 日志在完整输出 measurement（`#wrong=0`、window marker 完整）后出现 `double free or corruption` 并以 `-6` 退出；该原始 artifact 保留，随后用同参数 retry（返回 0，100.93 GB/s，`#wrong=0`）替代该 paired aggregate 的 clean-after 值，没有把这次 cleanup 异常当作性能样本。

结果目录：[`SHM/P2P/CE 四组 7-rep formal`](../results/nccl-hybrid-path/shm-control-p0/20260831-ag64m-four-controls-reps7/)。

## 10. 更新后的结论与后续

1. 精确 measured-window 结果确认：四 edge ring 在 64M 和 256M 都有稳定干扰；single relay 在 256M 仅约 0.6–0.9%，disjoint pair 约 0.9–1.6%，不能脱离 paired bracket 和 baseline gear noise 宣称单 edge 已被清晰分离。
2. SHM 控制的干扰显著强于 P2P 控制和显式 CE relay；transport 日志与 `NCCL_P2P_DISABLE` 配置相互吻合。当前仅将“基础资源竞争 + system-scope fence 放大”作为待验证假设，不据此宣称根因。
3. official/experiment `.sys` clean 等价性、`.sys/.gpu` release 对照、SHM AG/AR/RS 对照、P2 代表 profile 和 primitive 细粒度时序已完成；当前剩余工作是把 primitive 结果与 release scope 消融做关键路径交叉核对，再决定是否进入真实双路 collective，NCU 仍保持在候选机制进一步缩小之后。

## 11. official/experiment `.sys` clean 等价性

使用独立的 `experiment-sys` 构建目录和同一套 instrumented nccl-tests，运行 AG/AR/RS、64M/256M、每点 7 次，共 84 个 clean case。所有 case 均返回 0，`#wrong=0`，victim measured window 的 begin/end marker 各 1 次。

| collective / size | official mean (GB/s) | experiment mean (GB/s) | paired experiment vs official |
| --- | ---: | ---: | ---: |
| AG 64M | 101.317 | 101.194 | -0.121% ± 0.287% [-0.541%, +0.366%] |
| AG 256M | 108.014 | 107.963 | -0.048% ± 0.087% [-0.130%, +0.120%] |
| AR 64M | 113.724 | 113.677 | -0.041% ± 0.117% [-0.264%, +0.123%] |
| AR 256M | 119.560 | 119.563 | +0.002% ± 0.016% [-0.017%, +0.033%] |
| RS 64M | 101.060 | 100.886 | -0.171% ± 0.640% [-1.492%, +0.787%] |
| RS 256M | 107.361 | 107.360 | -0.001% ± 0.074% [-0.130%, +0.102%] |

括号内为 7 次 paired 相对差的 Student-t 95% CI 半宽和逐次范围；6 个区间均包含 0。因此 clean gate 通过，后续 `.gpu` 差异不需要由 experiment release 与 official release 的基础性能差异解释。结果目录：[`sys clean equivalence`](../results/nccl-hybrid-path/backend-equivalence/20260831-sys-clean-reps7/)。

## 12. experiment `.sys -> .gpu` 四环 scope 消融

`experiment-sys` 与 `experiment-gpu` 使用同一实验源码树的独立 build directory；`.gpu` 只替换 `prims_simple.h` 中两个 Ring+Simple post-send fence 为 `fence_acq_rel_gpu()`，保留 `st_relaxed_sys_global(connStepPtr, step)` 和其他路径的 system fence。两套 build 的 64M/256M 四环矩阵各为 AG/AR/RS × 7 次，共 42 个 case；两组均通过 victim/relay 返回码、`#wrong=0`、唯一 measured markers 和 100 ms relay telemetry 覆盖审计。

slowdown 使用每个 case 的 clean-before/clean-after bracket。`scope-removable fraction` 使用每个 repetition 的原始 busbw delta `(bracket clean - concurrent)` 计算，表示此次消融减少的额外 delta 比例，不是根因概率：

| size / collective | experiment-sys slowdown | experiment-gpu slowdown | scope-removable fraction |
| --- | ---: | ---: | ---: |
| 64M AG | 12.24% ± 1.81% [10.82%, 16.55%] | 9.89% ± 0.76% [8.72%, 11.16%] | 17.5% ± 13.4% [3.1%, 45.8%] |
| 64M AR | 8.58% ± 2.21% [6.43%, 12.49%] | 5.01% ± 0.40% [4.61%, 5.92%] | 38.7% ± 12.4% [22.9%, 59.1%] |
| 64M RS | 11.85% ± 1.59% [9.97%, 14.53%] | 9.20% ± 0.71% [8.37%, 10.55%] | 21.8% ± 10.5% [11.0%, 42.5%] |
| 256M AG | 4.04% ± 1.17% [3.07%, 6.19%] | 2.67% ± 0.10% [2.50%, 2.84%] | 28.9% ± 17.5% [12.2%, 56.7%] |
| 256M AR | 2.54% ± 1.05% [1.59%, 5.01%] | 1.42% ± 0.10% [1.31%, 1.62%] | 37.2% ± 18.2% [10.2%, 71.3%] |
| 256M RS | 3.50% ± 0.14% [3.30%, 3.67%] | 3.02% ± 0.08% [2.91%, 3.17%] | 13.8% ± 4.1% [8.3%, 20.2%] |

结果显示 `.gpu` 在所有已测 collective/size 上同向减少 slowdown，64M 的减少约 18–39%，256M 的减少约 14–37%，但 residual 仍存在，不能把 fence scope 写成唯一根因或产品修复。四环 relay useful rate 在 `.gpu` 组的均值为 AG 64M 8.890、AR 9.916、RS 9.429 GB/s；256M 分别为 9.730、10.079、9.494 GB/s。

结果目录：

- [`experiment-sys ring 64M`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-sys-ring-64m-reps7/)
- [`experiment-gpu ring 64M`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-gpu-ring-64m-reps7/)
- [`experiment-sys ring 256M`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-sys-ring-256m-reps7/)
- [`experiment-gpu ring 256M`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-gpu-ring-256m-reps7/)

## 13. experiment `.gpu` + SHM AG/AR/RS 对照

使用 `.gpu` victim、64M、`-n5000 -w100`，执行 alone 与 P2P victim + SHM aggressor 两组各 7 次。三类 collective 共 42 个 victim case 均返回 0、`#wrong=0`、window marker 完整；21 个 SHM aggressor 均正常退出、ready/window 完整，transport 日志全部分类为 `shm-direct`。

| collective | control | bracket slowdown |
| --- | --- | ---: |
| AG | alone | 0.18% ± 0.28% [-0.21%, 0.64%] |
| AG | SHM aggressor | 67.86% ± 1.56% [66.36%, 69.80%] |
| AR | alone | -0.02% ± 0.07% [-0.13%, 0.08%] |
| AR | SHM aggressor | 67.65% ± 3.14% [63.39%, 72.83%] |
| RS | alone | 0.13% ± 0.72% [-1.24%, 0.94%] |
| RS | SHM aggressor | 69.32% ± 0.90% [67.72%, 70.77%] |

三类 collective 的 `.gpu`-SHM 结果都保持强干扰，和 `.gpu` 四环 CE 中仅部分可消除的 slowdown 形成对比；这支持“SHM 路径还有独立的 mapped-host/polling/通信资源因素”的定位方向，但不构成具体硬件根因证明。结果目录：[`experiment-gpu SHM AG`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-gpu-shm-ag-64m-reps7/)、[`experiment-gpu SHM AR/RS`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-gpu-shm-ar-rs-64m-reps7/)。

## 14. P2 代表 profile：Inspector、Nsight Systems、CUPTI

### 14.1 Inspector 开销和 rank/channel 记录

对 `.gpu` AG 64M 使用 4 卡、100 次 warmup、1000 次 measured iteration，Inspector off/on clean 及 Inspector-on 四环并发各采 1 次。NCCL debug 单独写入每个 case 的文件，避免污染 nccl-tests 表格解析。

| case | busbw (GB/s) | measured time (us) |
| --- | ---: | ---: |
| Inspector off clean | 100.24 | 502.13 |
| Inspector on clean | 99.29 | 506.91 |
| Inspector on + four-ring | 84.45 | 595.98 |

Inspector-on 相对 off 的单次诊断开销约为 0.95%，不混入正式吞吐结论。clean-on 和 concurrent-on 各生成 8416 条 `coll_perf` 记录，4 个 rank 各 2104 条，全部为 `kernel_gpu` timing；每条记录包含 12 个 channel kernel event，Inspector profiler thread 的 `droppedOps=0`。concurrent-on 的 collective exec p50/p90/p99 为 `496/553/593 us`，clean-on 为 `492/504/511 us`；rank 0–3 的 p99 约为 `587/576/603/597 us`。这些数据用于定位 rank/channel 长尾，不能单独把变化归因到某一条 fence 指令。

原始 Inspector 目录：[`Inspector representative trace`](../traces/nccl-hybrid-path/20260831-gpu-ag64m-inspector-v2/)。

### 14.2 Nsight Systems overlap

Nsight Systems 以 background-first P0 runner 捕获 `.gpu` AG 64M 四环 clean-before、concurrent、clean-after 和 host relay 子进程；`--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none`，子进程跟踪开启。Nsight SQLite 中可分离出 3 个 `all_gather_perf` 进程和 1 个 `host_relay` 进程：中间 victim 包含 8416 个 NCCL Ring kernel，kernel 时间范围为 trace-relative `19.025–20.481 s`；host relay memcpy activity 范围为 `10.608–22.085 s`，覆盖 victim kernel 区间。

按 host_relay PID 统计，Nsight CUPTI 表包含 351 条 DtoH 与 343 条 HtoD activity；其中分别有 51 条 DtoH、48 条 HtoD 与中间 victim 的 GPU kernel 时间段相交，对应 13.637 GB 和 12.835 GB 的 activity bytes。Nsight 的全 trace GPU kernel summary 记录 25248 个 kernel，其中 96.1% 时间属于 NCCL Ring kernel；这些是时间线/活动重叠证据，不是物理 CE instance 或 descriptor/credit 根因证据。

结果和导出统计：[`Nsight representative trace`](../traces/nccl-hybrid-path/20260831-gpu-ag64m-ring/)，包括 `.nsys-rep`、`.sqlite` 和 `cuda_gpu_kern_sum`/`cuda_gpu_mem_time_sum` CSV。

### 14.3 CUPTI memcpy/channel representative case

用仓库已有 `libcupti_memcpy_channel_trace.so` 分别给 relay 和 victim 设置独立 CSV。relay CSV 共 576 条 activity，metadata `droppedRecords=0`：D2H 292 条、H2D 284 条；D2H 平均 activity duration 60.763 ms，H2D 平均 64.639 ms。记录覆盖 4 张 GPU 的 12 个 `(device, channelType, channelID)` 组合和 8 个 stream。victim CSV 共 424 条、dropped=0，仅包含 112320 bytes 的小 H2D setup activity，没有 P2P copy-kind 记录；这与 NCCL P2P 主路径由 GPU kernel 执行相符，不能解释为 NVLink 没有工作。

channel ID 和 channelType 在本报告中只作为 CUPTI 的公开、作用域化 activity metadata 保存，不能命名为物理 CE instance。结果目录：[`CUPTI representative trace`](../traces/nccl-hybrid-path/20260831-gpu-ag64m-ring-cupti/)。

## 15. experiment `.gpu` 64M 方向/拓扑矩阵

在 `.gpu` build 上固定 AG/AR/RS、64M、`-n5000 -w100`，对每个 collective 的 D2H-only（`0:0`）、H2D-only（`1:1`）、single relay（`0:1`）和 disjoint pair（`0:1,2:3`）各做 7 次 `clean-before -> treatment -> clean-after`。共 84 个 case，全部 victim/relay 返回 0，`#wrong=0`，三个 measured window 各有唯一 begin/end marker，84 个 treatment 的 relay overlap 均被 100 ms telemetry 覆盖。GPU snapshot 的 P-state 全为 P0；2352 条 GPU snapshot 中 6 条保留了原始非零 throttle reason `0x0000000000000001`，未静默删除。

下表为 7 次 paired repetition 的均值；括号内为 Student-t 95% CI 半宽和逐次范围。`before-only` 仅以 clean-before 为分母，`bracket` 使用同一 case 的 clean-before/clean-after 算术平均；`relay traffic` 是 measured window 内的方向或双向有效流量，不是 relay 整个生命周期的平均值。

| collective | background | before-only slowdown | bracket slowdown | relay traffic (GB/s) |
| --- | --- | ---: | ---: | ---: |
| AG | D2H-only | 1.78% ± 0.66% [1.23%, 2.85%] | 1.67% ± 0.34% [1.33%, 2.23%] | 13.03 ± 0.02 |
| AG | H2D-only | 3.33% ± 0.32% [3.04%, 4.05%] | 3.28% ± 0.13% [3.14%, 3.56%] | 12.21 ± 0.01 |
| AG | single relay | 1.98% ± 0.41% [1.58%, 2.76%] | 1.90% ± 0.22% [1.56%, 2.27%] | 12.63 ± 0.04 |
| AG | disjoint pair | 5.24% ± 0.37% [4.56%, 5.79%] | 5.29% ± 0.38% [4.54%, 5.90%] | 14.25 ± 0.89 |
| AR | D2H-only | 0.76% ± 0.13% [0.63%, 1.05%] | 0.75% ± 0.11% [0.64%, 1.00%] | 13.05 ± 0.01 |
| AR | H2D-only | 2.07% ± 0.14% [1.91%, 2.31%] | 2.07% ± 0.12% [1.95%, 2.30%] | 12.25 ± 0.01 |
| AR | single relay | 0.94% ± 0.09% [0.79%, 1.04%] | 0.94% ± 0.07% [0.83%, 1.04%] | 12.62 ± 0.02 |
| AR | disjoint pair | 2.84% ± 0.13% [2.66%, 3.03%] | 2.85% ± 0.15% [2.66%, 3.09%] | 14.21 ± 1.33 |
| RS | D2H-only | 1.00% ± 0.76% [0.19%, 2.17%] | 0.74% ± 0.44% [0.24%, 1.43%] | 13.01 ± 0.02 |
| RS | H2D-only | 3.28% ± 0.35% [3.00%, 3.98%] | 3.59% ± 0.27% [3.05%, 3.86%] | 12.21 ± 0.02 |
| RS | single relay | 1.71% ± 0.68% [0.95%, 2.53%] | 1.63% ± 0.26% [1.10%, 1.91%] | 12.64 ± 0.04 |
| RS | disjoint pair | 5.27% ± 0.66% [4.01%, 6.10%] | 4.85% ± 0.37% [4.17%, 5.45%] | 14.36 ± 0.93 |

方向/拓扑上，H2D-only 在三类 collective 都高于 D2H-only，disjoint pair 又高于 single relay；这与 relay 所覆盖的 GPU/host 资源位置和并行 edge 数量相关，但不能单凭该相关性指定物理 CE、L2、HBM、NVLink scheduler 或 fence 为根因。single relay 的 bracket slowdown 约 0.94–1.90%，仍属于小效应；disjoint pair 的 AG/AR/RS 分别约 5.29%/2.85%/4.85%，因此下一步 primitive 时序应优先选择 disjoint 或四环代表点，并保留 D2H/H2D 作为方向性对照。

结果目录：[`experiment-gpu 64M direction/topology`](../results/nccl-hybrid-path/fence-scope-ablation/20260831-experiment-gpu-direction-topology-64m-reps7/)。

## 16. P2 primitive trace：system/GPU scope 细粒度对照

### 16.1 实验范围和审计

本轮在 `third_party/nccl` 为 `wait`、`fence`、`store`、`post`、`copy` 分别建立独立的
instrumented build；system-scope build 使用：

```bash
make -C third_party/nccl -j80 src.build \
  BUILDDIR=<独立目录> \
  PRIMITIVE_TRACE_FENCE_SCOPE=sys PRIMITIVE_TRACE=1 \
  PRIMITIVE_TRACE_KIND=<wait|fence|store|post|copy> \
  NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

GPU-scope instrumented build 使用同样的 `sm_70` 配置和独立目录，scope 取 `gpu`。victim
固定为 4 卡、Ring+Simple、64M、disjoint relay `0:1,2:3`；relay 为 255M full relay，
CPU 绑定和前述 formal 相同。每个 primitive 的代表点均执行 AG/AR/RS 各一次
`clean-before -> concurrent -> clean-after`，采样周期为 N=128。

5 个 primitive × 3 个 collective 的 system-scope 代表点共 15 个 case 均通过审计：
victim、relay 和 victim helper 返回码均为 0，`#wrong=0`，每个 measured window 的
begin/end marker 唯一，relay telemetry 覆盖完整，N=128 均无 overflow。instrumented
library 只用于 primitive 时序，不把其吞吐直接当作 release 真值。

### 16.2 system-scope 代表性时序

下表为 system-scope build 的采样记录，时间单位为 µs；箭头表示
`clean-before -> concurrent`，带宽为对应 victim 测量窗口的 algbw GB/s。原始 manifest
同时保存 mean、p50、p90、p99、max 和每 rank/channel 文件。

| primitive | AG：p50 / p99；algbw | AR：p50 / p99；algbw | RS：p50 / p99；algbw |
| --- | --- | --- | --- |
| post | 30.8→99.1 / 52.2→475.8；131.80→67.78 | 72.1→199.2 / 101.9→893.5；74.87→40.19 | 35.3→102.1 / 58.0→506.0；132.42→69.25 |
| fence | 37.8→112.3 / 51.6→500.8；132.38→79.52 | 77.8→210.6 / 100.1→916.9；75.13→43.59 | 45.0→117.5 / 59.1→520.6；132.97→76.08 |
| store | 0.9→0.7 / 2.8→3.1；132.13→71.63 | 1.5→0.9 / 3.8→3.9；74.47→40.19 | 0.5→0.3 / 2.0→1.9；133.04→67.90 |
| copy | 417.2→447.5 / 437.8→869.3；129.01→69.30 | 755.4→828.4 / 793.1→1468.6；73.66→42.96 | 425.3→460.8 / 451.3→969.3；130.33→67.92 |
| wait | 7.9→91.9 / 41.3→952.9；129.44→72.40 | 16.6→215.8 / 106.4→1725.7；74.09→41.07 | 6.7→102.6 / 41.9→971.1；130.30→70.19 |

在该 instrumented disjoint case 中，wait、post、fence 和 copy 的 concurrent 长尾均明显
变长；store 本身保持在亚微秒到数微秒量级，没有观察到同方向的时序放大。这里的 algbw
下降包含 primitive 打点和 relay 干扰的共同影响，不能用这张表直接计算各 primitive 的
因果百分比，也不能把所有 channel 的记录相加为 critical path。

wait trace 还记录了原始 `spinLoadCount`。以 concurrent、N=128 为例，AG 的 wait-recv
spin p50/p90/p99 为 `6/3924/5555`，wait-send 为 `1/1968/3356`；AR 分别为
`135/7150/9496` 和 `6/5399/7528`；RS 分别为 `8/4124/5879` 和 `1/2170/3727`。
这说明长尾伴随 polling load 增长，但仍不能单独区分 mapped-host、调度、L2/HBM 或
system-domain completion 的具体物理来源。

### 16.3 采样周期和容量边界

对每个 primitive 又执行了 AG 64M disjoint 的 N=1/8/32/128 扫描。每个 N=1、8、32
case 的每个阶段均报告 48 条 overflow；N=128 的每个阶段均为 0。四档采样下，
concurrent 的 p50/p99 范围如下：

| primitive | concurrent p50 范围 (µs) | concurrent p99 范围 (µs) |
| --- | ---: | ---: |
| post | 96.9–103.0 | 474.7–491.0 |
| fence | 108.9–112.5 | 495.8–514.4 |
| store | 0.7–0.8 | 3.08–3.14 |
| copy | 436.9–449.7 | 745.2–903.5 |
| wait | 93.8–109.9 | 890.7–991.6 |

因此后续需要使用 N=128（或先扩大 ring capacity 后再使用更密采样）作为正式 primitive
统计点；低 N 的记录只能作为定性检查，不能把 overflow 后的样本数当作完整分布。

结果目录：

- [`system post representative`](../results/nccl-hybrid-path/primitive-trace/20260831-sys-post-disjoint-64m-allcoll/)
- [`system fence representative`](../results/nccl-hybrid-path/primitive-trace/20260831-sys-fence-disjoint-64m-allcoll/)
- [`system store representative`](../results/nccl-hybrid-path/primitive-trace/20260831-sys-store-disjoint-64m-allcoll/)
- [`system copy representative`](../results/nccl-hybrid-path/primitive-trace/20260831-sys-copy-disjoint-64m-allcoll/)
- [`system wait representative`](../results/nccl-hybrid-path/primitive-trace/20260831-sys-wait-disjoint-64m-allcoll/)
- [`system primitive sample sweeps`](../results/nccl-hybrid-path/primitive-trace/)

GPU-scope instrumented 的 representative 和 sweep 目录使用同一父目录下不带 `sys-` 前缀的
`20260831-{post,fence,store,copy,wait}-disjoint-64m[-allcoll]`；release scope 的正式
吞吐结论仍以第 12 节的 7-rep paired matrix 为准。

### 16.4 当前证据边界

system/GPU scope 的 release 对照已经显示 `.gpu` 只能移除部分 slowdown，且 residual
仍然存在；本轮 primitive trace 进一步显示 concurrent 下 wait/fence/post/copy 具有
长尾，而 store 单独没有同等放大。两者方向一致，但 instrumented build 的吞吐不能与
release slowdown 直接相除。因此当前最稳妥的结论是：

```text
CE relay 与 NCCL Ring+Simple 的并发会放大若干等待、排序和工作区间阶段；
post fence scope 是可检验的贡献候选，但不是已证明的唯一根因。
```

真实双路 collective、NCU 单 kernel site 和产品级 `.gpu` 修改仍未进入。

## 17. 标准满负载 D2H：4 卡 AllGather 尺寸 A/B

为避免把 duty shaping 或完整 host relay 混入 D2H 结论，执行了一组最直接的单次
clean/concurrent A/B。victim 使用 4 卡 V100 的 `all_gather_perf`，命令为：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 ./all_gather_perf -b 16M -e 1G -f 2 -g 4 -n 30 -w 10
```

背景进程只使用 GPU0，执行 255M pinned-host D2H，`dutyCycle=1.0`，每次
`cudaMemcpyAsync(D2H)` 后调用 `cudaStreamSynchronize`，然后无限循环。clean 和
concurrent 两次有效运行均为 NCCL 2.31.2（`nccl-library=23102`），所有行
`#wrong=0`。

| NCCL size | clean busbw (GB/s) | D2H concurrent busbw (GB/s) | NCCL slowdown |
| ---: | ---: | ---: | ---: |
| 16M | 74.58 | 71.38 | 4.29% |
| 32M | 90.60 | 87.02 | 3.95% |
| 64M | 102.53 | 99.01 | 3.43% |
| 128M | 105.73 | 103.54 | 2.07% |
| 256M | 108.34 | 106.89 | 1.34% |
| 512M | 110.39 | 109.00 | 1.26% |
| 1G | 111.68 | 111.49 | 0.17% |

D2H 进程的实测带宽窗口为 13.092、13.092 和 12.023 GB/s，总生命周期平均
12.648 GB/s；最后一个窗口与 NCCL 运行重叠，下降说明 NCCL 也会反向影响 D2H。

本批只有一次 clean 和一次 concurrent，不能给出统计置信区间，但方向很清楚：在
满负载标准 D2H 下，16–64M 的 NCCL AllGather 下降约 3.4–4.3%，128M 约 2.1%，
256M 以上降至约 1.3% 或更低，1G 基本不可见。这里的背景是纯 D2H，不是四环
D2H→H2D relay；后者的干扰结果不能与本表直接比较。

原始数据：[`standard-d2h-ag-size-sweep-20260901`](../results/nccl-hybrid-path/standard-d2h-ag-size-sweep-20260901/)。
