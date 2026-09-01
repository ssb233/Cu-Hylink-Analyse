# NCCL 集合通信与主机拷贝背景流量实验结果

本文记录实验方案
[`nccl-d2h-interference-experiment-plan.md`](nccl-d2h-interference-experiment-plan.md)
截至 2026-08-28 在当前机器上的执行结果。当前机器实际只有 3 张 V100，因此将方案中
原定的四卡矩阵适配为 GPU 0、1、2 三卡；所有结果都应限定在这个硬件、单机 NVLink、
Ring+Simple 和当前 NCCL 构建上。

本文给出的是 NCCL 层的阶段性性能和定位结论，不是 HBM、L2、Copy Engine 或某条
memory-fence 指令的最终根因证明。

## 1. 测量范围与环境

| 项目 | 值 |
| --- | --- |
| GPU | 3 × Tesla V100-SXM2-32GB，CUDA visible devices 0,1,2 |
| GPU 拓扑 | 任意 GPU 对均为 NV2（NVLink） |
| Driver / CUDA | 580.178.04 / CUDA 12.6.85 |
| NCCL 测试 | `allgather`、`allreduce`、`reducescatter` |
| NCCL 路径 | `NCCL_ALGO=RING`、`NCCL_PROTO=SIMPLE` |
| NCCL build | `third_party/nccl` v2.31.2，V100 `sm_70`，原始 `.sys` 版本 |
| 背景流量 | 每张目标 GPU 一个 255 MiB D2H 或 H2D memcpy，单次 memcpy 后同步 |
| 主要指标 | `nccl-tests` 的 `busbw`；表中单位为 GB/s |

构建和硬件身份快照保存在
[`environment/20260828T075936Z/`](../results/nccl-contention/environment/20260828T075936Z/)，
构建契约和正确性检查由
[`test_nccl_sys_build_contract.sh`](../../tests/nccl/test_nccl_sys_build_contract.sh)
覆盖。

需要特别说明：外部 `/home/songxb26/HyLink/nccl` 当时已有用户的未提交 `.gpu` 修改，
实验没有覆盖它；本报告使用仓库内 `third_party/nccl` 的 `.sys` 构建。因此这不是
“官方库与第三方库完全等价”的证明，结果的构建身份必须以各结果目录中的 metadata
为准。

## 2. 主效应：D2H 会稳定降低 64 MiB NCCL 吞吐

### 2.1 阶段 B：短批次与 steady-state

64 MiB、255 MiB/卡背景流量的 regime sweep 共 162 个 case，全部通过解析和状态检查，
每个配置 3 次重复。D2H slowdown 的范围如下：

| Collective | `warmup=5`，`iters=20` | `warmup=5`，`iters=100/500` | `warmup=20`，`iters=20` | `warmup=20`，`iters=100/500` |
| --- | ---: | ---: | ---: | ---: |
| AG | 4.37% | 9.49–9.65% | 8.78% | 8.99–9.11% |
| AR | 7.43% | 7.11–7.17% | 6.97% | 6.94–7.00% |
| RS | 11.66% | 14.68–14.79% | 12.95% | 14.44–14.55% |

因此，D2H 影响不是只发生在 warmup 或最初几次 collective。除 AG 的最短
`warmup=5, iters=20` 之外，短批次和较长 measured window 得到的方向一致；后续主矩阵
固定使用 `warmup=20, iters=100` 作为可比较的 steady-state 点。

原始结果和解析摘要：
[`regime-sweep/b-stage-64m-20260828/`](../results/nccl-contention/regime-sweep/b-stage-64m-20260828/)。

### 2.2 阶段 C：64 MiB 敏感区与 256 MiB 吞吐区

正式主矩阵共 504 个 case：3 种 collective、64/256 MiB、两种 warmup、两种 measured
iterations、3 个生命周期场景、每个点 7 次重复，全部通过检查。下表取
`warmup=20, iters=100` 行，D2H 场景为三张 GPU 同时产生背景流量。

| Collective | Size | Clean-before | D2H-all | Clean-after | D2H slowdown | 背景实际带宽/卡 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG | 64 MiB | 64.83 | 58.62 | 64.78 | 9.58% | 12.59 GB/s |
| AG | 256 MiB | 67.11 | 65.54 | 67.20 | 2.34% | 12.59 GB/s |
| AR | 64 MiB | 71.49 | 66.31 | 71.48 | 7.25% | 12.71 GB/s |
| AR | 256 MiB | 75.45 | 73.79 | 75.53 | 2.20% | 12.66 GB/s |
| RS | 64 MiB | 71.09 | 60.41 | 71.11 | 15.02% | 12.54 GB/s |
| RS | 256 MiB | 74.11 | 70.81 | 74.00 | 4.45% | 12.59 GB/s |

64 MiB 是当前配置下的同步敏感区：AG、AR、RS 都有可重复的下降，RS 最大。256 MiB
仍有下降，但相对吞吐损失明显较小。4 MiB 的 triage 结果中 clean-after 在部分点不能
稳定回到 clean baseline，因此暂不用于根因归因；1 GiB 对比度较低，也不适合作为第一
个细粒度诊断点。

正式矩阵：
[`main-effect/formal-64-256-20260828/`](../results/nccl-contention/main-effect/formal-64-256-20260828/)。
尺寸 triage：
[`main-effect/triage-20260828/`](../results/nccl-contention/main-effect/triage-20260828/)。

### 2.3 生命周期恢复

所有正式 64/256 MiB 点的 clean-after 基本回到 clean-before，表中恢复率约为
99.8–100.1%。这说明当前实验中的下降主要与背景流量活跃期间的竞争有关，而不是
停止背景后永久改变了 NCCL 或 GPU 的性能状态。这个结论不能排除更短时间尺度上的
队列排空或时钟变化，只说明当前 clean-after 测量窗口没有观察到持续残留。

## 3. GPU-local 与方向对照

### 3.1 H2D 不是负对照

使用 64 MiB、`warmup=20`、`iters=100`、7 次重复，对比全卡 H2D 和仅 GPU0 H2D：

| Collective | Clean busbw | H2D all-GPU | slowdown | H2D GPU0-only | slowdown | 背景实际带宽/卡 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG | 64.76 | 53.89 | 16.79% | 63.00 | 2.76% | 11.38 / 11.38 GB/s |
| AR | 71.55 | 61.17 | 14.51% | 70.53 | 1.33% | 11.43 / 11.41 GB/s |
| RS | 71.08 | 55.51 | 21.90% | 67.90 | 4.29% | 11.39 / 11.32 GB/s |

这里的两列背景带宽分别对应 all-GPU 与 GPU0-only 运行；两组运行不是同一时刻的严格
配对，但实际带宽量级相近。结果仍然清楚地表明：全卡 H2D 的影响显著强于只在 GPU0
注入 H2D。因而 H2D 不能作为“无影响”的负对照，当前证据也不支持继续把现象表述为
“只有 D2H 才会触发的 system-scope fence 问题”。至少需要把方向、GPU-local 路径和
共享同步/数据移动资源共同纳入解释。

原始结果：

- [`locality-direction/h2d-all-64m-20260828/`](../results/nccl-contention/locality-direction/h2d-all-64m-20260828/)
- [`locality-direction/h2d-gpu0-64m-20260828/`](../results/nccl-contention/locality-direction/h2d-gpu0-64m-20260828/)

第一个目录是在 runner 动态支持 H2D 场景前启动的，目录内部的场景名仍显示为
`d2h-all`；其 `run-metadata.txt` 和实际背景程序均明确记录为 `background_direction=h2d`。
第二个目录的场景名已经为 `h2d-all`。

### 3.2 GPU0-only D2H 的 rank 传播

Inspector 代表性运行使用 64 MiB、3 次重复、周期 dump 1 ms。下表是
`coll_exec_time` 的 p50，列顺序为 rank 0/1/2；单位是该构建 Inspector trace 的 us。

| Collective | Clean-before | D2H all-GPU | D2H GPU0-only |
| --- | --- | --- | --- |
| AG | 681 / 682 / 682 | 719 / 713 / 712 | 677 / 693 / 693 |
| AR | 1241 / 1242 / 1241 | 1307 / 1307 / 1305 | 1240 / 1256 / 1256 |
| RS | 623.5 / 624 / 625 | 698 / 695 / 695 | 633 / 648 / 648 |

GPU0-only D2H 下，受影响的并不总是本地 rank 0：AG/AR/RS 中 rank 1/2 的 kernel
区间可能成为更慢的 straggler，而 rank 0 接近 clean。这与 collective 内部跨 rank
同步及完成时间传播相符，但不能据此断定具体是哪一条 NCCL primitive 或 fence 变慢。

全卡 D2H 下三张卡的 p50 都整体变长；clean-after 的 rank p50 又回到 clean 附近。
完整 trace 和按 rank/channel 的摘要见：

- [`inspector/regime-all-20260828/trace-summary.md`](../results/nccl-contention/inspector/regime-all-20260828/trace-summary.md)
- [`inspector/regime-gpu0-20260828/trace-summary.md`](../results/nccl-contention/inspector/regime-gpu0-20260828/trace-summary.md)

## 4. Inspector 校准与证据边界

Inspector OFF/ON 校准共 60 个 case：3 种 collective、64/256 MiB、5 次重复，全部通过。
正式校准矩阵的时间开销为 -0.09% 到 +0.24%，记录的 timing source 为 100% `kernel_gpu`；
因此 Inspector trace 可以用于本报告的 rank/channel 定位，吞吐真值仍采用 Inspector OFF
结果。独立 smoke case 的单点开销为 +0.91%，所以不能把“开销很小”推广到所有配置。

校准结果：
[`inspector/calibration-sys-20260828/summary.md`](../results/nccl-contention/inspector/calibration-sys-20260828/summary.md)。

当前 Inspector 证据支持的表述是：

> 背景主机拷贝使某些 collective、rank 或 channel 对应的 NCCL GPU kernel 执行区间变长。

当前证据不支持的表述是：

> 已经证明额外时间发生在 `fence_acq_rel_sys()`，或已经定位到某个 L2 slice、HBM
> partition、Copy Engine 仲裁单元。

周期性 Inspector dump 还可能制造 p99/max 尾部，因此 trace 中的 p50/p90 更适合做
阶段定位；正式宏观 slowdown 使用无 Inspector 的 7 次重复结果。

## 5. 当前阶段结论

1. 在 3 卡 V100 NVLink、Ring+Simple、255 MiB/卡背景条件下，AG、AR、RS 的 64 MiB
   NCCL 通信都能观察到稳定的背景流量干扰；RS 的相对下降最大。
2. 64 MiB 的影响在 `warmup=5/20` 和 `iters=20/100/500` 中基本持续存在，不是只由
   warmup 或前几次操作造成；clean-after 能恢复。
3. 256 MiB 仍有影响但对吞吐的相对占比更小；4 MiB 的清理不稳定，1 GiB 对比度不足，
   当前应优先用 64 MiB 做底层诊断。
4. 背景只放在 GPU0 时，全卡 H2D/D2H 的影响大幅减弱；但 GPU0-only 仍可能通过
   collective 同步把远端 rank 变成 straggler，说明不能只看本地 rank 的 kernel。
5. H2D all-GPU 也会造成明显下降，而且在本轮实际带宽约 11.4 GB/s 时下降大于
   D2H all-GPU 的 64 MiB 主矩阵。因此 D2H-only 的 system-fence 假设目前只能算候选
   机制，不能算结论。

## 6. 尚未完成的工作与下一步

目前还没有完成以下三类证据，因此暂不进入最终根因表述：

- NCCL primitive 内部的 wait、copy/reduce、fence、step-store 低扰动打点；
- NCCL 场景下 clean/D2H/H2D 的 Nsight Systems 代表性时序采样；
- `.sys -> .gpu` 只修改 post fence 的作用域消融及 correctness/残余开销分析。

按照方案，下一步应保持 64 MiB 敏感点，先采集少量无 Inspector 的 Nsight Systems
clean、D2H-all、H2D-all 代表场景，确认背景 memcpy 与 NCCL kernel 的真实重叠以及停止
后的排空时间；随后再实现 `sys-inst-post`/`sys-inst-wait` 等独立构建。由于 H2D 已经
表现为有效干扰源，primitive 分析必须同时保留 D2H 和 H2D，而不能只做 D2H-vs-clean。

在 primitive 数量级和 `.gpu` 消融结果出来前，不应把当前现象写成 HBM/L2/CE 的确定
资源边界，也不应把 `MEMBAR.SYS` 反汇编出现本身当成因果证据。

## 7. 原始数据索引

| 内容 | 目录 |
| --- | --- |
| B regime sweep | [`regime-sweep/b-stage-64m-20260828/`](../results/nccl-contention/regime-sweep/b-stage-64m-20260828/) |
| C 尺寸 triage | [`main-effect/triage-20260828/`](../results/nccl-contention/main-effect/triage-20260828/) |
| C 正式 7-repetition 主矩阵 | [`main-effect/formal-64-256-20260828/`](../results/nccl-contention/main-effect/formal-64-256-20260828/) |
| Inspector 开销校准 | [`inspector/calibration-sys-20260828/`](../results/nccl-contention/inspector/calibration-sys-20260828/) |
| Inspector 全卡 D2H | [`inspector/regime-all-20260828/`](../results/nccl-contention/inspector/regime-all-20260828/) |
| Inspector GPU0-only D2H | [`inspector/regime-gpu0-20260828/`](../results/nccl-contention/inspector/regime-gpu0-20260828/) |
| H2D all-GPU | [`locality-direction/h2d-all-64m-20260828/`](../results/nccl-contention/locality-direction/h2d-all-64m-20260828/) |
| H2D GPU0-only | [`locality-direction/h2d-gpu0-64m-20260828/`](../results/nccl-contention/locality-direction/h2d-gpu0-64m-20260828/) |

