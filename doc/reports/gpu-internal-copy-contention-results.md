# 四卡 D2D / D2H / H2D 资源竞争实验结果

本文是对 [`gpu-internal-copy-contention-experiment-plan.md`](gpu-internal-copy-contention-experiment-plan.md)
第一批实验的执行记录和阶段性分析。实验对象为 4 张 Tesla V100-SXM2-32GB，
目标是判断 allpairs `cudaMemcpyPeerAsync` 与背景 D2H/H2D 是否存在竞争，以及
竞争是否具有 GPU-local 的方向性。

本文不是最终的 HBM/L2 根因证明。Copy Engine 内部仲裁、L2 cache policy 和
具体 memory partition 仍然需要后续消融实验。

## 1. 实验环境与测量口径

| 项目 | 值 |
| --- | --- |
| GPU | 4 × Tesla V100-SXM2-32GB |
| GPU topology | 任意 GPU 对均为 `NV2` |
| Host topology | 四张卡均在 NUMA node 0 |
| Driver / CUDA | 580.178.04 / CUDA 12.6.85 |
| PCIe | 当前查询为 Gen3 x16 |
| D2D payload | 255 MiB = 267,386,880 bytes |
| D2D warmup | 固定 10 次 |
| D2D pattern | ring=4 条有向边，allpairs=12 条有向边 |
| allpairs stream 语义 | 每张 GPU 1 个 source stream，3 个 outgoing copy 在该 stream 上串行排队 |
| host copy | 每个参与 GPU 1 个 pinned host buffer、1 个 device buffer、1 个 stream；每次 memcpy 后同步 |

理论参数、HBM/L2/NVLink 模型和传输路径见
[`v100-32gb-memory-hierarchy-and-copy-path-analysis.md`](v100-32gb-memory-hierarchy-and-copy-path-analysis.md)。

## 2. A 阶段：基线稳定性与 repeat 诊断

### 2.1 长批次基线稳定性

使用 allpairs、255 MiB、warmup=10、repeats=200、无背景流量独立运行 10 次，
所有运行成功：

| 统计 | allpairs aggregate GB/s |
| --- | ---: |
| mean | 186.573 |
| median | 186.506 |
| min / max | 186.071 / 187.495 |
| range | 0.76% |

在相同 repeat 配置下，基线波动远小于后续 D2H 效应。

原始目录：[`baseline-stability/`](../results/gpu-contention/baseline-stability/)。

### 2.2 repeat-sweep 暴露的测量 regime

在无背景 allpairs 下改变 measured repeats，结果如下：

| repeats | aggregate GB/s |
| ---: | ---: |
| 20 | 177.986 |
| 50 | 179.817 |
| 100 | 179.845 |
| 200 | 186.881 |
| 500 | 190.738 |

这不是普通的 1% 以内随机噪声，而是 batch length 对测量吞吐有系统影响。追加的
冷态/热态试验中，500 次运行后 GPU pclk 已为 1530 MHz，但立即启动一个新的
20-repeat benchmark 仍只有 179.819 GB/s；所以不能简单归因于“GPU 还没有升到高时钟”。

`nvidia-smi dmon` 显示 20-repeat 运行很短，只有过渡性的 pclk 样本；500-repeat
运行则稳定出现 pclk=1530 MHz、mclk=877 MHz。当前最谨慎的表述是：**短批次和长
批次进入了不同的 CE/NVLink steady-state 或进程初始化/队列深度 regime；后续的
chunk-level event 已经把它分解为“前约 100 次 round 的阶段性状态 + 后续
steady-state”，但触发该阶段转折的具体 CE/clock/队列机制仍未定位。**

原始目录：

- [`timing-repeat-sweep/`](../results/gpu-contention/timing-repeat-sweep/)
- [`timing-repeat-probe/`](../results/gpu-contention/timing-repeat-probe/)

后续机制分析不能混合 20-repeat 和 500-repeat 的 aggregate；两者必须作为不同协议
分别比较。

### 2.3 分块 event 诊断：repeat 差异主要来自前段状态

为分解 batch length 的影响，`d2d_multi_peer_bw` 新增可选
`--chunkRepeats=10`。该模式在每个 source stream 上记录连续 chunk 的 CUDA event，
但不在 chunk 边界做 host synchronize；默认 `chunkRepeats=0` 时不创建这些额外
events，保留原有测量语义。20-repeat 产生 2 个 chunk，500-repeat 产生 50 个
chunk。

255 MiB、allpairs、每个场景 3 次的 aggregate 结果如下。分块模式会引入少量 event
开销，因此这里主要使用它观察时间形态，不把绝对值与无诊断主效应表混合：

| repeats | 场景 | aggregate median GB/s | 3 次范围 GB/s |
| ---: | --- | ---: | ---: |
| 20 | none | 179.781 | 176.486–179.813 |
| 20 | GPU0-only D2H | 130.020 | 130.004–130.023 |
| 20 | all-GPU D2H | 127.809 | 127.456–127.862 |
| 500 | none | 190.846 | 190.826–190.849 |
| 500 | GPU0-only D2H | 173.758 | 173.744–173.802 |
| 500 | all-GPU D2H | 171.666 | 171.645–173.920 |

500-repeat 每个 source 的分块均值（3 次运行合并）显示出一致的阶段变化：

| 场景 / source | chunk 0–8（前 90 次） | chunk 9 | chunk 10 | chunk 11 | chunk 12–49 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none / 任意 source | 约 44.96 | 约 45.5 | 约 48.44 | 约 48.44 | 约 48.44 |
| GPU0 D2H / source0 | 约 32.51 | 32.44 | 32.46 | 38.78 | 48.31 |
| GPU0 D2H / source1–3 | 约 44.95 | 约 45.95 | 约 48.43 | 约 48.43 | 约 48.43 |
| all-GPU D2H / source0–3 | 约 32.02 | 约 32–33 | 约 36.5–37.3 | 约 41.5–43.3 | 约 47.80–47.84 |

这给出一个比“500-repeat 竞争只有 -10%”更准确的解释：

1. 无背景时，前约 90–100 次 round 处于约 45 GB/s 的初始阶段，随后进入约
   48.4 GB/s 的阶段；20-repeat 几乎完全落在前一阶段，500-repeat 则包含后面的
   高吞吐阶段。
2. GPU0-only D2H 只把 GPU0 的初始 source 阶段压到约 32.5 GB/s，远端 source
   仍处于无背景的初始值；约第 110 次 round 后，GPU0 也恢复到约 48.3 GB/s。
3. 全卡 D2H 将四个 source 的初始阶段都压到约 32 GB/s，随后同样恢复到约
   47.8 GB/s。因此 D2H 影响更像是改变/延长前段 CE/数据路径状态，而不是把
   稳态带宽永久压低。

一次配套的无 profiler `nvidia-smi dmon` 记录中，pclk 从 135/1290 MHz 过渡到
1530 MHz，而 mclk 保持约 877 MHz；但 dmon 的 1 秒采样无法把这个过渡精确对齐到
chunk 边界，且早期 chunk 在 pclk 已到 1530 MHz 后仍然偏低。因此时钟变化可能
参与初始化阶段，但不能单独作为根因。该记录中的标准 `sm`/`mem` 仍不是 CUDA
kernel occupancy/精确 DRAM utilization。

原始数据：[`timing-repeat-diagnostic/`](../results/gpu-contention/timing-repeat-diagnostic/)，
配套 telemetry：[`telemetry-d2h-all/`](../results/gpu-contention/timing-repeat-diagnostic/telemetry-d2h-all/)。

### 2.4 `chunkRepeats` 粒度对照：转折点稳健，但绝对带宽受诊断扰动

为了检查前述阶段转折是否依赖于 chunk 划分，补做了
`repeats=500`、255 MiB、allpairs、每个配置 2 次的 `chunkRepeats=1/5/20`
对照。三种粒度都在约第 100 次 measured round 附近从低阶段进入高阶段，说明这个
转折不是 `chunkRepeats=10` 特有的分组边界；但 chunk event 本身会改变绝对吞吐，
尤其是 `chunkRepeats=1` 会为每个 memcpy 插入 event，因此不能把下面数值当作无诊断
主效应的直接替代。

| `chunkRepeats` | none median GB/s | GPU0-only D2H median GB/s | all-GPU D2H median GB/s |
| ---: | ---: | ---: | ---: |
| 1 | 191.272 | 177.666 | 174.924 |
| 5 | 191.266 | 174.254 | 172.251 |
| 20 | 190.793 | 175.549 | 171.442 |

分块粒度改变了 aggregate 的绝对值和 slowdown 大小，但没有消除 source-local 的
早期现象：GPU0-only D2H 下主要是 source0 的前段带宽下降，全卡 D2H 下四个 source
都出现类似的前段下降，随后逐步恢复。因此当前结论应限定为：**阶段转折位置对
chunk 粒度具有一定稳健性；chunk 诊断适合定位 phase boundary，不适合精确测量
无 event 插入时的性能损失。**

原始数据：[`chunk-sweep/`](../results/gpu-contention/timing-repeat-diagnostic/chunk-sweep/)。

### 2.5 100 ms clock/pstate 对照：graphics clock 不能单独解释转折

对 `repeats=500`、`chunkRepeats=10`、255 MiB、allpairs 的 none、GPU0-only D2H、
all-GPU D2H 各执行 1 次；在 benchmark 命令区间内用
`nvidia-smi --query-gpu=... -lms 100` 采集每张卡的 timestamp、P-state、graphics
clock、memory clock、粗粒度 GPU/memory utilization 和 power。结果如下：

| 场景 | aggregate GB/s | 命令区间 | graphics clock 状态 | memory clock | `utilization.gpu` 均值 |
| --- | ---: | ---: | --- | ---: | --- |
| none | 190.821 | 9.730 s | 135/1290/1425 → 1530 MHz | 877 MHz | 92.4–93.4% |
| GPU0-only D2H | 174.064 | 10.230 s | 全程 1530 MHz | 877 MHz | GPU0=100%，远端约 84–85% |
| all-GPU D2H | 171.803 | 10.721 s | 全程 1530 MHz | 877 MHz | 四卡均 100% |

none 场景在开始约 1 秒内完成 graphics clock 的升频；但 D2H 场景是在前一个
场景之后执行，第一条有效采样已经是 P0/1530 MHz，仍然保留了前约 100 次 round
的低阶段：GPU0-only D2H 的 source0 前 110 次约 32.5 GB/s，第 111 个 chunk
约 42.1 GB/s，随后约 48.3 GB/s；all-GPU D2H 的 source 前 110 次约 32 GB/s，
随后逐步恢复到约 47.8 GB/s。由此可以排除“D2H 场景只是因为 graphics clock
尚未升到 1530 MHz”这一简单解释，但不能据此区分 CE 仲裁、队列状态或其他内部
数据路径机制。

这里的 `utilization.gpu` 是 `nvidia-smi` 的粗粒度活动比例，不能解释为 SM
occupancy；`utilization.memory` 也不是 HBM/L2 字节计数。由于三个场景是串行
执行，clock 对照不是随机化的独立重复；逐次 synchronize 对照见下一节。

原始数据：[`high-res-telemetry/`](../results/gpu-contention/timing-repeat-diagnostic/high-res-telemetry/)。

### 2.6 逐次 synchronize 对照：固定低阶段，但改变 source-local 计时语义

为直接检验“队列深度/批量异步提交”是否参与 repeat-regime 差异，增加
`--syncEachIteration=0|1`。设置为 1 时，每个 measured round 排队完 allpairs
的 12 条 D2D 拷贝后，依次对四个 source stream 执行 `cudaStreamSynchronize`；warmup
仍固定 10 次，`chunkRepeats=0`。异步模式和逐次同步模式各完成
`repeats=20/500`、none/GPU0-only D2H/all-GPU D2H、3 次重复。

下表为每组 3 次的 aggregate median（括号内为 min–max）：

| repeats | 计时模式 | none | GPU0-only D2H | all-GPU D2H |
| ---: | --- | ---: | ---: | ---: |
| 20 | async batch | 179.794 (176.523–179.806) | 129.766 (129.486–129.986) | 127.823 (127.787–127.888) |
| 20 | sync each round | 179.109 (178.720–179.245) | 129.746 (129.130–130.247) | 127.133 (126.953–127.386) |
| 500 | async batch | 190.766 (190.731–190.795) | 173.100 (172.878–173.397) | 171.034 (171.033–171.045) |
| 500 | sync each round | 179.316 (179.299–179.319) | 130.668 (130.640–130.683) | 127.340 (127.323–127.343) |

结果有三点：

1. 逐次同步使 20/500 两个 repeat regime 收敛到同一量级：无背景约 179 GB/s，
   GPU0-only D2H 约 130 GB/s，全卡 D2H 约 127 GB/s。它阻止了异步批量模式在长
   batch 后进入约 190 GB/s 的高阶段。
2. 在逐次同步协议内，500-repeat 的 D2H 下降重新稳定为约 -27.1%（GPU0-only）
   和 -29.0%（all-GPU），而同一矩阵的 async batch 分别约 -9.3% 和 -10.3%。
   这支持“持续排空/限制队列会保持早期低阶段”的测量学解释，但还不能证明具体
   是 CE 仲裁还是其他内部队列机制。
3. 逐次同步会破坏原有 per-source local 计时的可解释性。以 repetition-2 为例，
   500-repeat、GPU0-only D2H 的 async source GB/s 为
   `43.22/47.70/47.70/47.70`，主要只有 source0 较慢；sync each round 则为
   `32.67/32.67/32.67/32.67`。四个 source 的 event 起止区间几乎相同，说明
   host-paced round loop 把 source0 的等待耦合到了其他 source 的测量区间。因此
   sync 模式适合作为“队列深度控制”，不适合作为 source-local 归因主协议。

背景流量的实际 aggregate 也随计时模式变化，例如 500-repeat 下 GPU0-only D2H
的 async/sync median 分别约为 11.22/9.72 GB/s，全卡 D2H 约为 12.85/11.43 GB/s；
所以该对照不是恒定背景压力的纯同步消融。后续需要 background per-GPU
bytes/GB/s 或 duty-cycle 控制后再做精确压力匹配。

原始数据：[`sync-each-iteration/`](../results/gpu-contention/timing-repeat-diagnostic/sync-each-iteration/)；
复现实验脚本：[`run_sync_each_iteration_matrix.sh`](../../scripts/run_sync_each_iteration_matrix.sh)。

## 3. C 阶段：主效应重复复现

### 3.1 用户默认 20-repeat：原始现象稳定

7 次独立六场景矩阵均成功：

| pattern | background | mean GB/s | median GB/s | min–max GB/s | 相对同 pattern none |
| --- | --- | ---: | ---: | ---: | ---: |
| ring | none | 193.961 | 193.972 | 193.907–193.975 | — |
| ring | D2H | 193.329 | 193.330 | 193.322–193.333 | -0.33% |
| ring | H2D | 193.164 | 193.163 | 193.161–193.169 | -0.41% |
| allpairs | none | 179.809 | 179.811 | 179.783–179.826 | — |
| allpairs | D2H | 127.935 | 128.037 | 127.586–128.149 | **-28.82%** |
| allpairs | H2D | 178.202 | 178.217 | 178.143–178.233 | -0.89% |

这确认了用户提供的论坛现象在当前四卡机器和当前代码上可重复：allpairs+D2H
有明显下降，ring 和 H2D 是负对照。

原始目录：[`main-effect/repeats-20/`](../results/gpu-contention/main-effect/repeats-20/)。

### 3.2 500-repeat：效应仍在，但幅度不同

将 measured repeats 提高到 500 后，再做 7 次独立六场景矩阵：

| pattern | background | mean GB/s | median GB/s | min–max GB/s | 相对同 pattern none |
| --- | --- | ---: | ---: | ---: | ---: |
| ring | none | 193.993 | 193.994 | 193.991–193.995 | — |
| ring | D2H | 193.357 | 193.358 | 193.353–193.359 | -0.33% |
| ring | H2D | 193.200 | 193.201 | 193.197–193.203 | -0.41% |
| allpairs | none | 190.773 | 190.770 | 190.764–190.794 | — |
| allpairs | D2H | 171.250 | 170.996 | 170.851–172.140 | **-10.37%** |
| allpairs | H2D | 188.400 | 188.362 | 188.329–188.656 | -1.26% |

500-repeat 的基线和 treatment 也都很稳定，但 D2H 下降从 20-repeat 的约 28.8%
变为约 10.4%。这是本轮实验最重要的测量学发现：**D2H 竞争效应依赖 memcpy
批次深度/steady-state，不能只报告一个 repeat 配置。**

原始目录：[`main-effect/repetition-1/`](../results/gpu-contention/main-effect/repetition-1/)
至 [`repetition-7/`](../results/gpu-contention/main-effect/repetition-7/)。

## 4. D 阶段：单 GPU 背景注入与 GPU-local 映射

### 4.1 20-repeat D2H：source-local 证据很强

以 allpairs 无背景的 per-source 约 44.96 GB/s 为基线，分别只在 GPU k 上运行
255 MiB D2H。现在每个注入点均有 7 次独立运行；下表给出 aggregate 的中位数，
以及被施压 source 的 7 次均值：

| 背景 | D2D aggregate median | 被施压 source mean | 其他 source | 解释 |
| --- | ---: | ---: | ---: | --- |
| D2H GPU0 | 129.741 | GPU0=32.444 | GPU1–3 约 44.81–44.91 | 主要只有 source0 变慢 |
| D2H GPU1 | 129.957 | GPU1=32.448 | GPU0/2/3 约 44.94–44.95 | 主要只有 source1 变慢 |
| D2H GPU2 | 129.454 | GPU2=32.408 | GPU0/1/3 约 44.95 | 主要只有 source2 变慢 |
| D2H GPU3 | 129.499 | GPU3=32.399 | GPU0–2 约 44.96 | 主要只有 source3 变慢 |
| D2H GPU0–3 | 127.850 | 31.96 左右 | 31.96 左右 | 四个 source 同时变慢 |

单 GPU D2H 将被施压 source 从约 44.96 GB/s 降至约 32.40–32.45 GB/s，下降约
27.8%；7 次 paired aggregate 相对无背景基线为 -27.82% 到 -27.93%。远端
source 基本保持基线。全卡 D2H 的 paired aggregate 中位数相对基线为 -28.90%，
四个 source 均降到约 31.96 GB/s。

这满足计划中的第一种强判据：**D2H 影响跟随施压 GPU 的 source stream 一一对应，
支持 GPU-local source-side data-movement resource 竞争。** 这里的“resource”仍
不能进一步细分为 HBM controller、L2、CE 或内部 fabric。

原始目录：[`per-gpu-injection/repeats-20/`](../results/gpu-contention/per-gpu-injection/repeats-20/)。

### 4.2 20-repeat H2D：方向负对照

7 次独立运行的 aggregate 统计如下：

| 背景 | D2D aggregate median | 7 次范围 | paired 相对无背景基线 |
| --- | ---: | ---: | ---: |
| H2D GPU0 | 179.582 | 176.448–179.636 | median -0.13%；有一个孤立的 -1.87% 点 |
| H2D GPU1 | 179.592 | 179.577–179.633 | -0.11% |
| H2D GPU2 | 179.597 | 179.570–179.611 | -0.12% |
| H2D GPU3 | 179.565 | 179.555–179.581 | -0.14% |
| H2D GPU0–3 | 178.200 | 178.149–178.261 | -0.89% |

单卡 H2D 的被施压 source 均值仍约 44.90 GB/s；GPU0 第 2 轮出现 176.448 GB/s，
且多个非本地 source 同时偏低，形态与 source-local D2H 不一致，因此作为孤立
异常点保留，不作为方向性效应。除该点外，单卡 H2D aggregate 均在 179.55 GB/s
以上。

因此在同一机器、同一 payload 和同一 allpairs 代码下，**D2H 与 H2D 的方向性不是
总 host traffic 字节数差异可以解释的**。背景 aggregate 的量级相近，但 D2H 产生
明显 source-local slowdown，H2D 没有。

原始目录：[`per-gpu-injection/repeats-20/`](../results/gpu-contention/per-gpu-injection/repeats-20/)。
其中 `d2h-repetition-2` 至 `d2h-repetition-7` 和 `h2d-repetition-2` 至
`h2d-repetition-7` 为本次补充的 paired 注入批次。

### 4.3 500-repeat D2H：source-local 仍存在但幅度变小

500-repeat 的 D2H 单卡注入又做了 3 轮。代表性结果为：被施压 source 约
43.35–43.95 GB/s，未施压 source 约 47.67–47.71 GB/s；全卡 D2H 时四个 source
约 42.81–42.87 GB/s。相对于 500-repeat 无背景约 47.69 GB/s，单卡注入的 local
source 下降约 8%–9%，全卡下降约 10%。

所以 source-local 关系不是 20-repeat 的偶然；只是长批次会改变效应幅度。

原始目录：[`per-gpu-injection/d2h-repetition-2/`](../results/gpu-contention/per-gpu-injection/d2h-repetition-2/)
和 [`d2h-repetition-3/`](../results/gpu-contention/per-gpu-injection/d2h-repetition-3/)。

## 5. B 阶段：host copy 第一批矩阵

本轮执行了 6 个 size（4K、1M、16M、64M、255M、512M）、4 个前缀 GPU 集合
（1/2/3/4 卡）和 D2H/H2D，每个点运行约 3 秒、单次采样。它不是计划中全部
15 个 GPU 子集和全部尺寸的最终矩阵，但足以定位当前同步 cadence 下的量级。

### 5.1 255 MiB 和 512 MiB

| direction / size | 1 GPU | 2 GPU | 3 GPU | 4 GPU |
| --- | ---: | ---: | ---: | ---: |
| D2H / 255M | 12.985 | 13.563 | 13.752 | 14.853 |
| H2D / 255M | 12.226 | 12.880 | 12.061 | 14.115 |
| D2H / 512M | 12.816 | 14.047 | 13.858 | 16.296 |
| H2D / 512M | 12.084 | 13.172 | 13.131 | 13.489 |

### 5.2 size 变化

单 GPU D2H 从 4K 的 0.493 GB/s 上升到 1M 的 11.896 GB/s，在 16–64M 后约
13.0 GB/s；单 GPU H2D 从 4K 的 0.466 GB/s 上升到 1M 的 11.062 GB/s，在
16–64M 后约 12.2–12.3 GB/s。当前程序的“每次 memcpy 后同步”限制了小 payload
和多 GPU aggregate 的扩展，不能将这些数值直接当作 PCIe 的最大能力。

原始数据和每个 case 的 command/log/JSON：
[`host-copy-matrix/first-pass/`](../results/gpu-contention/host-copy-matrix/first-pass/)。

## 6. Profiling 与硬件观测

### 6.1 Nsight Systems

P0–P4 均成功生成 `.nsys-rep`，并用同一口径执行了
`cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum`：

| profile | D2D result | P2P operations | Host-copy operations | 关键摘要 |
| --- | ---: | ---: | ---: | --- |
| P0 allpairs none | 190.870 GB/s | 3720，20.863 s total，5.608 ms/op | 无 | 只有 P2P |
| P1 allpairs + D2H all | 173.077 GB/s | 3720，22.601 s total，6.076 ms/op | D2H 422，34.786 s total，82.432 ms/op | 同时出现 P2P/D2H |
| P2 allpairs + D2H GPU0 | 178.825 GB/s | 3720，21.163 s total，5.689 ms/op | D2H 333，7.508 s total，22.547 ms/op | source0=44.706，source1–3≈47.87 |
| P3 allpairs + H2D all | 186.904 GB/s | 3720，20.926 s total，5.625 ms/op | H2D 362，32.565 s total，89.960 ms/op | 同时出现 P2P/H2D |
| P4 ring + D2H all | 192.074 GB/s | 1240，6.880 s total，5.549 ms/op | D2H 238，18.785 s total，78.929 ms/op | ring 负对照 |

P2 使用了独立 wrapper，确保 background 真的只有 GPU0；初始误标的
`p2-allpairs-d2h-gpu0.nsys-rep` 实际使用了全卡 background，不用于结论。有效文件
为 [`p2-allpairs-d2h-gpu0-correct.nsys-rep`](../traces/gpu-contention/p2-allpairs-d2h-gpu0-correct.nsys-rep)。

这些 CLI 摘要确认了操作类型和数量，但 `*_time_sum` 是跨 operation 的累计时间，
不是一个精确的 wall-clock overlap 百分比。精确重叠比例仍应在 GUI/SQLite 时间线上
读取。所有 trace 和摘要位于 [`doc/traces/gpu-contention/`](../traces/gpu-contention/)
及 [`profiling-runs/`](../results/gpu-contention/profiling-runs/)。

### 6.2 dmon、PCIe 和 NVLink counter

在无 profiler 的 allpairs+D2H 全卡运行中：

- mclk 保持约 877 MHz，pclk 从 idle 的 135 MHz 过渡到 1530 MHz；
- 标准 `nvidia-smi dmon` 的 `sm` 列在 steady window 为约 100，`mem` 列约 14–20；
- 这个 `sm=100` 是设备级 busy/utilization 口径，不能当作 CUDA kernel occupancy，
  因为 workload 没有用户 SM kernel；
- PCIe TX 在 steady window 约 3.5 GB/s/GPU，和四卡 background aggregate 约
  12.439 GB/s 同量级，支持 D2H 使用当前 PCIe host path；
- V100 当前驱动的 GPM 扩展指标（SM activity、SM occupancy、DRAM、PCIe、NVLink）
  返回 `-`；
- `nvidia-smi nvlink -gt d` 对全部 link 的 Data Tx/Rx 返回 `N/A`，不能以此证明
  NVLink counter 为 0。

原始 telemetry：[`telemetry/allpairs-d2h-all/`](../results/gpu-contention/telemetry/allpairs-d2h-all/)。

### 6.3 phase-lock 诊断：edge order 与 stream topology trace

为直接观察前述 phase-lock 现象，补采了 3 份 Nsight Systems trace。三份 trace
均使用 4 卡 allpairs、255 MiB、warmup=10、repeats=300、全卡 D2H background，
采样工具为 Nsight Systems 2024.7.1，命令口径为：

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --force-overwrite=true --output=<trace-name> \
  bash scripts/run_phase_lock_diagnostic.sh --deviceList=0,1,2,3 \
  --edgeOrders=<edge-order> --streamModes=<stream-mode> \
  --repeatsList=300 --scenarios=d2h-all --runs=1 --size=255M \
  --outputRoot=<raw-result-directory>
```

| trace | `edgeOrder` | `streamMode` | D2D aggregate（Nsight 运行） |
| --- | --- | --- | ---: |
| [`source-major-per-source-d2h.nsys-rep`](../traces/gpu-contention/phase-lock-diagnostic/source-major-per-source-d2h.nsys-rep) | source-major | per-source | 173.215 GB/s |
| [`destination-major-per-source-d2h.nsys-rep`](../traces/gpu-contention/phase-lock-diagnostic/destination-major-per-source-d2h.nsys-rep) | destination-major | per-source | 173.726 GB/s |
| [`source-major-per-edge-d2h.nsys-rep`](../traces/gpu-contention/phase-lock-diagnostic/source-major-per-edge-d2h.nsys-rep) | source-major | per-edge | 570.809 GB/s |

`cuda_gpu_mem_time_sum` 的摘要如下。这里的累计时间跨越不同 GPU memory
operation，不能直接当作 wall-clock overlap 百分比；它用于确认 operation 数量和
平均时长的变化：

| trace | P2P count / total / average | D2H count / total / average |
| --- | --- | --- |
| source-major / per-source | 3720 / 22.637 s / 6.085 ms | 452 / 36.311 s / 80.334 ms |
| destination-major / per-source | 3720 / 22.647 s / 6.088 ms | 459 / 37.181 s / 81.004 ms |
| source-major / per-edge | 3720 / 20.790 s / 5.589 ms | 228 / 18.381 s / 80.618 ms |

`cuda_api_sum` 还确认了 stream 拓扑：前两份 trace 的 `cudaStreamCreate` 总数为
8（4 条 D2D source stream + 4 条 D2H stream），`per-edge` trace 为 16（12 条
D2D edge stream + 4 条 D2H stream）。因此 `per-edge` 的吞吐提升不是换了 payload
或减少了 P2P 操作，而是取消了每个 source GPU 内 3 条 outgoing copy 的显式串行
约束。

无 profiler 的正式矩阵中，300-repeat 的 3 次均值为：

| `edgeOrder` | `streamMode` | 无背景 | 全卡 D2H | D2H 相对下降 |
| --- | --- | ---: | ---: | ---: |
| source-major | per-source | 188.864 | 159.715 | -15.44% |
| destination-major | per-source | 188.841 | 159.917 | -15.32% |
| source-major | per-edge | 581.875 | 578.200 | -0.63% |
| destination-major | per-edge | 581.877 | 578.201 | -0.63% |

这组结果支持两个阶段性结论：

1. 在当前 `per-source` 实现中，改变全局 host 提交顺序几乎没有改变结果。原因是
   `destination-major` 虽然交错了不同 source stream 的 API 提交，但每个 source
   stream 内的目的端相对顺序仍未改变；因此它不能证明“同一 source stream 内的
   三条 edge 顺序”不重要，只能证明当前这类跨 source 的交错不足以解除 phase-lock。
2. `per-edge` 将每条有向边放到独立 stream 后，300-repeat 的 D2H 下降从约 15.4%
   降到约 0.63%，且 20-repeat 也从约 127.6 GB/s 提升到约 578.0 GB/s。这与
   “D2H 触发有限宽度的调度/仲裁干扰窗口，source stream 的串行队列通过
   head-of-line blocking 反复暴露慢边；拆分 edge stream 后串行依赖和有限 eligible
   work 约束被解除”的解释一致。

这里的 `phase-lock` 不应理解为四张 GPU 上的 P2P 必须在同一时刻精确启动。对三份
SQLite trace 重新统计后，`source-major/per-source` 的 3720 个 P2P 中位数为约
5.532 ms，但有 262 个大于 8 ms，慢样本集中在约 13.4 ms；
`destination-major/per-source` 有相同形态，265 个 P2P 大于 8 ms。慢波内部不同
GPU 的 P2P 启动时间已经可以相差约 0.1-1 ms，接近转折时还出现只有 GPU1/GPU3
继续产生慢 P2P、GPU0/GPU2 已经恢复的阶段。因此更准确的术语是
`queue-phase coupling` 或 wave-level cadence coupling：只要多个 source queue
仍落入同一个有限宽度干扰窗口，就可能出现慢波，不要求完全时间对齐。

慢边也不是均匀随机分布。`source-major/per-source` 的 262 个慢样本中，242 个位于
每个 source stream 的第二条 outgoing edge，20 个位于第三条，第一条没有慢样本；
`destination-major/per-source` 得到相同的约 92%“第二队列位置”集中性。这说明
当前现象与同一 source stream 内的 FIFO 位置和提交 cadence 强相关，而不是单纯的
随机 HBM 带宽抖动。

`source-major/per-edge` trace 中同样有 3720 个 P2P，但全部位于约
5.58-5.61 ms，没有大于 8 ms 的样本。因此 `per-edge` 不只是用并行 makespan
隐藏单条慢边；增加独立 stream 和 eligible P2P 后，底层调度状态发生了变化，长尾
本身也消失了。该实验同时改变了 stream 依赖、eligible work 数量以及潜在的
CE/NVLink path 调度，尚不能仅凭这一组结果在三者之间确定唯一原因。

原始矩阵目录：[`phase-lock-diagnostic/20260826T094355Z-197181/`](../results/gpu-contention/phase-lock-diagnostic/20260826T094355Z-197181/)。
trace 对应的命令、JSON 和后台日志目录：[`phase-lock-diagnostic/traces/20260826T094945Z/`](../results/gpu-contention/phase-lock-diagnostic/traces/20260826T094945Z/)。

## 7. 当前证据能支持什么

### 支持的判断

1. 在 20-repeat 和 500-repeat 两种协议中，allpairs+D2H 都可稳定复现下降；ring
   几乎不受影响，H2D 影响很小。
2. D2H 只施加在 GPU k 时，主要是 source=k 的 allpairs stream 变慢；这比单纯的
   全局 aggregate 下降更支持 GPU-local source-side data-movement competition。
3. 20-repeat 的原始 -28.8% 现象不是一次偶然；7 次结果的 D2H median 为 128.037
   GB/s，离散度远小于效应量。
4. Nsight Systems 确认 P2P 与 host memcpy 两类 GPU memory operation 同时存在，
   并且 P2P operation 的平均时间在 D2H treatment 中增加。
5. 依据理论分析文档的读写计数，allpairs 的本地 HBM 估算仍远低于 V100 官方
   900 GB/s 峰值，因此不是“总 HBM 带宽已经达到 900 GB/s”的简单解释。
6. 分块 event 诊断显示 20/500 差异来自约前 100 次 round 的阶段性状态；D2H
   source-local 竞争在初始阶段最强，长批次后趋向无背景 steady-state，不能把
   两个 repeat 配置合并成一个固定 slowdown。
7. `chunkRepeats=1/5/20` 的边界对照均观察到约第 100 次 round 的阶段转折；因此
   转折不是单一 chunk 分组造成，但 event 插入会改变绝对带宽，仍需独立的逐次
   synchronize/高分辨率时钟对照来判断其触发机制。
8. 100 ms clock/pstate 对照中，D2H 场景在 graphics clock 已为 1530 MHz 时仍有
   相同的早期低阶段；graphics clock 不是充分解释，但该采样仍不能定位具体硬件
   仲裁单元。
9. 逐次 synchronize 对照使 20/500 收敛到相同低阶段，并使 GPU0-only D2H 的
   source-local 差异扩散到所有 source event 区间；因此队列深度是重要控制变量，
   但 sync 协议本身不适合替代异步协议做 local 归因。
10. 新增的 edge-order 对照中，`source-major` 与 `destination-major` 在
    `per-source` 模式下结果几乎一致；这说明仅交错不同 source stream 的 host
    提交顺序不足以解除同一 source stream 内的 FIFO 依赖和 queue-phase coupling。
11. 新增的 `per-edge` 对照将 12 条有向 P2P 边拆到 12 条独立 stream 后，300-repeat
    D2H 相对下降由约 15.4% 降至约 0.63%，且 trace 中约 13.4 ms 的单条 P2P
    长尾也消失。它支持“同一 source stream 的串行依赖、head-of-line blocking 和
    有限 eligible work 参与了现象”，但不支持“只有各 GPU copy 完全时间对齐才会
    变慢”，也仍不能据此定位具体 CE/NVLink 仲裁单元。
12. Q1 的 `per-edge + source-chain` 在保留 source 内严格 edge 依赖的情况下，D2H
    相对下降仅约 0.50%，没有恢复 per-source 的约 14.78% 下降；因此 FIFO/HOL
    不是单独充分条件，独立 stream 带来的 stream identity、eligible work 或内部
    queue/engine mapping 变化至少是必要的参与因素。
13. Q2 的 1/2/3 streams-per-source 扫描显示，2 stream 并未消除竞争：300-repeat
    D2H 相对下降约 37.57%，trace 中 1236 个 P2P（每个 round 的 4 个 source GPU）
    大于 8 ms；3 stream 才将下降压到约 0.64% 且消除长尾。这给出了当前 255M
    workload 的明显并行度阈值，但仍不能把阈值直接映射为具体 CE 或 NVLink 资源。
14. Q3 的六种单 stream edge permutation 中，300-repeat D2H 下降均约 14.79%–15.42%；
    六份 trace 的慢样本约 92.2%–92.5% 位于第二队列位置，且主要慢 edge 会随该
    位置移动。因此当前证据更支持 FIFO position/cadence，而不是固定的
    source-destination pair。
15. Q4 的一次性 source 启动偏移没有消除竞争：20-repeat 的 D2H 下降在六个配置中为
    28.44%–28.95%，300-repeat 为 12.69%–24.37%。trace 中 delay kernel 的实际持续
    时间与 250–12000 us 目标一致，第一条 measured P2P 的 source 起始 span 可从
    0.599 ms 增至 11.925 ms，但慢样本仍有 250–441 个且约 92.0%–95.5% 位于第二
    queue position；按 round index 的 measured P2P start-span 的 p50 还从 52.344 ms
    增至 860.915 ms。因此“只有四卡 P2P 精确同时启动才会变慢”不成立；一次性初始
    phase 不是充分条件，也不是当前长尾的唯一解释。该实验不能给出 steady-state
    phase-tolerance 窗口，因为 offset 没有在每个 round 重复注入。
16. Q2b 的三种“一个 source-major edge 独立、另外两个 edge 共享”映射得到几乎相同
    的结果：20-repeat D2H 下降为 37.55%–37.67%，300-repeat 为 37.53%–37.57%。
    三份 300-repeat trace 都有 8 条 stream、1236 个含 warmup 的慢 P2P（measured
    为 1200 个）和 309 个慢波；慢样本 100% 位于实际共享 stream 的第二个 operation
    （分析器的 queue position 1）。慢的物理 edge 会随被共享的第二个 source-major
    edge 改变，而被独立隔离的 edge 不出现该长尾。因此问题关键是剩余的两-copy FIFO
    队列，而不是“哪一条 copy 被隔离”；仅增加到 2 条 stream 不足以进入三条独立
    eligible stream 的无长尾状态。
17. 在保持 D2D=255M 和 two-stream assignment 不变、仅把全卡 D2H background 改为
    16K 后，三种 assignment 的 300-repeat D2D 下降仅为 0.06%–0.12%，20-repeat
    为 -1.16%–0.19%（36 个 case 全部成功）。因此此前的 37.5% slowdown 需要足够
    高的持续 D2H 数据移动强度；“存在 D2H operation”本身不是充分条件。该矩阵没有
    采集 trace，所以它证明的是 aggregate slowdown 消失，不能单独证明每条 P2P
    长尾也消失。
18. 将同一口径的 background D2H 增加到 64K 后，三种 assignment 的 300-repeat D2D
    下降仍仅为 0.32%–0.45%，20-repeat 为 -0.94%–0.46%；背景 D2H 在 D2D 并行期间
    约为 7.35–7.74 GB/s，仍未复现 255M background 的约 37.5% slowdown。在 Q2e/Q2f
    执行前，这只说明已测点的转折位于 64K 与 255M 之间；后续 16M 和 8M 结果已将
    明显竞争区间进一步收窄。该结果仍提示单次 D2H 粒度和同步 cadence 可能共同作用，
    不能把“D2H 存在”或单一 aggregate GB/s 直接等同于 CE 饱和。该矩阵同样未采集 trace。
19. 将 background D2H 提高到 16M 后，三种 assignment 的 300-repeat D2D 下降为
    33.34%–33.51%，20-repeat 为 32.59%–33.29%；背景 D2H 在 D2D 并行期间约为
    12.71–13.04 GB/s，已重新进入明显竞争区间。与 64K 的 0.32%–0.45% 下降对比，
    当前离散实验把强度转折收窄到 64K 与 16M 之间；但 background worker 每次同步，
    所以该区间同时混合了单次传输量、D2H 数据率和 operation cadence。
20. 将 background D2H 进一步降到 8M 后，三种 assignment 的 300-repeat D2D 下降仍为
    30.90%–31.07%，20-repeat 为 30.12%–31.06%；背景 D2H 在 D2D 并行期间约为
    12.78–12.86 GB/s。8M 比 16M 的 slowdown 略弱但仍然显著，因此当前可观测的
    强度转折进一步收窄到 64K 与 8M 之间；这仍不是单一 CE 字节带宽阈值的证明，且
    两轮矩阵都未采集 trace，不能据此推断慢 P2P 的 queue-position 分布。
21. 将 background D2H 再降到 4M 后，三种 assignment 的 300-repeat D2D 下降为
    29.02%–29.15%，20-repeat 为 27.43%–28.99%；背景 D2H 在 D2D 并行期间约为
    12.79–12.95 GB/s。4M 相比 8M/16M 有小幅恢复，但仍明显低于 clean，说明当前
    明显竞争区间至少延伸到 4M；结合 64K 的近似无影响结果，离散实验只能将转折
    保持在 64K 与 4M 之间。本轮按要求未执行 1M，且没有采集 trace。

### 仍不能声称的判断

- 不能仅凭这些结果定位到具体 HBM channel、L2 slice、CE instance 或 NVLink
  scheduler；
- 不能把 `nvidia-smi dmon sm=100` 当作 SM occupancy；
- 不能用当前 `N/A` 的 NVLink data counter 计算精确 NVLink utilization；
- 不能把 20-repeat 的 -28.8% 外推为任意运行时长都会出现的固定下降比例；
- 不能把 host-copy 第一批 aggregate 当作完整 PCIe/NUMA capacity characterization。

## 8. 下一步执行顺序

Q1-Q4 以及 Q2b/Q2c/Q2d/Q2e/Q2f/Q2g 已按以下顺序完成。结果表明，在进入 HBM/L2 kernel 之前，仍应把解释限定在
以下三个同时被 `per-edge` 改变、且目前尚未被硬件计数器单独拆开的因素：

1. 同一 source 的 edge 是否存在严格 FIFO 依赖；
2. 每个 source 同时有多少个 eligible P2P；
3. 独立 stream 是否改变 CE/内部队列或 NVLink path 的选择。

### 8.1 通用实验口径

下面 Q1-Q4、Q2b、Q2c、Q2d、Q2e、Q2f、Q2g 使用同一固定口径，除被测变量外不改变其他参数：

```text
pattern=allpairs
devices=0,1,2,3
size=255M
warmup=10
repeats=20,300
background=none,d2h-all
independent repetitions=3（筛选），显著配置补到 7 次
```

无 profiler 运行用于正式 aggregate/per-source 性能；每个配置另采 1 份 Nsight
Systems trace，用 SQLite 提取每条 P2P 的 source、destination、stream、start、end、
duration 和所在 round/queue position。至少报告：

```text
aggregate GB/s
per-source GB/s
P2P duration p50/p90/p99/max
duration > 8 ms 的数量和比例
慢边按 source/destination/queue-position 的计数
每个慢波包含的 GPU 数和波内 start-time span
background aggregate 及 per-GPU GB/s
```

所有配置继续使用异步 batch 作为主协议。`syncEachIteration=1` 只保留为队列深度
控制，不能和异步协议的绝对带宽或 source-local 结果混合解释。

### 8.2 Q1：独立 stream，但用 event 恢复 source-chain 依赖（最高优先级）

目的：区分“解除串行依赖”与“stream identity/engine mapping”两个因素。

在 `per-edge` 的 12 条 stream 上增加可选模式：

```text
--streamDependency=none|source-chain
```

`source-chain` 对每个 source 的三条 edge 使用 CUDA event 建立：

```text
edge-position-0
-> event record
-> edge-position-1 stream waits event
-> event record
-> edge-position-2 stream waits event
-> 下一 round 的 position-0 等待上一 round position-2
```

三条 edge 仍属于三个不同 stream，但 GPU 可见的执行依赖与原 `per-source` FIFO
一致。事件必须在 measured range 前完成对象创建，不能在每个 round 动态创建/销毁。

比较：

```text
per-source                         单 stream + 串行依赖
per-edge + source-chain            三 stream + 串行依赖
per-edge + dependency none         三 stream + 无串行依赖
```

判读：

- `source-chain` 恢复约 13.4 ms 长尾和 D2H slowdown：严格依赖/head-of-line
  blocking 是主条件，stream identity 不是关键；
- `source-chain` 仍与无依赖 per-edge 一样快：不同 stream 引起的 engine/queue
  mapping 更重要；
- 只恢复部分 slowdown：FIFO 依赖和 stream/engine mapping 都参与。

建议结果目录：

```text
doc/results/gpu-contention/queue-phase-diagnostic/q1-source-chain/
doc/traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/
```

#### Q1 已执行结果（2026-08-26）

筛选矩阵为 3 个配置 × 2 个 repeats × 2 个背景场景 × 3 次重复，完整结果见
[`Q1 原始目录`](../results/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T102517Z-219542/)。300-repeat 的均值如下：

| 配置 | 无背景 (GB/s) | 全卡 D2H (GB/s) | 相对下降 |
| --- | ---: | ---: | ---: |
| per-source + none | 188.935 | 161.017 | 14.78% |
| per-edge + none | 581.875 | 578.193 | 0.63% |
| per-edge + source-chain | 193.854 | 192.892 | 0.50% |

Q1 的三份 trace 和逐条 SQLite 分析位于 [`Q1 trace 目录`](../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/) 及 [`Q1 trace analysis`](../results/gpu-contention/queue-phase-diagnostic/q1-source-chain/trace-analysis/20260826T103320Z/)。结果为：

- per-source trace 有 3720 个 P2P、4 条 stream、260 个 duration >8 ms 的 operation，
  p99=13.465 ms、max=13.502 ms；68 个慢波均包含 4 个 source GPU，慢波 start span
  的 p50/p90/max 为 2.089/2.576/3.138 ms；
- per-edge + none 有 12 条 stream，3720 个 P2P 全部约 5.586–5.605 ms，无慢波；
- per-edge + source-chain 同样有 12 条 stream，3720 个 P2P 全部约 5.531–5.583 ms，
  仍无慢波。

因此 Q1 选择“不同 stream 引起的 eligible work / 内部 queue 或 engine mapping 更
重要”这一分支，但结论仍不具体指向 CE 或 NVLink path；source-chain 与 per-source
同时改变了 stream identity，后续仍需 Q2 的 1/2/3 stream 扫描拆分这两个因素。

### 8.3 Q2：每个 source 的 eligible stream 数量扫描

目的：确定长尾是否存在并行度阈值。

将当前二值 `per-source|per-edge` 扩展为：

```text
--streamsPerSource=1|2|3
```

建议映射：

- 1：三条 outgoing edge 共用一个 stream；
- 2：queue position 0/2 使用 stream 0，position 1 使用 stream 1；
- 3：每条 outgoing edge 独立 stream。

该实验使用 `streamDependency=none`，保持所有 stream 可独立推进。对每个并行度统计
慢 P2P 数量和 p99，而不只比较 aggregate。

判读：

- 从 1 到 2 个 stream 即消除长尾：只需一个替代 eligible work 即可避开阻塞；
- 必须 3 个 stream 才消除：可能需要覆盖全部 destination/NVLink path；
- 随 stream 数平滑改善：更像队列深度/调度利用率效应；
- aggregate 改善但单条长尾仍在：主要是 makespan 掩盖，而不是慢点消失。

建议结果目录：

```text
doc/results/gpu-contention/queue-phase-diagnostic/q2-stream-count/
doc/traces/gpu-contention/queue-phase-diagnostic/q2-stream-count/
```

#### Q2 已执行结果（2026-08-26）

完整矩阵见 [`Q2 原始目录`](../results/gpu-contention/queue-phase-diagnostic/q2-stream-count/20260826T104819Z-233894/)，trace 和逐条分析见 [`Q2 trace 目录`](../traces/gpu-contention/queue-phase-diagnostic/q2-stream-count/20260826T105130Z/)。300-repeat 的 3 次均值如下：

| streams/source | 无背景 (GB/s) | 全卡 D2H (GB/s) | 相对下降 | >8 ms P2P | 慢波数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 188.945 | 160.254 | 15.19% | 259 | 68 |
| 2 | 260.516 | 162.653 | 37.57% | 1236 | 309 |
| 3 | 581.872 | 578.164 | 0.64% | 0 | 0 |

2-stream 的 position 0/2 共享一个 stream，position 1 使用独立 stream。trace 显示
所有慢样本均为每个 source 的第三条 edge（即共享 stream 中的第二个 operation），
而独立 stream 上的中间 edge 没有同样长尾；因此 1→2 没有打破剩余共享 stream 的
后继队列耦合，只有 2→3 才完全消除慢波。详细的 p50/p90/p99/max、stream 映射和
逐条记录在上述 trace-analysis JSON 中。

Q2 支持“当前 255M allpairs 需要三条独立 eligible stream 才能避开该慢波”的判断，
但它仍不能区分是 stream identity、CE/内部 queue mapping，还是具体 pair/path
调度导致共享队列中的后继 operation 变慢。Q2b 进一步验证该慢点跟随共享 stream
中的第二个 operation，而不是跟随被隔离的固定 edge；下一步继续做 Q3 的六种单
stream edge permutation。

### 8.3.1 Q2b：2-stream one-vs-two copy assignment 消融

目的：在总 stream 数固定为 2 的情况下，区分“某个具体 source-major edge 较特殊”
和“任何两-copy 共享 FIFO 队列都会产生慢点”。三种 assignment 的值对应三个
source-major edge 位置，`0`/`1` 是两个 stream slot：

```text
0,1,1    position 0 独立，position 1/2 共享
1,0,1    position 1 独立，position 0/2 共享
1,1,0    position 2 独立，position 0/1 共享
```

固定 `streamsPerSource=2`、`streamDependency=none`、`edgeOrder=source-major`，每个
配置执行 20/300 repeats、无背景和全卡 D2H，各 3 次。完整逐 case 结果见
[`Q2b 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154321Z-303566/)。

| assignment | 独立 edge 位置 | 20-repeat 无背景 | 20-repeat 全卡 D2H | 下降 | 300-repeat 无背景 | 300-repeat 全卡 D2H | 下降 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 0 | 260.492 | 162.686 | 37.55% | 260.422 | 162.674 | 37.53% |
| `1,0,1` | 1 | 260.455 | 162.397 | 37.65% | 260.495 | 162.620 | 37.57% |
| `1,1,0` | 2 | 260.482 | 162.350 | 37.67% | 260.421 | 162.647 | 37.54% |

单位为 GB/s。三种 assignment 的 clean 和 D2H aggregate 都在同一范围内，20 与
300 repeats 也没有显著差异。因此改变“哪个 copy 独立”不是解除当前竞争的有效
控制变量。

三份 300-repeat D2H trace 的逐条分析见
[`Q2b trace analysis`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/trace-analysis/20260826T154647Z/)，
原始 `.nsys-rep` 见 [`Q2b trace 目录`](../traces/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154647Z/)：

| assignment | stream 数 | 全部 P2P | >8 ms（含 warmup） | measured >8 ms | 慢波数 | queue position 1 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 8 | 3720 | 1236 | 1200 | 309 | 100% |
| `1,0,1` | 8 | 3720 | 1236 | 1200 | 309 | 100% |
| `1,1,0` | 8 | 3720 | 1236 | 1200 | 309 | 100% |

`queue position 1` 指每条实际 CUDA stream 上的第二个 operation。对应的慢 edge 为：

```text
0,1,1 -> 0->3, 1->3, 2->3, 3->2
1,0,1 -> 0->3, 1->3, 2->3, 3->2
1,1,0 -> 0->2, 1->2, 2->1, 3->1
```

所以慢点随共享 stream 中的第二个操作移动；独立 stream 上的 edge 没有同样长尾。
Q2b 将 Q2 的结论从“2 stream 不够”具体化为：只保留一个两-copy 共享队列时，
后继 operation 仍会被 D2H 背景放大；需要把三个 outgoing edge 全部放到独立
eligible stream，才进入 Q2 中观察到的无长尾状态。随后用 Q2c/Q2d 降低背景 D2H 的
单次传输量，检验该效应是否依赖 D2H 强度。该实验仍不能把现象定位到具体 CE
instance、L2 slice、HBM channel 或 NVLink scheduler。

### 8.3.2 Q2c：保持 D2D=255M、降低 D2H background 到 16K

目的：在不改变 two-stream D2D workload 和 assignment 的情况下，降低背景 D2H 的
单次通信量，判断 255M background 下的竞争是否需要较高的持续 D2H 数据移动强度。

本轮使用 `--d2dSize=255M --backgroundSize=16K`，其他参数与 Q2b 相同：三种
assignment、20/300 repeats、无背景/全卡 D2H、每个配置 3 次。完整结果见
[`Q2c 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T162155Z-326564/)。

| assignment | 20-repeat clean / D2H | 20-repeat 下降 | 300-repeat clean / D2H | 300-repeat 下降 | 300-repeat D2H background |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 257.111 / 260.097 | -1.16% | 260.454 / 260.302 | 0.06% | 3.435 |
| `1,0,1` | 260.459 / 260.115 | 0.13% | 260.510 / 260.207 | 0.12% | 3.420 |
| `1,1,0` | 260.481 / 259.995 | 0.19% | 260.436 / 260.120 | 0.12% | 3.495 |

单位为 GB/s。20-repeat 的第一行 clean 均值包含一次 250.366 GB/s 的启动样本，
因此以 300-repeat 及其余重复作为稳定性参考。与 Q2b 中 background=255M 时约
37.5% 的下降相比，background=16K 后 aggregate D2D 基本恢复到 clean 水平。

该结果支持“D2H 的持续数据移动/CE 数据路径占用强度是重要条件”，不支持“只要
存在 D2H copy 就必然竞争”。本轮没有采集 Nsight Systems trace，因此暂不把
aggregate 恢复等同于每条 P2P 的长尾已完全消失；若需要确认，应对 300-repeat 的
16K background 采集一份代表性 trace。

### 8.3.3 Q2d：保持 D2D=255M、background D2H 改为 64K

目的：在 Q2c 的 16K 低压点基础上提高 background D2H 单次传输量，寻找从“无明显
影响”到原始 255M slowdown 的过渡区间。

本轮使用 `--d2dSize=255M --backgroundSize=64K`，其他参数与 Q2b/Q2c 相同：三种
assignment、20/300 repeats、无背景/全卡 D2H、每个配置 3 次。完整结果见
[`Q2d 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T162713Z-330735/)。

| assignment | 20-repeat clean / D2H | 20-repeat 下降 | 300-repeat clean / D2H | 300-repeat 下降 | 300-repeat D2H background |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 257.044 / 259.467 | -0.94% | 260.435 / 259.598 | 0.32% | 7.354 |
| `1,0,1` | 260.435 / 259.446 | 0.38% | 260.482 / 259.432 | 0.40% | 7.350 |
| `1,1,0` | 260.499 / 259.294 | 0.46% | 260.455 / 259.276 | 0.45% | 7.743 |

单位为 GB/s。与 Q2b 的 background=255M 时约 37.5% 下降相比，64K background 下
aggregate D2D 仍保持在约 259–260 GB/s；它没有复现 two-stream 的明显竞争。结合
Q2c 的 16K 结果，当前可观测 transition 位于 64K 和 255M 之间，或者由单次传输
粒度与 worker 的“每次 memcpy 后同步”cadence 共同决定；该表述是 Q2e/Q2f 之前的
中间结论。由于本轮没有 trace，不能把 aggregate 恢复进一步解释为 P2P 长尾必然消失。

### 8.3.4 Q2e：保持 D2D=255M、background D2H 改为 16M

目的：在 Q2d 的 64K 低压点基础上继续提高 background D2H 单次传输量，确认明显
two-stream slowdown 是否在中等背景强度下重新出现。

本轮使用 `--d2dSize=255M --backgroundSize=16M`，其他参数与 Q2b/Q2c/Q2d 相同：
三种 assignment、20/300 repeats、无背景/全卡 D2H、每个配置 3 次。完整结果见
[`Q2e 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T163221Z-334532/)。

| assignment | 20-repeat clean / D2H | 20-repeat 下降 | 300-repeat clean / D2H | 300-repeat 下降 | 300-repeat D2H background |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 257.356 / 173.486 | 32.59% | 260.456 / 173.464 | 33.40% | 13.035 |
| `1,0,1` | 260.416 / 174.966 | 32.81% | 260.548 / 173.239 | 33.51% | 12.822 |
| `1,1,0` | 260.462 / 173.748 | 33.29% | 260.447 / 173.611 | 33.34% | 12.709 |

单位为 GB/s。36 个 case 全部成功。16M background 在 D2D 并行期间约为 12.71–13.04
GB/s，D2D aggregate 从 clean 的约 260.4–260.5 GB/s 降至约 173.2–173.6 GB/s，
说明高 D2H 强度下 two-stream 竞争重新出现。三种 assignment 的下降幅度相近，未
显示“哪一条 source-major edge 被独立放置”是主要变量。

与 Q2d 的 64K 结果相比，当前离散实验把明显 slowdown 的强度转折收窄到 64K 与 16M
之间。不过 background worker 每次 memcpy 后执行 `cudaStreamSynchronize`，所以该
区间同时受单次传输量、D2H 数据率和 operation cadence 影响，不能直接当作 CE 的
纯字节带宽阈值。本轮没有采集 trace，结论限于 aggregate slowdown。

### 8.3.5 Q2f：保持 D2D=255M、background D2H 改为 8M

目的：在 64K 与 16M 之间增加一个 8M 点，进一步收窄明显竞争出现的背景强度区间。

本轮使用 `--d2dSize=255M --backgroundSize=8M`，其他参数与 Q2b-Q2e 相同：三种
assignment、20/300 repeats、无背景/全卡 D2H、每个配置 3 次。完整结果见
[`Q2f 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T163827Z-338551/)。

| assignment | 20-repeat clean / D2H | 20-repeat 下降 | 300-repeat clean / D2H | 300-repeat 下降 | 300-repeat D2H background |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 256.113 / 178.974 | 30.12% | 260.432 / 179.960 | 30.90% | 12.860 |
| `1,0,1` | 260.475 / 180.783 | 30.59% | 260.497 / 179.558 | 31.07% | 12.780 |
| `1,1,0` | 260.494 / 179.583 | 31.06% | 260.433 / 179.907 | 30.92% | 12.820 |

单位为 GB/s。36 个 case 全部成功。8M background 在 D2D 并行期间约为 12.78–12.86
GB/s，300-repeat 的 D2D slowdown 仍为 30.90%–31.07%；相较 16M 的 33.34%–33.51%
略弱，但已经明显偏离 clean。结合 Q2d 的 64K 结果，本轮把明显竞争转折收窄到
64K 与 8M 之间；Q2g 的 4M 结果又将总体可观测区间进一步收窄到 64K 与 4M 之间。

8M 和 16M 两轮都没有采集 Nsight Systems trace，因此只能确认 aggregate D2D
slowdown，不能单独说明每条慢 P2P 的 queue position。由于 background worker 的
同步 cadence 随 size 改变，64K–8M 区间仍应理解为当前实现和协议下的操作强度区间。

### 8.3.6 Q2g：保持 D2D=255M、background D2H 改为 4M

目的：在 64K 与 8M 之间继续增加一个 4M 点，判断明显 two-stream slowdown 是否仍然
存在。本轮只执行 4M，不执行 1M。

本轮使用 `--d2dSize=255M --backgroundSize=4M`，其他参数与 Q2b-Q2f 相同：三种
assignment、20/300 repeats、无背景/全卡 D2H、每个配置 3 次。完整结果见
[`Q2g 原始目录`](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T164908Z-344216/)。

| assignment | 20-repeat clean / D2H | 20-repeat 下降 | 300-repeat clean / D2H | 300-repeat 下降 | 300-repeat D2H background |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 256.820 / 186.379 | 27.43% | 260.460 / 184.871 | 29.02% | 12.786 |
| `1,0,1` | 260.475 / 186.666 | 28.34% | 260.525 / 184.572 | 29.15% | 12.853 |
| `1,1,0` | 260.504 / 184.984 | 28.99% | 260.466 / 184.707 | 29.09% | 12.953 |

单位为 GB/s。36 个 case 全部成功。4M background 在 D2D 并行期间约为 12.79–12.95
GB/s，仍然触发明显竞争；300-repeat D2D 从 clean 的约 260.46–260.52 GB/s 降至
约 184.57–184.87 GB/s。相较 8M 的 30.90%–31.07% 和 16M 的 33.34%–33.51%，4M
有小幅恢复，但没有回到 64K 的 0.32%–0.45% 区间。

因此当前离散实验将明显竞争的强度转折保持在 64K 与 4M 之间；由于本轮未执行 1M，
不能继续向更小粒度收窄。background worker 每次 D2H 后同步，size 同时改变单次传输
量、D2H 数据率和 operation cadence，不能把 64K–4M 区间直接解释为单一 CE 阈值。
本轮未采集 Nsight Systems trace，结论限于 aggregate D2D slowdown。

### 8.4 Q3：单 stream 内 edge permutation

目的：判断慢边跟随 FIFO 位置还是固定物理 GPU pair。

保持每个 source 只有一个 stream，增加显式 permutation。每个 source 有三个 outgoing
destination，执行全部 6 种相对顺序；至少包括：

```text
0,1,2    forward
2,1,0    reverse
1,2,0    rotate-left
2,0,1    rotate-right
0,2,1
1,0,2
```

建议接口：

```text
--edgePermutation=0,1,2
```

这里的数字表示该 source 排序后 destination list 的相对位置，不是固定 GPU id。
JSON 必须输出每个 source 的实际 destination 顺序。

判读：

- 慢边随第二队列位置移动：强支持 queue cadence/position 机制；
- 慢边固定在某个 source-destination pair：支持特定 NVLink/目的端路径；
- 慢边同时依赖位置和物理 edge：队列 cadence 与拓扑路径共同参与。

建议结果目录：

```text
doc/results/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/
doc/traces/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/
```

#### Q3 已执行结果（2026-08-26）

完整矩阵见 [`Q3 原始目录`](../results/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/20260826T110853Z-249443/)，六份 trace 及其命令见 [`Q3 trace 目录`](../traces/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/20260826T111616Z/)。六种 permutation 的 300-repeat 结果为：

| permutation | 无背景 (GB/s) | 全卡 D2H (GB/s) | 相对下降 | >8 ms P2P | 第二队列位置占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,1,2 | 188.963 | 160.430 | 15.10% | 257 | 92.22% |
| 2,1,0 | 188.860 | 159.829 | 15.37% | 266 | 92.48% |
| 1,2,0 | 188.797 | 160.868 | 14.79% | 267 | 92.51% |
| 2,0,1 | 188.848 | 159.733 | 15.42% | 263 | 92.40% |
| 0,2,1 | 188.839 | 160.285 | 15.12% | 261 | 92.34% |
| 1,0,2 | 188.957 | 160.185 | 15.23% | 268 | 92.54% |

六种顺序的 aggregate 都仍在同一低性能区间，但主要慢边跟随 permutation 中的
第二个元素移动。例如 GPU0 的顺序 `1→2→3` 时主要慢 `0→2`，顺序 `2→3→1`
时主要慢 `0→3`，顺序 `3→1→2` 时主要慢 `0→1`。这排除了“单一固定 pair
必然慢”的简单解释，支持慢点由 source stream 的 FIFO position/cadence 触发。
Q3 仍不能单独区分 queue mapping 与具体链路仲裁；下一步进入 Q4 的 source offset
扫描。

### 8.5 Q4：可控 source 启动偏移扫描

目的：直接测量 queue-phase coupling 的容忍窗口，验证完全时间对齐不是必要条件。

保持 `streamsPerSource=1` 和相同 edge 顺序，在 measured batch 开始前给四个 source
stream 注入不同的一次性 device-side 延迟。建议使用读取 `%globaltimer` 的单线程
delay kernel，避免用 host `sleep` 或额外 CE memcpy 制造偏移。delay kernel 只在
batch 开始前运行一次，不在每个 round 重复。

第一轮配置：

```text
GPU0,1,2,3 offset = 0,0,0,0 us
GPU0,1,2,3 offset = 0,250,500,750 us
GPU0,1,2,3 offset = 0,500,1000,1500 us
GPU0,1,2,3 offset = 0,1000,2000,3000 us
GPU0,1,2,3 offset = 0,2000,4000,6000 us
GPU0,1,2,3 offset = 0,4000,8000,12000 us
```

建议接口：

```text
--sourceOffsetsUs=0,500,1000,1500
```

delay kernel 本身可能改变 graphics clock，因此每个 offset 配置都必须同时执行 clean
和 D2H，且在 measured start event 之前结束。报告初始目标 offset、trace 中实际第一条
P2P start offset，以及随后每个 round 的相位漂移；本轮使用 `sourceOffsetDelayKernels`
和 round-level P2P start-span 完成这两项校验。

判读：

- 小于某一 offset 时仍形成慢波，超过后长尾消失：该阈值给出干扰窗口宽度的量级；
- 部分 GPU 已偏移仍保持慢边：支持有限窗口，不支持精确对齐假设；
- offset 很大仍不改变慢边：单 GPU 本地 queue cadence 比跨 GPU phase 更重要。

建议结果目录：

```text
doc/results/gpu-contention/queue-phase-diagnostic/q4-source-offset/
doc/traces/gpu-contention/queue-phase-diagnostic/q4-source-offset/
```

#### Q4 已执行结果（2026-08-26）

完整矩阵见 [`Q4 原始目录`](../results/gpu-contention/queue-phase-diagnostic/q4-source-offset/20260826T113225Z-267432/)，
其中包含每个 case 的 command、JSON 和后台日志。六份 300-repeat D2H trace 及命令见
[`Q4 trace 目录`](../traces/gpu-contention/queue-phase-diagnostic/q4-source-offset/20260826T113959Z/)，
逐条分析见 [`Q4 trace analysis`](../results/gpu-contention/queue-phase-diagnostic/q4-source-offset/trace-analysis/20260826T113959Z/)。

| sourceOffsetsUs (GPU0,1,2,3) | 20-repeat 无背景 / D2H | 20-repeat 下降 | 300-repeat 无背景 / D2H | 300-repeat 下降 | 第一条 measured P2P start span (ms) | >8 ms P2P / 慢波 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,0,0,0 | 178.678 / 127.865 | 28.44% | 188.862 / 160.372 | 15.09% | 0.599 | 260 / 68 |
| 0,250,500,750 | 179.719 / 127.833 | 28.87% | 189.896 / 165.790 | 12.69% | 0.828 | 250 / 66 |
| 0,500,1000,1500 | 179.739 / 127.889 | 28.85% | 187.674 / 158.147 | 15.73% | 1.483 | 257 / 70 |
| 0,1000,2000,3000 | 179.807 / 127.918 | 28.86% | 186.371 / 152.544 | 18.15% | 2.784 | 260 / 78 |
| 0,2000,4000,6000 | 179.780 / 127.843 | 28.89% | 185.105 / 149.044 | 19.48% | 5.653 | 338 / 112 |
| 0,4000,8000,12000 | 179.820 / 127.764 | 28.95% | 181.426 / 137.207 | 24.37% | 11.925 | 441 / 165 |

Q4 的 delay kernel 实际持续时间分别约为 0.251/0.502/0.751、0.501/1.001/1.501、
1.002/2.001/3.001、2.001/4.001/6.001 和 4.001/8.001/12.001 ms（GPU1/2/3），
说明设备侧延迟确实执行。第一条 P2P 的 absolute start 还受到 warmup 完成顺序、host
enqueue 顺序以及 GPU0 无 delay 的影响，所以不能把它直接当作目标 offset 向量；这里
用 delay kernel duration 验证注入，用 P2P start span 验证实际启动错峰。

Q4 没有发现一个能让长尾消失的“一次性启动偏移阈值”。20-repeat 几乎不受 offset
幅度影响，仍保持约 128 GB/s 的 D2H 结果；300-repeat 则随初始 phase 改变恢复过程和
最终均值，但并未进入 per-edge 的无长尾状态。所有 trace 的慢点仍集中于第二 queue
position，且 2–12 ms 的一次性错峰反而增加慢波/慢样本。因此当前证据进一步支持
source 内 FIFO/cadence 与 D2H 作用下的内部 queue/engine 状态；不能声称具体 CE、
L2/HBM 或 NVLink scheduler 已被定位。Q4 只改变 measured batch 的初始 phase，未在
每个 round 重复施加 gate，所以后续若要测 steady-state phase window，必须另设计独立
的 per-round device-side 同步实验。

### 8.6 Q5：同源 buffer 复用负对照（Q1-Q4/Q2b-Q2f 后按需执行）

当前每个 source 的三条 edge 读取同一个 255 MiB source buffer。`per-edge` 并发可能
改变瞬时 L2/HBM read 请求合并或复用。若 Q1/Q2 不能完全解释长尾，增加：

```text
--sourceBufferMode=shared|per-edge
```

`per-edge` 模式为每条 outgoing edge 分配独立 source buffer，并使用不同初始数据，
避免三条并发 copy 读取完全相同地址。若 shared buffer 下无长尾、per-edge buffer 下
长尾恢复，则数据复用/请求合并参与；若两者一致，则该候选解释减弱。

### 8.7 实施和验收顺序

交给执行模型时按以下顺序实施，每一步单独验证，不能同时加入多个变量：

1. 先写 SQLite 只读分析脚本，复现现有三份 trace 的统计：262/265/0 个大于 8 ms
   的 P2P，以及慢边约 92% 位于第二队列位置；
2. [已完成] 实现 Q1 `source-chain`，完成 smoke test、3 次筛选和关键配置 trace；
3. [已完成] 实现 Q2 stream count，完成 3 次筛选和关键配置 trace；
4. [已完成] 实现 Q2b 三种 two-stream assignment，完成 36 个矩阵 case 和三份关键 trace；
5. [已完成] 实现 Q2c，保持 D2D=255M、background D2H=16K，完成 36 个矩阵 case；
6. [已完成] 实现 Q2d，保持 D2D=255M、background D2H=64K，完成 36 个矩阵 case；
7. [已完成] 实现 Q2e，保持 D2D=255M、background D2H=16M，完成 36 个矩阵 case；
8. [已完成] 实现 Q2f，保持 D2D=255M、background D2H=8M，完成 36 个矩阵 case；
9. [已完成] 实现 Q2g，保持 D2D=255M、background D2H=4M，完成 36 个矩阵 case；
10. [已完成] 实现 Q3 全部 6 种 permutation，完成 3 次筛选和六份关键 trace；
11. [已完成] 实现 Q4 source offset，完成 6 个 offset 向量的 3 次筛选和 6 份 trace，
   并用 delay kernel duration / 第一条 P2P start span 校验实际设备时间线；
12. 仅在 Q1-Q4/Q2b-Q2g 留下数据复用歧义时执行 Q5；
13. 根据决策表更新第 6.3、7、8 章，不提前写入 CE/NVLink 具体根因。

每个新增 CLI 参数都需要：

- `--help` 文本；
- 参数合法性检查；
- JSON 原样记录；
- CLI 单元测试；
- 2-repeat 四卡 smoke test；
- 原默认参数回归测试。

基于 Q1-Q4 及 Q2b-Q2f 当前证据，推荐结论限定为：

> D2H 与单 source 串行队列形成 queue-phase coupling；head-of-line blocking 和有限
> eligible work 是当前长尾的重要条件。现有数据不要求各 GPU copy 完全对齐，也
> 尚不能区分 FIFO 依赖、stream/engine mapping 和具体 NVLink path 调度的相对贡献。

### 8.8 其余既有后续工作

完成 Q1-Q4、Q2b、Q2c、Q2d、Q2e、Q2f 和 Q2g 后再继续：

1. 让 background JSON 输出每个 GPU 的 bytes/GB/s；D1 的 D2D per-source 已经足够
   显示局部性，但还需要精确的背景压力值；
2. 对 D1 的 H2D 和 D2H 各做至少 5-7 次完整重复，并加入所有六个两卡组合/不同
   host CPU affinity，排除 host placement 影响；
3. 完成 B 阶段全部 GPU 子集和 256M/512M 对照，确认 255 MiB 是否位于 host copy
   steady region；
4. repeat protocol 固定后，再进入 E 阶段的 HBM-read/write kernel、L2-resident
   kernel 和 local D2D CE 对照；压力 kernel 用 Nsight Compute 验证其实际流量性质。

## 9. 原始结果索引

- [A3 baseline stability](../results/gpu-contention/baseline-stability/)
- [repeat sweep and probe](../results/gpu-contention/timing-repeat-sweep/)
- [20-repeat main effect](../results/gpu-contention/main-effect/repeats-20/)
- [500-repeat main effect](../results/gpu-contention/main-effect/)
- [per-GPU injection](../results/gpu-contention/per-gpu-injection/)
- [host-copy first pass](../results/gpu-contention/host-copy-matrix/first-pass/)
- [telemetry](../results/gpu-contention/telemetry/allpairs-d2h-all/)
- [Nsight Systems traces](../traces/gpu-contention/)
- [phase-lock diagnostic matrix](../results/gpu-contention/phase-lock-diagnostic/20260826T094355Z-197181/)
- [phase-lock trace commands and raw outputs](../results/gpu-contention/phase-lock-diagnostic/traces/20260826T094945Z/)
- [Q1 source-chain matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T102517Z-219542/)
- [Q1 source-chain traces](../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/)
- [Q2 stream-count matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/q2-stream-count/20260826T104819Z-233894/)
- [Q2 stream-count traces](../traces/gpu-contention/queue-phase-diagnostic/q2-stream-count/20260826T105130Z/)
- [Q2b two-stream assignment matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154321Z-303566/)
- [Q2b two-stream assignment traces](../traces/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154647Z/)
- [Q2b two-stream assignment trace analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/trace-analysis/20260826T154647Z/)
- [Q2c low-intensity D2H matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T162155Z-326564/)
- [Q2d 64K-intensity D2H matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T162713Z-330735/)
- [Q2e 16M-intensity D2H matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T163221Z-334532/)
- [Q2f 8M-intensity D2H matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T163827Z-338551/)
- [Q2g 4M-intensity D2H matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T164908Z-344216/)
- [Q3 edge-permutation matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/20260826T110853Z-249443/)
- [Q3 edge-permutation traces](../traces/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/20260826T111616Z/)
- [Q4 source-offset matrix and analysis](../results/gpu-contention/queue-phase-diagnostic/q4-source-offset/20260826T113225Z-267432/)
- [Q4 source-offset traces](../traces/gpu-contention/queue-phase-diagnostic/q4-source-offset/20260826T113959Z/)
- [Q4 source-offset trace analysis](../results/gpu-contention/queue-phase-diagnostic/q4-source-offset/trace-analysis/20260826T113959Z/)
