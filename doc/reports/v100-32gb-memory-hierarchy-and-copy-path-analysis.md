# Tesla V100 32GB 显存层次与 CUDA 拷贝路径理论分析

本文把 NVIDIA 官方规格、当前机器拓扑和仓库中的一次四卡实测放在同一个
模型中，用于回答两个问题：

1. 当前 `allpairs` 的每张 V100 是否已经超过 HBM2 理论带宽；
2. `cudaMemcpyPeerAsync`、D2H 和 H2D 分别经过哪些 GPU、Copy Engine（CE）、
   NVLink、PCIe 和主机内存路径。

本文的性能数值来自一次默认配置验证，不是重复实验后的统计结论。路径中公开资料
没有明确规定的 L2/CE 内部细节均标记为“工作模型”，后续需要用 Nsight Systems、
CUDA device properties 和硬件计数器验证。

## 1. 先给结论

- NVIDIA 官方给出的 Tesla V100 32GB HBM2 峰值显存带宽为 **900 GB/s**；V100
  的 GPU 规格为 80 个 SM、4096-bit HBM2 接口和 **6144 KB L2 cache**。
- 当前四卡 allpairs、255 MiB、20 次重复的结果中，按“每次 D2D payload 至少需要
  一次源 HBM 读和一次目的 HBM 写”的基础模型计算，每卡本地 HBM 流量约为：无背景
  **89.9 GB/s（900 GB/s 的 10.0%）**，D2H 背景 **66.9 GB/s（7.4%）**，
  H2D 背景 **92.0 GB/s（10.2%）**。因此，**不能用“allpairs 超过 900 GB/s”
  解释 D2H 场景的下降**。
- 同一个 allpairs 测试的 D2D aggregate 从 179.756 GB/s 降到 127.796 GB/s，
  但 D2D 加背景的估算 HBM 流量仍远低于峰值。这只排除了“总 HBM 带宽已经被用满”
  这一种简单解释，不能排除 HBM controller 分区、片上 fabric、CE 调度或 NVLink
  资源之间的局部仲裁。
- 纯 `cudaMemcpy*Async` 没有用户 CUDA kernel，主要工作应由 GPU Copy Engine
  完成，不能把它当作一个有 CUDA thread block 的 SM occupancy 实验。SM、CE、
  NVLink/PCIe 的真实活动需要从时间线和计数器确认。
- 当前机器每个 GPU 对其他 GPU 都是 `NV2`，因此当前 D2D 的工作路径是“源 GPU
  HBM2 → 源侧 CE/内存路径 → NVLink NV2 → 目的侧 CE/内存路径 → 目的 GPU
  HBM2”；D2H/H2D 则使用当前机器的 PCIe Gen3 x16 主机链路和 pinned host memory。

## 2. 资料来源与当前实验边界

### 2.1 官方资料

- [NVIDIA Tesla V100 Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf?nvid=nv-int-gc27-14212)：
  V100 32GB HBM2 容量、900 GB/s 显存带宽，以及 PCIe/NVLink 互联规格。
- [NVIDIA Volta Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/volta-architecture-whitepaper.pdf)：
  V100 的 SM、HBM2 总线、L2、每 SM 的 L1/shared-memory 组合容量和 Volta Copy
  Engine 描述。
- [CUDA C Programming Guide 12.6：Multi-GPU Programming](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)：
  P2P access、`cudaMemcpyPeer*`、NVLink/PCIe 和避免 host staging 的语义。
- [CUDA Runtime API 12.6.2](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_Runtime_API.pdf)：
  `cudaMemcpyPeerAsync`、`cudaMemcpyAsync`、`cudaDeviceProp::l2CacheSize` 和
  `asyncEngineCount` 的 API 定义。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)：
  CUDA memory operation、kernel 和 Copy Engine 时间线的采集与显示方式。

官方资料给出了容量、峰值链路带宽和 API 语义，但没有给出一个适用于所有驱动、
传输大小和 CE 配置的“单次 memcpy 的 L2 带宽”、CE 内部仲裁表或固定 CE 数量。
因此，本文不会把网上针对某一驱动的 `asyncEngineCount` 经验值当作 V100 的官方
硬件规格。

### 2.2 当前机器和结果

环境快照见 [`environment.txt`](../results/4gpu/default-4gpu-validation/environment.txt)，
关键条件如下：

| 项目 | 当前值 |
| --- | --- |
| GPU | 4 × Tesla V100-SXM2-32GB |
| 驱动 / CUDA toolkit | 580.178.04 / 12.6.85 |
| GPU 拓扑 | GPU 两两之间均为 `NV2` |
| NVLink 状态 | 每张卡 6 条 link，`nvidia-smi nvlink -s` 显示约 25.781 GB/s/link |
| 主机拓扑 | 四张卡均在 NUMA node 0；当前 `nvidia-smi -q` 为 PCIe Gen3 x16 |
| D2D payload | 255 MiB = 267,386,880 bytes |
| D2D warmup / measured repeats | 10 / 20 |
| allpairs copy 数 | 4 × 3 = 12 条有向边 |

结果汇总见 [`summary.csv`](../results/4gpu/default-4gpu-validation/summary.csv)，
本分析使用的三份原始 JSON 为：

- [allpairs 无背景](../results/4gpu/default-4gpu-validation/allpairs_none/d2d.json)
- [allpairs + D2H](../results/4gpu/default-4gpu-validation/allpairs_d2h/d2d.json)
- [allpairs + H2D](../results/4gpu/default-4gpu-validation/allpairs_h2d/d2d.json)

## 3. V100 32GB 的内存层次和互联规格

### 3.1 官方参数

| 层次/部件 | Tesla V100 32GB 相关参数 | 对本实验的含义 |
| --- | ---: | --- |
| HBM2 容量 | 32 GB | 每张卡的源/目的 buffer 都位于本卡 device memory |
| HBM2 峰值带宽 | **900 GB/s** | 本地 HBM2 的理论峰值；不是 NVLink 或 D2D wire rate |
| HBM2 memory interface | **4096 bit** | 影响本地 HBM channel 总带宽和读写仲裁 |
| SM 数量 | **80 SM** | 解释 GPU 计算资源规模；不是 memcpy 的并行线程数 |
| L2 cache | **6144 KB** | 所有 SM/片外内存路径共享的较低级 cache 容量 |
| 每 SM L1/shared 组合池 | **128 KB/SM** | 是 L1 data cache 与 shared memory 的组合容量，不等于 128 KB 独立 L1 |
| 每 SM 最大 shared memory | **96 KB** | 由 shared-memory 配置决定，剩余组合容量可作为 L1 |
| Volta NVLink | 6 links，25 GB/s/方向/link | 单卡单方向合计约 150 GB/s，双向合计约 300 GB/s |

这里的 `900 GB/s`、`6144 KB`、80 SM、4096-bit HBM2 和 NVLink 参数均来自上述
NVIDIA 官方 datasheet/whitepaper。`cudaDeviceProp::l2CacheSize` 可在运行时读取
实际设备报告的 L2 字节数；因此后续环境采集应同时记录官方型号和 runtime property。

### 3.2 “显存带宽”与“缓存容量”不是同一个上限

HBM2 的 900 GB/s 是本地显存介质的峰值带宽。它不表示：

- 一次 P2P memcpy 可以得到 900 GB/s 的 NVLink 传输速率；
- L2 具有 900 GB/s 或某个公开固定的带宽；
- CE、HBM controller、NVLink 和 PCIe 之间不存在共享仲裁；
- 所有访问模式都能达到 900 GB/s。

L2 的 6144 KB 是容量，不是带宽。当前每个 payload 为 255 MiB，单个 payload
约为 6 MiB L2 的 **42.5 倍**；allpairs 中每卡有 1 个 source buffer 和 3 个
destination buffer，共约 1020 MiB，约为 L2 的 **170 倍**。这说明整个 payload
不可能驻留在 L2，但不能据此断言每一个字节都绕过 L2，或精确推导 HBM transaction
数。CUDA 公共 API 没有为普通 `cudaMemcpy*` 提供固定的 L2 cache policy；本文的
“源读 + 目的写”是用于带宽上界估算的工作模型。

## 4. allpairs 的字节计数和每卡 HBM 负载

### 4.1 当前程序的并发语义

当前 `d2d_multi_peer_bw` 的 allpairs 不是 12 条完全独立的 stream：

- 每张 GPU 一个 source stream；
- 每张 GPU 的 3 个出方向 memcpy 在该 source stream 上串行排队；
- 4 个 source stream 可并行执行；
- 12 条有向边各自使用独立 destination buffer，避免多个 source 写同一块目的内存。

所以 aggregate 是四个 source stream 的总 payload 除以最长 source elapsed，而不是
把 12 条边的时间相加。

### 4.2 理论计数

令：

```text
B = 255 MiB = 267,386,880 bytes
N = 4 GPUs
E = N × (N - 1) = 12 directed edges
R = 20 measured repeats
```

则：

```text
allpairs logical D2D bytes = E × R × B
                           = 64,172,851,200 bytes
                           = 64.1728512 GB

每张 GPU 的 source-side logical bytes = 3 × R × B
                                     = 16,043,212,800 bytes

每张 GPU 的 destination-side logical bytes = 3 × R × B

每张 GPU 的本地 HBM traffic 估算
  = source HBM read + destination HBM write
  = 2 × 3 × R × B
  = 32,086,425,600 bytes
```

`aggregateGBps` 只把每条 D2D logical payload 计数一次；为了估算本地 HBM，必须再
把 source read 和 destination write 都算进去。因此在对称四卡 allpairs 中：

```text
estimated HBM GB/s per GPU ≈ D2D aggregate GB/s ÷ 4 × 2
```

这是容量/字节流量模型，不是硬件计数器的精确读数。实际值可能受 cache policy、
写分配、ECC、协议包头和计数器口径影响。

### 4.3 实测结果换算

| 场景 | D2D aggregate | D2D logical / GPU | D2D HBM 估算 / GPU | 背景 aggregate | 背景 / GPU | 合计 HBM 估算 / GPU | 占 900 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| allpairs，无背景 | 179.756 | 44.939 | 89.878 | — | — | **89.878** | **9.99%** |
| allpairs + D2H | 127.796 | 31.949 | 63.898 | 11.844 | 2.961 | **66.859** | **7.43%** |
| allpairs + H2D | 178.159 | 44.540 | 89.080 | 11.633 | 2.908 | **91.988** | **10.22%** |

表中单位均为 GB/s，`D2D HBM 估算 / GPU` 已包含一次本地读和一次本地写；D2H
背景增加的是本地 HBM read，H2D 背景增加的是本地 HBM write。背景程序的累计
时间窗口与 D2D CUDA event 窗口并非严格相同，因此“合计 HBM 估算”只用于量级
判断，不能当作精确的同步带宽测量。

### 4.4 是否超过 HBM 理论上限？

**没有。** 即使采用读写都计入的模型，当前三种 allpairs 场景每卡估算值只有
约 66.9–92.0 GB/s，明显低于官方 900 GB/s 峰值。D2H 场景的 D2D aggregate
相对无背景下降约 28.9%，不能解释为“总 HBM 带宽从 900 GB/s 被打满”。

更准确的结论是：

- 已经可以排除“按总字节数计算的 HBM 带宽饱和”作为首要解释；
- 不能排除 HBM controller 的局部 channel/partition 仲裁、读写方向不对称、
  CE 到内存系统的访问粒度，或其他片上共享资源；
- 900 GB/s 是峰值规格而不是该 memcpy 模式的保证吞吐，低于峰值仍可能有内部
  资源排队。

## 5. 各类拷贝的完整传输流程

### 5.1 D2D：P2P + 当前 NV2 NVLink 工作路径

当前程序使用 `cudaMemcpyPeerAsync`，并对使用到的 GPU 对启用 peer access。CUDA
官方文档说明，GPU peer copy 会根据拓扑使用 NVLink 或 PCIe；启用 peer access 时
不需要先把 payload 拷到 host 再转发。结合本机 `nvidia-smi topo -m` 的 `NV2`，
当前实验的预期路径为：

```text
GPU A HBM2
   │  source read
   ▼
GPU A memory subsystem / optional L2 path
   │
   ▼
P2P Copy Engine / DMA path
   │
   ▼
GPU A NVLink endpoint
   │  NV2: the bonded two-link GPU-to-GPU path
   ▼
GPU B NVLink endpoint / memory fabric
   │
   ▼
GPU B HBM2
```

需要注意：

1. “source read + destination write”是本地 HBM 字节模型；公开 CUDA 文档没有承诺
   某个版本中 CE/DMA endpoint 一定位于源侧、目的侧，还是由某个内部 copy path
   统一处理，也没有公开 L2 是否对该 memcpy 完全 bypass 的固定规则。
2. NV2 表示该 GPU 对之间有两条 bonded NVLink。allpairs 每张卡有三个 peer，
   理论上会使用本卡六条 link；ring 每张卡只有一个 peer，主要施压一个 NV2 link
   bundle。
3. 如果 peer access 不可用、被禁用，或拓扑没有 NVLink，不能继续使用上面的 NVLink
   假设；应按 CUDA/拓扑选择的 PCIe 或 host-staging 路径重新分析。
4. `cudaDeviceCanAccessPeer` 和 `cudaMemcpyPeerAsync` 的成功只说明 API/访问关系
   可用，不单独构成“实际所有字节都走 NVLink”的证明。应同时观察 NVLink counter
   和 Nsight Systems memory operation。

### 5.2 D2H：GPU HBM2 到 pinned host memory

当前 background 程序用 `cudaMallocHost` 分配 page-locked/pinned host buffer，并在
GPU stream 上执行 `cudaMemcpyAsync`。当前 x86 主机的工作路径为：

```text
GPU HBM2
   │  device read
   ▼
GPU Copy Engine
   │
   ▼
GPU PCIe endpoint
   │  PCIe Gen3 x16, upstream
   ▼
PCIe root complex / host bridge
   │
   ▼
CPU memory controller
   │
   ▼
pinned host DRAM buffer
```

CPU 线程负责提交 API、处理同步和循环控制，payload 本身不需要由 CPU 逐字节复制。
`pinned` 的作用是让 GPU 能够对稳定的物理 host pages 做异步 DMA；如果传入 pageable
memory，runtime 可能需要内部 staging，并且异步/同步语义会改变。

### 5.3 H2D：pinned host memory 到 GPU HBM2

H2D 是上述路径的反向：

```text
pinned host DRAM buffer
   │
   ▼
CPU memory controller
   │
   ▼
PCIe root complex / host bridge
   │  PCIe Gen3 x16, downstream
   ▼
GPU PCIe endpoint
   │
   ▼
GPU Copy Engine
   │
   ▼
GPU memory subsystem / optional L2 path
   │
   ▼
GPU HBM2
```

因此 H2D 对本地 HBM 的主要模型是一次 write，D2H 的主要模型是一次 read。两者
都可能共享 PCIe、root complex、host memory controller、GPU CE 调度和 GPU 内部
内存路径；方向不同还可能触发不同的读写仲裁行为。

### 5.4 SM、CE、NVLink、PCIe 的角色

| 操作 | 用户 SM kernel | GPU CE/DMA | 本地 HBM | 设备间/主机链路 |
| --- | --- | --- | --- | --- |
| P2P D2D | 没有用户 kernel | 主要由 CE 完成 | 源 read + 目的 write | 当前拓扑预期为 NVLink NV2 |
| D2H | 没有用户 kernel | 主要由 CE 完成 | 源 read | 当前主机预期为 PCIe Gen3 x16 → DRAM |
| H2D | 没有用户 kernel | 主要由 CE 完成 | 目的 write | DRAM → PCIe Gen3 x16 |

Volta whitepaper 将 GPU Copy Engine 描述为在多个 GPU 之间或 GPU 与 CPU 之间
传输数据的 DMA-like 硬件。这个结论不等于“SM 永远完全没有活动”：驱动提交、
同步、页表/错误处理和系统中其他工作仍可能出现在时间线中，但不能用 Nsight
Compute 对一个 memcpy API 直接套用 CUDA kernel 的 occupancy 定义。

## 6. 链路负载的第二个 sanity check

### 6.1 NVLink

官方 V100 NVLink 规格是 6 × 25 GB/s/方向/link，即每卡约 150 GB/s 单方向、
300 GB/s 双向。当前 allpairs 的每卡 D2D logical 出方向速率约为
`aggregate / 4`：

| 场景 | 每卡 D2D 出方向 logical rate | 150 GB/s 单方向 NVLink 峰值占比 |
| --- | ---: | ---: |
| allpairs，无背景 | 44.939 GB/s | 29.96% |
| allpairs + D2H | 31.949 GB/s | 21.30% |
| allpairs + H2D | 44.540 GB/s | 29.69% |

这说明 allpairs 的 aggregate payload 也没有接近“每卡所有 NVLink 单向原始带宽”的
简单上限。不过 ring 的每卡出方向约为 `193.957 / 4 = 48.489 GB/s`，而单个
NV2 bundle 的单方向理论值约为 `2 × 25 = 50 GB/s`，已经约为 97%。这解释了为什么
ring 和 allpairs 不能只按 aggregate 数字直接比较：ring 主要压满一个 peer bundle，
allpairs 则让一张卡的多个 link bundle 同时参与调度。

### 6.2 PCIe/主机背景流量

PCIe Gen3 每 lane 的编码为 8 GT/s、128b/130b。由当前 x16 链路推导的单方向
payload 上限约为：

```text
8 GT/s × 16 lanes × 128/130 ÷ 8 ≈ 15.75 GB/s
```

这是根据当前链路代际和宽度推导的链路 payload 上限，不是 V100 HBM 的官方规格，
实际还会受到协议开销、root complex、host DRAM 和 NUMA 影响。allpairs 背景结果
折算到每卡约 2.961 GB/s（D2H）和 2.908 GB/s（H2D），约为该推导值的 18.8% 和
18.5%，也没有显示出“每卡 PCIe x16 已被粗粒度打满”。这同样不能排除共享 root
complex 或主机内存系统的局部仲裁。

## 7. 对当前现象的解释边界

用户提供的论坛现象是：四卡 allpairs 在 D2H 背景下明显下降，H2D 和 ring 变化较小，
两卡控制也不明显。当前仓库的一次默认运行复现了相同方向的信号：

| pattern | 背景 | D2D aggregate | 相对同 pattern 无背景 |
| --- | --- | ---: | ---: |
| ring | none | 193.957 GB/s | — |
| ring | D2H | 193.319 GB/s | -0.3% |
| ring | H2D | 193.163 GB/s | -0.4% |
| allpairs | none | 179.756 GB/s | — |
| allpairs | D2H | 127.796 GB/s | -28.9% |
| allpairs | H2D | 178.159 GB/s | -0.9% |

这个结果目前支持以下判断：

1. **不是简单的总 HBM 带宽超限。** 读写都算入后仍只有约 7–10% 的理论峰值。
2. **不是简单的单个 NV2 bundle 饱和。** allpairs 每卡出方向约 45 GB/s，低于
   6-link 单向总和；但多 link 的 fabric/CE 调度仍可能有共享资源。
3. **不是粗粒度 PCIe aggregate 饱和的充分证据。** 背景 aggregate 只有约 12 GB/s，
   且每卡 host traffic 约 3 GB/s；但 root complex、CPU memory controller 和 DMA
   队列仍可能出现局部竞争。
4. **D2H/H2D 的方向差异值得重点研究。** 两种背景 aggregate 量级接近，而 D2H
   对 allpairs 的影响明显更大，提示需要区分 GPU-local read/write arbitration、
   CE queue 竞争、NVLink/fabric 交互和主机路径，而不是只比较总字节数。

当前不能据此断言具体是 L2、HBM controller、CE 还是 NVLink scheduler。特别是
“L2 cache 只有 6 MB”只能说明 working set 远大于 L2，不能单独证明 L2 是瓶颈。

## 8. 下一步测量建议

### 8.1 先记录 runtime 的实际属性

为每张卡增加或单独运行一个 device-property probe，至少记录：

```text
name
major / minor
totalGlobalMem
memoryBusWidth
memoryClockRate
l2CacheSize
multiProcessorCount
asyncEngineCount
deviceOverlap
```

其中 `asyncEngineCount` 是 CUDA runtime 对异步引擎的设备属性，不应从“6 条 NVLink”
或 stream 数量推断；它也不等同于公开的 NVLink link 数。正式报告要把 runtime
返回值和 driver/toolkit 版本一起保存。

### 8.2 用时间线确认 CE 和实际链路

对以下最小集合采集 Nsight Systems：

```text
allpairs + none
allpairs + D2H
allpairs + H2D
ring + D2H
```

重点观察 CUDA memory operation、D2D/D2H/H2D 的开始结束时间、CE channel/track、
stream wait，以及是否出现 host staging。并行采集 `nvidia-smi nvlink` 的 data
counter；如果 D2D 真的走 NVLink，D2D 窗口内应看到相应 NVLink RX/TX 增长，而不应
出现与 payload 相当的 PCIe 变化。

### 8.3 让背景流量成为可控变量

在保持 255 MiB D2D 不变的情况下，逐步增加：

- 背景 GPU 集合：`{0}`、`{1}`、`{0,1}`、`{0,1,2,3}`；
- 背景方向：D2H、H2D；
- payload：4 MiB、64 MiB、255 MiB、1 GiB；
- D2D pattern：ring、allpairs；
- host NUMA/CPU affinity。

同时保留每个 source GPU 的 elapsed。若只在施压 GPU 的 source stream 或同一 GPU
的 destination activity 上变慢，才有较强的 GPU-local 竞争证据。

### 8.4 统计规则

当前 `default-4gpu-validation` 只有一次默认配置运行，适合做理论量级 sanity check，
不适合做最终因果结论。正式主实验至少进行 5–7 次独立运行，报告 median、离散度、
每卡 source elapsed，并把 profiler 运行与性能运行分开。SM occupancy 只对实际
执行的 CUDA kernel 报告；对纯 memcpy，应报告 CE/memory-operation timeline 和
NVLink/PCIe counter。

## 9. 参考结果和官方链接

- [当前四卡验证报告](four-gpu-cuda-copy-validation.md)
- [四卡汇总 CSV](../results/4gpu/default-4gpu-validation/summary.csv)
- [四卡环境快照](../results/4gpu/default-4gpu-validation/environment.txt)
- [NVIDIA Developer Forum：D2H background traffic interferes with 4-GPU allpairs `cudaMemcpyPeerAsync` bandwidth](https://forums.developer.nvidia.com/t/d2h-background-traffic-interferes-with-4-gpu-allpairs-cudamemcpypeerasync-bandwidth/372474)
