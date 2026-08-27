# 四卡 D2D 与 D2H 的 GPU 内部资源竞争实验方案

## 1. 目标与结论边界

本方案用于检验以下工作假设：四卡 allpairs `cudaMemcpyPeerAsync` 与并发
D2H `cudaMemcpyAsync` 在 GPU 内部共享了某个有限资源，因此 D2D 带宽下降。
候选资源包括：

1. GPU 本地 HBM 读带宽或显存分区；
2. L2 cache、L2 slice 或 L2 到片外接口；
3. Copy Engine（CE）或 CE 到内存/NVLink/PCIe 的内部仲裁路径；
4. 更宽泛的 GPU-local crossbar/fabric 仲裁资源。

实验目标不是找出 NVIDIA 未公开的具体仲裁单元，也不是解决性能下降，而是形成
一条可重复、可证伪的证据链，使最终结论至少达到下列强度之一：

- **否定 GPU-local 竞争猜想**：退化能由 PCIe/CPU/NUMA、计时错误、时钟变化或
  profiler 扰动解释；
- **支持 GPU-local 资源竞争**：退化随施压 GPU、传输方向和 GPU-local 负载变化，
  且不能由 NVLink/PCIe 链路饱和单独解释；
- **更像 HBM/显存路径竞争**：退化主要随设备内存读流量和 DRAM 活跃度变化；
- **更像 L2/片上路径竞争**：低 DRAM 流量的 L2 压力也能稳定复现退化，且出现
  明显的工作集阈值；
- **只能归因于 CE/内部仲裁**：证明了源 GPU 本地性和方向性，但 HBM/L2
  消融不能把两者可靠分开。

纯 `cudaMemcpy*Async` 是 Copy Engine 工作负载。公开工具通常不能把一次 CE
传输的所有 L2、HBM 和内部仲裁计数器像 CUDA kernel 那样归因到单次操作。
因此，本方案不会声称仅凭一张 profiler 时间线“严格证明 L2 竞争”。最终证据来自
控制变量、方向性、局部性、压力响应曲线和硬件计数器的共同指向。

参考实验：[NVIDIA Developer Forums：D2H background traffic interferes with
4-GPU allpairs cudaMemcpyPeerAsync bandwidth](https://forums.developer.nvidia.com/t/d2h-background-traffic-interferes-with-4-gpu-allpairs-cudamemcpypeerasync-bandwidth/372474)。

## 2. 当前仓库已经完成的部分

本方案直接建立在当前实现之上，不重新设计复现程序。

- `src/cuda_copy/d2d_multi_peer_bw.cu`
  - 支持四卡 ring 和 allpairs；
  - ring 为 4 条有向边，allpairs 为 12 条有向边；
  - 每个源 GPU 使用一条 stream，allpairs 中三个出方向依次入队；
  - 每条有向边使用独立目标 buffer；
  - JSON 包含每个源 GPU 的 elapsed time 和带宽，以及 aggregate 带宽。
- `src/cuda_copy/host_copy_background.cu`
  - 每个 GPU 使用 pinned host buffer、device buffer 和独立 stream；
  - 支持 D2H/H2D；
  - 循环执行 async memcpy 和 stream synchronize；
  - 支持 ready/stop marker，并输出累计带宽。
- `scripts/run_4gpu_copy_matrix.sh`
  - 已覆盖 `ring/allpairs × none/d2h/h2d` 六种场景；
  - 保存环境、命令、原始 log、JSON 和汇总 CSV。
- `doc/reports/four-gpu-cuda-copy-validation.md`
  - 已观察到 allpairs 从约 180 GB/s 降至约 128 GB/s；
  - ring 在 D2H/H2D 背景下基本不变；
  - 当前结果是单次验证，不能单独作为因果结论。

最重要的现有能力是 **per-source 计时**。后续只对 GPU k 施加 D2H 时，可以判断
是否主要是 source=k 的 D2D stream 变慢。这是证明 GPU-local 竞争的核心证据。

## 2A. 本轮执行状态（2026-08-26）

详细数据见 [`gpu-internal-copy-contention-results.md`](gpu-internal-copy-contention-results.md)。

- [x] A3：allpairs 无背景的 10 次长批次基线已完成；repeats=200 的 range 为 0.76%。
- [partial] A2：repeat-sweep 发现 20→500 次的 aggregate 从 177.986 增至 190.738 GB/s；
  新增 `--chunkRepeats=10` 后确认前约 90–100 次 round 处于较低阶段，随后进入较高
  throughput 阶段；`chunkRepeats=1/5/20` 对照也在约第 100 次 round 观察到同一转折。
  100 ms clock/pstate 采样已确认 D2H 场景在 P0/1530 MHz 下仍保留早期低阶段；
  per-iteration synchronize 对照已确认逐次排空会固定在约 179/130/127 GB/s 的低阶段，
  但会改变 source-local 计时语义，
  因此 20 和 500 继续作为两个独立协议，不能混合统计。
- [partial] B：已完成 4K/1M/16M/64M/255M/512M × 1/2/3/4 卡前缀集合 × D2H/H2D
  的第一批矩阵；全部 48 个点成功，但还不是全部 GPU 子集、全部尺寸和重复统计。
- [x] C：20-repeat 和 500-repeat 各完成 7 次六场景主效应矩阵；allpairs+D2H 分别为
  -28.82% 和 -10.37%，ring/H2D 负对照成立。
- [x] D1：20-repeat 的 D2H/H2D GPU0/1/2/3/all 注入各完成 7 轮；另完成 500-repeat
  D2H 的三轮注入。20-repeat 下 D2H source-local 映射稳定，H2D 单卡仅有一个孤立
  异常点，全卡约 -0.9%。
- [x] P0–P4：Nsight Systems trace 和 CLI memory-operation 摘要已采集；P2 另用独立
  wrapper 修正为真正的 GPU0-only background。当前 V100 的 GPM 和 NVLink data counter
  返回 `-`/`N/A`，未把不支持当作 0。
- [x] repeat-regime diagnostic：`repeats=20/500`、none/GPU0-only D2H/all-GPU D2H
  各完成 3 次分块采样；`chunkRepeats=10`，并完成一条无 profiler dmon 对照。
- [x] chunk-granularity boundary sweep：`repeats=500` 下完成
  `chunkRepeats=1/5/20`、none/GPU0-only D2H/all-GPU D2H 各 2 次；三种粒度均观察到
  约第 100 次 round 的阶段转折，但 event 插入会改变绝对带宽。
- [x] high-resolution clock/pstate probe：上述 3 个场景各完成 1 次 100 ms
  `nvidia-smi` 采样；D2H 场景全程 1530 MHz，不能由 graphics clock 单独解释早期低阶段。
- [x] per-iteration synchronize control：实现 `--syncEachIteration=0|1`，完成
  async/sync × repeats=20/500 × none/GPU0-only D2H/all-GPU D2H × 3 次配对矩阵；
  逐次同步固定低阶段，但不再保持异步模式的 source-local 计时语义。

本轮阶段性结论是：**存在稳定的 GPU-local、D2H-specific source-side 竞争信号，且
不能由总 HBM 900 GB/s 饱和解释；分块诊断已解释 20/500 的阶段性差异，但触发该
阶段转折的 CE/clock/队列机制和 HBM/L2/CE 具体归因尚未完成；逐次同步实验表明
队列深度是重要控制变量，但其 host-paced 语义不能替代异步 source-local 主协议。**

## 3. 总体策略

实验分为六个阶段，严格按顺序执行：

1. 冻结环境并验证测量方法；
2. 建立 D2H/H2D 自身的尺寸—并发带宽矩阵；
3. 重复论坛主实验，确认效应稳定；
4. 用单 GPU 背景和 D2D 拓扑消融定位竞争端点；
5. 用压力强度、工作集和受控 kernel 区分 HBM、L2 与 CE/fabric；
6. 采集必要的时序与计数器证据，统一判读。

每进入下一阶段都需要上一阶段满足验收条件。不要一开始就进行大规模 profiling：
profiler 会增加开销，而且无法弥补缺失的对照实验。

## 4. 通用实验规则

### 4.1 固定项

每个可比较组必须保持以下条件相同：

- GPU 顺序、CUDA 可见设备和 P2P 拓扑；
- GPU/显存时钟策略、电源限制和持久化状态；
- D2D payload、warmup、repeats 和同步方式；
- D2H/H2D buffer 是否 pinned、stream 数和同步频率；
- CPU affinity、NUMA 结点和 host buffer 首次触页位置；
- 机器上的其他 GPU/PCIe/NVLink 负载；
- 构建产物和 Git revision。

每组开始和结束记录温度、P-state、SM clock、memory clock 和 power。若比较期间
发生明显降频，该组作废并重跑，不能将降频造成的退化归因于资源竞争。

### 4.2 重复、随机化与统计

- 每个场景至少 7 次独立运行；首次 dry run 不计入统计；
- 将同一实验块中的场景随机排序，或使用交错顺序 `baseline → treatment →
  baseline → treatment`，避免温度和时间漂移；
- 每次正式测量至少持续 3 秒，优先通过增加 repeats 达到，而不是改变 payload；
- 报告 median、P25、P75、min、max 和相对基线变化；
- 主效应同时报告 aggregate D2D 和四个 per-source D2D；
- 稳定效应的建议判据：7 次中至少 6 次同方向，median 变化大于基线运行间
  MAD 的 3 倍；
- profiler 运行与无 profiler 性能运行分开。结论中的带宽数值来自无 profiler
  运行，profile 只用于解释时序和计数器。

### 4.3 吞吐定义

- D2D aggregate：12 条有向边的总字节数除以四个源 stream 中最长 elapsed；
- per-source：该源的三个有向拷贝总字节数除以该源 stream elapsed；
- D2H/H2D aggregate：所有参与 GPU 完成的 host-transfer 字节数除以同一墙钟窗口；
- 所有结果统一使用 GB/s（`1e9` bytes/s），同时保留精确 bytes；
- 不能把 12 条有向边的 aggregate 与单链路峰值直接比较。

## 5. 阶段 A：环境与测量正确性

### A1. 拓扑与路径确认

记录：

```bash
nvidia-smi -q
nvidia-smi topo -m
nvidia-smi nvlink -s
nvidia-smi nvlink -p
nvcc --version
```

确认所有测试 GPU 对均走 NVLink，并记录四张卡到 CPU/NUMA 的关系。不要仅因为
`cudaDeviceCanAccessPeer` 返回 true 就假设实际流量一定走 NVLink；在 D2D 基线时
还应观察 NVLink 计数器确实增长、PCIe 不出现与 D2D 字节量相当的增长。

### A2. 计时自检

对 ring 和 allpairs 分别检查：

1. warmup 不进入计时；
2. start/stop event 与被测 memcpy 位于同一 source stream；
3. aggregate 使用最大 source elapsed，而非各 stream 时间相加；
4. 增加 repeats 后 GB/s 稳定，而不是随 repeats 系统性增长；
5. 同步每轮与全部排队后再同步分别运行一次，记录二者差异，但论坛复现仍使用
   当前的 no-sync-each-iteration 语义。

### A3. 空闲与热稳定性

在无背景流量下连续运行 10 次 allpairs。若 max/min 差异大于 5%，先处理系统噪声、
温度或时钟变化，再进入下一阶段。

**阶段 A 验收：** D2D 确实走 NVLink；计时定义无误；基线变异足以支持识别约
29% 的已观察退化。

## 6. 阶段 B：D2H/H2D 带宽矩阵

这一阶段回答两个问题：host copy 在什么 payload 后达到稳态带宽，以及增加并发
GPU 数是否已经使主机 PCIe/内存子系统饱和。

### B1. Payload 扫描

建议使用严格的 2 的幂：

```text
4 KiB, 8 KiB, 16 KiB, 32 KiB, 64 KiB, 128 KiB, 256 KiB, 512 KiB,
1 MiB, 2 MiB, 4 MiB, 8 MiB, 16 MiB, 32 MiB, 64 MiB, 128 MiB,
256 MiB, 512 MiB, 1 GiB
```

255 MiB 作为论坛复现点额外保留，但不取代 256 MiB。若 1 GiB × 4 pinned host
buffer 对系统内存压力过大，可以在 512 MiB 截止，并在结果中记录原因。

### B2. GPU 子集矩阵

每个 payload、每个方向分别测量：

| 组别 | GPU 集合 | 目的 |
| --- | --- | --- |
| S0–S3 | `{0}`、`{1}`、`{2}`、`{3}` | 单卡能力及卡间差异 |
| P01…P23 | 所有六个两卡组合 | PCIe/NUMA 路径差异 |
| T012…T123 | 所有四个三卡组合 | 并发增长曲线 |
| Q0123 | 四卡 | 主实验背景能力 |

分别输出 per-GPU 和 aggregate 带宽。现有 background 程序只输出 aggregate 时，
后续实现应补充每 GPU counter；在补充前不能用 aggregate 推断某一张卡的 D2H
是否达到瓶颈。

### B3. 结果图

至少生成：

1. x=payload（log2），y=per-GPU GB/s，每个 GPU 一条线；
2. x=payload，y=aggregate GB/s，按 1/2/3/4 GPU 分组；
3. x=参与 GPU 数，y=aggregate 与 per-GPU GB/s，D2H/H2D 分图；
4. 四卡 payload × direction 的热力表。

若 aggregate 带宽随 GPU 数很早停止增长，只说明主机/PCIe 路径存在共享瓶颈，
不能解释为什么 H2D 不影响 allpairs。它会成为后续解释中的控制变量。

**阶段 B 验收：** 找到延迟主导区、带宽爬升区和稳态区；确认 255/256 MiB 位于
稳态区；得到每卡 D2H/H2D 能力和并发缩放曲线。

## 7. 阶段 C：稳定复现主效应

固定 payload=255 MiB，执行现有六场景矩阵：

```bash
./scripts/build_cuda_copy.sh
./scripts/run_4gpu_copy_matrix.sh \
  --size=255M --repeats=<使单场景至少持续3秒的值> \
  --outputRoot=doc/results/4gpu/main-effect/run-01
```

重复 7 次，并交错或随机化场景顺序。现有 runner 固定顺序，因此正式实验前建议让
runner 接受 scenario order/seed，或每次分别指定 `--patterns`、`--directions` 以人工
交错。

主比较为：

- allpairs+D2H 对 allpairs+none；
- allpairs+H2D 对 allpairs+none；
- ring+D2H 对 ring+none；
- ring+H2D 对 ring+none。

预期的必要边界条件是：allpairs+D2H 稳定下降，而 ring+D2H、allpairs+H2D
不出现同量级下降。如果所有模式都下降，应先检查 PCIe、时钟和系统负载；如果
D2H/H2D 都下降，则“D2H 特有的源端读竞争”假设不成立。

**阶段 C 验收：** allpairs+D2H 的效应量显著超过基线噪声，且方向/拓扑负对照成立。

## 8. 阶段 D：定位竞争发生在哪张 GPU、哪一端

### D1. 单 GPU D2H 注入（最高优先级）

保持四卡 allpairs 不变，背景依次放在：

```text
none, GPU0 only, GPU1 only, GPU2 only, GPU3 only, GPU0-3 all
```

每个场景同时做 D2H 和 H2D。分析 JSON 中四个 per-source elapsed，而不只看
aggregate。

对背景 GPU k，定义：

```text
local-source slowdown(k) = BW_source_k(treatment) / BW_source_k(baseline)
remote-source slowdown(k,j) = BW_source_j(treatment) / BW_source_j(baseline)
```

判读：

- 主要只有 source=k 变慢：强支持源 GPU 本地资源竞争；
- 主要是以 k 为 destination 的边变慢：支持目的端或入站 NVLink 路径竞争；
- 所有 source 同比例变慢：更像全局同步、共享 PCIe/CPU 或 aggregate 计时耦合；
- 单卡背景几乎无影响、四卡背景突然出现影响：存在总压力阈值或跨 GPU 共享资源；
- 效应随背景 GPU 数近似累加：更像每 GPU 局部瓶颈累积。

当前程序只给 per-source、不给 per-edge 时间。如果 D1 无法区分 source/destination，
再增加“一条 stream/event 对应一条 edge”的诊断模式。该模式只用于定位，不替代
原始论坛语义。

### D2. D2D 拓扑压力阶梯

在相同 D2H 背景下逐步增加每个 source 的 fan-out：

```text
1 edge/source（ring） → 2 edges/source → 3 edges/source（allpairs）
```

另外固定总 edge 数，比较：

- 边集中在少数 source；
- 边均匀分散到四个 source。

若退化由每个 source 的 fan-out 决定，而不是全局 edge 总数决定，强烈支持源端
GPU-local memory/CE/fabric 竞争。

### D3. D2H 占空比与强度响应

给背景循环增加可控间隔或 token-bucket，使每张卡获得约：

```text
0%, 10%, 25%, 50%, 75%, 100% 的单卡最大 D2H 带宽
```

绘制 D2D slowdown 对实际 D2H GB/s，而不是对“设置档位”。

- 平滑、单调下降：典型共享带宽/仲裁竞争；
- 到某一点突然下降：可能是 CE 调度或内部队列阈值；
- 与 D2H 带宽不相关，只与 memcpy 提交/同步频率相关：更像 CE/driver/API 调度，
  而非 HBM 带宽。

### D4. 传输次数与字节率解耦

保持总 D2H GB/s 尽量相近，比较“小 payload 高频提交”和“大 payload 低频提交”。

- 只随 GB/s 变化：支持数据通路带宽竞争；
- 随 operation rate 或同步次数变化：支持 CE 队列、driver 或同步开销；
- 两者都变化：需要二维回归，不能简单归因于 HBM。

**阶段 D 验收：** 得到 slowdown 的 GPU-local incidence map、fan-out 响应和 D2H
压力响应曲线。

## 9. 阶段 E：区分 HBM、L2 与 Copy Engine/fabric

### E1. 工作集尺寸扫描

D2D payload 和 D2H payload 不必始终相同。分别固定其中一个为 256 MiB，扫描另一个：

```text
256 KiB, 512 KiB, 1 MiB, 2 MiB, 4 MiB, 8 MiB, 16 MiB,
32 MiB, 64 MiB, 128 MiB, 256 MiB, 512 MiB
```

V100 的 L2 容量应从实际设备属性/官方规格记录，不在脚本中硬编码。扫描范围必须
跨越 L2 容量上下两侧。

需要注意：同一个 buffer 的循环 D2H 即使工作集小于 L2，也不能直接假定数据全部
命中 L2；Copy Engine 的 cache 行为和策略可能不同。工作集阈值只能作为线索，不能
单独证明 L2。

判读：

- 只有工作集远大于 L2 时才随字节率明显退化：更像 HBM/显存路径；
- 在 L2 容量附近有可重复拐点：支持 L2 参与；
- 对工作集不敏感，但对 memcpy 并发数/提交频率敏感：更像 CE/fabric 仲裁。

### E2. 受控 HBM 压力 kernel

增加一个仅用于消融的 streaming kernel：每个线程顺序/跨步读取或写入远大于 L2
的 device buffer，并用足够并行度产生可调 HBM 带宽。需要四种模式：read、write、
read+write、idle。实际 kernel 带宽由 CUDA event 测量。

在 allpairs 基线中，对单个 source GPU 注入该 kernel，逐级增加实际 HBM GB/s。

判读：

- HBM-read 压力复现与 D2H 相似的 source-local slowdown：支持源 HBM 读路径竞争；
- HBM-write 影响明显更小，且与 D2H/H2D 非对称一致：进一步支持方向性；
- kernel 已接近 HBM 峰值但 D2D 不变：简单的 HBM 总带宽解释变弱。

这个 kernel 使用 SM，会引入 SM 到 L2 的流量，但被测 D2D 仍由 CE 执行。因此
需要结合下一项 L2-resident kernel，不能只做一个“大数组 bandwidth kernel”就宣布
HBM 是根因。

### E3. 受控 L2 压力 kernel

构造工作集明显小于 L2 的 pointer-chase 或循环读写 kernel，并通过重复访问使 DRAM
流量尽可能低、L2 sector/request rate 尽可能高。另做一个大工作集、相同指令结构的
版本作为对照。

Nsight Compute 在这里才是合适工具：它是 CUDA kernel profiler，可确认压力 kernel
的 DRAM throughput、L2 hit rate、L2 sector throughput 和 replay/stall 特征。它不用于
直接分析 `cudaMemcpyPeerAsync` 的 CE 内部行为。可关注的指标族包括：

```text
dram__bytes_read / dram__bytes_write / dram__throughput
lts__t_sectors* / lts__throughput
L2 hit rate
```

指标名称如何映射到具体 profiler，不影响实验设计；采集实现阶段再选择对应的完整
metric 名称。

判读：

- L2-resident 压力在低 DRAM throughput 下仍显著降低 D2D：强支持 L2/片上路径；
- 只有大工作集、高 DRAM throughput 的版本降低 D2D：更支持 HBM；
- 两种 kernel 都不复现 D2H 效应：D2H 特有的 CE/PCIe 出站内部路径成为首选解释；
- 两种都降低，但程度只随总流量变化：只能归因于共同内存路径，不能细分。

### E4. CE 与内部路径消融

比较以下背景：

| 背景 | CE | PCIe | HBM/L2 数据路径 | 用途 |
| --- | --- | --- | --- | --- |
| D2H memcpy | 是 | GPU→Host | 设备读 | 原始效应 |
| H2D memcpy | 是 | Host→GPU | 设备写 | 方向负对照 |
| D2D local memcpy | 是 | 否 | 设备读+写 | CE+本地内存对照 |
| HBM read kernel | 否 | 否 | 设备读 | HBM/SM 路径对照 |
| L2-resident kernel | 否 | 否 | 主要片上 | L2 对照 |
| 空 CUDA API/同步循环 | 否 | 否 | 极低 | host/driver 提交对照 |

如果只有 D2H 能复现，而三个 device-only 压力源均不能复现，则最合理表述是
“D2H 使用的 CE→PCIe 出站/源端内部仲裁路径与 NVLink peer copy 竞争”，而不是
“HBM 带宽饱和”。

**阶段 E 验收：** 至少完成 HBM-read、HBM-write、L2-resident 和 local D2D
四个消融，得到能够支持或削弱每个候选机制的结果。

## 10. Profiling 观测设计

先定义需要的观测量，再选择能够提供这些观测量的工具。这里不评价工具在当前
机器上的安装、权限、版本或硬件支持情况；Nsight Systems、Nsight Compute、DCGM、
CUPTI、NVML 等名称只表示一类可能的采集手段，不构成实验设计的依赖。

需要的观测量分为三类：

1. **操作级时序**：D2D/D2H 的开始、结束、重叠比例、stream 顺序和同步等待；
2. **GPU 级数据通路速率**：HBM、L2、NVLink、PCIe 和 CE 的吞吐或活跃度；
3. **压力 kernel 的性质**：确认它主要制造 HBM 流量还是 L2 流量。

### 10.1 Nsight Systems：必要的时序证据

用途：

- 验证 D2H 与 D2D 在测量窗口内确实重叠；
- 检查 `cudaMemcpyPeerAsync`、D2H memcpy、CUDA events 和 stream synchronize
  的时间关系；
- 判断退化体现为单次/批次 D2D Memcpy 持续时间增长，还是 host/API/sync 等待增长；
- 检查是否存在意外串行化、初始化、allocation 或 page-fault 污染测量窗口。

Nsight Systems 不是 L2 cache profiler。即使它能采 GPU metrics，这些通常也是
设备级时间序列，不能自动归因到某一个 CE memcpy。

仅 profile 四个代表场景，每个 2–5 秒即可：

```text
P0: allpairs + none
P1: allpairs + D2H on all GPUs
P2: allpairs + D2H on GPU0 only
P3: allpairs + H2D on all GPUs
P4: ring + D2H on all GPUs（推荐额外负对照）
```

采集命令模板：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --output=doc/traces/<scenario> \
  <启动背景并运行D2D的受控wrapper> <scenario>
```

应由 wrapper 同时启动 background 和 D2D，使两个进程都属于 profiler 启动的
process tree；否则 `.nsys-rep` 可能只包含 D2D 的 CUDA trace，而看不到背景进程的
D2H Memcpy 行。建议在 benchmark 中用 NVTX 标记 `warmup`、`steady-background`、
`measured-d2d`，随后用 NVTX capture range 限制文件大小。

设备级 metrics 可另做一轮采样，不要与主 trace 混在唯一一次 profile 中，以便区分
时序采集与硬件计数器采集带来的扰动。

### 10.2 Nsight Compute：只用于压力 kernel

Nsight Compute 面向 CUDA kernel。当前主工作负载是 CE memcpy，因此直接运行：

```bash
ncu ./build/cuda_copy/d2d_multi_peer_bw ...
```

不会得到足以判断 D2D memcpy 的 L2/HBM 归因，甚至可能没有可分析 kernel。它只在
E2/E3 的 HBM/L2 压力 kernel 中必要，用来证明“这个对照负载实际制造了预期的
HBM 或 L2 压力”。

压力 kernel profile 与 D2D 性能测试也应分开：ncu 的 replay 会改变并发关系，
不能用 replay 后的 D2D 带宽作为主结果。

### 10.3 设备级遥测：辅助相关证据

设备级遥测适合采集较长窗口中的 DRAM active、PCIe TX/RX、NVLink TX/RX、
clock、power 和 memory-copy utilization。它们有助于确认方向和饱和趋势，但
设备级聚合数据本身不足以区分 L2 与 HBM，因此只是辅助证据。

若使用，最小指标集合为：

```text
DRAM activity
PCIe TX/RX bytes or throughput
NVLink TX/RX bytes or throughput
memory-copy utilization
SM activity（确认纯 memcpy 阶段接近空闲）
SM clock, memory clock, power, temperature
```

无论由哪种工具承载，都应记录采样周期，并与 measured range 对齐。

### 10.4 最小必要工具结论

| 工具/手段 | 是否必要 | 原因 |
| --- | --- | --- |
| CUDA events + benchmark JSON | 必要 | 主性能结果和 per-source 定位 |
| Nsight Systems | 必要 | 证明并发、排除串行化并定位时间增长 |
| NVLink/PCIe/clock 计数器 | 强烈建议 | 排除路径变化、链路/降频解释 |
| Nsight Compute | 条件必要 | 仅在 HBM/L2 压力 kernel 消融中验证负载性质 |
| 设备级遥测 | 辅助 | 适合长时趋势，但不能单独区分 HBM/L2 |

官方工具定位可参考 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
和 [DCGM Profiling 文档](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-api/dcgm-api-profiling.html)。

## 11. 如何阅读 `.nsys-rep`

### 11.1 打开与定位

在 Nsight Systems GUI 中打开 `.nsys-rep`。先确认左侧/时间线中包含两个进程：
D2D benchmark 和 background copy。如果只有一个，
该 trace 不能用于判断两类 memcpy 是否重叠，应修正启动方式后重采。

依次展开：

1. **Processes / Threads**：确认 background worker 和 D2D host thread 的生命周期；
2. **CUDA API**：查看 `cudaMemcpyPeerAsync`、`cudaMemcpyAsync`、
   `cudaStreamSynchronize`、event record/synchronize；
3. **CUDA HW / GPU rows**：按 GPU 展开 CUDA Copy/Memcpy 行；
4. **NVTX**：只框选 `measured-d2d` 与 `steady-background` 重叠区；
5. **GPU Metrics**（若采集）：观察 DRAM、PCIe、NVLink、clock 曲线。

不要把初始化阶段的 `cudaMalloc`、`cudaMemset` 和首次 context creation 纳入比较。

### 11.2 每个场景要读取的内容

对 P0–P4 使用相同缩放比例，记录：

- 每张 GPU 的 peer memcpy 开始、结束和持续时间；
- 同一 source stream 上三个 outgoing memcpy 是否仍串行入队；
- 四个 source stream 是否同时工作；
- D2H 与 D2D 的真实重叠比例；
- D2H 开始后 D2D memcpy duration 是否立刻增长，停止后是否恢复；
- `cudaMemcpyPeerAsync` API 本身是否仍快速返回；
- host 主要等待在何处：stream sync、event sync，还是 API submission；
- 是否出现长空洞、context serialization 或意外 device-wide sync；
- NVLink throughput 是否下降，PCIe throughput 是否与 D2H 区间对应；
- memory clock/P-state 是否在 treatment 中改变。

关键判读：

- API 提交很快，但 GPU memcpy bar 拉长：GPU 数据通路/CE 仲裁证据；
- GPU memcpy bar 不变，host sync/API 变长：优先调查 host/driver/synchronization；
- 只有 GPU0 D2H 时 GPU0 source row 拉长：强 GPU-local 证据；
- D2H 并未覆盖 D2D 测量窗口：该 profile 无效；
- 所有 GPU 行在固定间隔同时出现空洞：可能存在全局同步或测量结构问题。

### 11.3 CLI 摘要

GUI 之外保留可重复的 CLI 摘要：

```bash
nsys stats \
  --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  <scenario>.nsys-rep
```

摘要用于检查 memcpy 时间和字节数，不能替代 GUI/SQLite 中的 per-GPU、per-time-range
分析。需要自动化时导出 SQLite，并只查询 NVTX measured range 内的 CUDA memop；
查询脚本必须随报告保存，避免手工框选产生不可复现的数字。

## 12. 结论决策表

| 观测组合 | 最合理解释 | 不能声称的内容 |
| --- | --- | --- |
| 单 GPU D2H 只降低同卡 source D2D；随 D2H GB/s 单调下降 | GPU-local 源端共享数据路径 | 尚不能区分 HBM/L2/CE |
| HBM-read kernel 复现，write 明显较弱；L2-resident 低 DRAM 负载不复现 | HBM/显存读路径更可能 | 不能说已定位具体 memory partition |
| L2-resident kernel 在低 DRAM 流量下复现，且有 L2 容量附近拐点 | L2 或 L2 后片上路径更可能 | 不能说某个 L2 slice 被确定占满 |
| 只有 D2H CE 复现；HBM/L2 kernel 和 local D2D CE 均不复现 | D2H 特有 CE→PCIe/内部仲裁路径 | 不能归因于 HBM 饱和 |
| D2D 下降伴随 NVLink 已到峰值，D2H 只改变源端 | NVLink 与源端仲裁共同限制 | 不能仅凭 NVLink utilization 证明根因 |
| PCIe/CPU NUMA 改变即可消除效应，所有 GPU 同时变化 | 主机/PCIe 路径因素优先 | 不支持 GPU-local 结论 |
| profiler 下有差异、无 profiler 无差异 | profiler artifact | 不能报告竞争成立 |

推荐的最终措辞示例：

> 在固定时钟、拓扑和主机放置下，四卡 allpairs 的下降与施加 D2H 的源 GPU
> 一一对应，并随该 GPU 的实际 D2H 字节率单调增加；ring、H2D 和远端 GPU 是
> 负对照。Nsight Systems 显示 API 提交未变而 GPU peer-copy duration 增长，且
> D2H/D2D 确实重叠。这支持源 GPU 内部 memory/copy/fabric 数据路径发生竞争。
> 进一步的 HBM/L2 压力消融显示……。由于 Copy Engine 内部仲裁计数器不可完整
> 归因，本实验不声称定位到未公开的具体硬件单元。

## 13. 推荐执行顺序与产物

### 第一批：无需新增 benchmark 功能

1. 阶段 A：稳定性和路径确认；
2. 阶段 B：使用现有 background 程序做 payload/GPU-subset 矩阵；
3. 阶段 C：7 次正式主效应复现；
4. 阶段 D1：单 GPU D2H/H2D 注入，分析已有 per-source JSON；
5. P0–P4 的 Nsight Systems trace。

这一批已经足以判断是否存在 GPU-local 竞争，并决定是否值得继续区分 HBM/L2。

### 第二批：需要小幅扩展工具

1. background JSON 增加 per-GPU bytes/GB/s；
2. runner 支持独立 D2D size、background size、background GPU list 和场景随机顺序；
3. 增加 fan-out=1/2/3 的 D2D edge 配置；
4. 增加 D2H duty-cycle/目标带宽控制；
5. 增加 NVTX measured range；
6. `--chunkRepeats=1/5/20`、100 ms clock/pstate 和 per-iteration synchronize 对照
   已完成；后续以异步 batch 为主协议，使用逐次同步作为队列深度控制，并补充
   background per-GPU bytes/GB/s 后再进行压力匹配。

### 第三批：机制消融

1. streaming HBM read/write kernel；
2. L2-resident 与大工作集对照 kernel；
3. local D2D CE 背景；
4. Nsight Compute 验证压力 kernel 的实际 HBM/L2 特征；
5. 汇总决策表并形成最终报告。

目录建议：

```text
doc/results/gpu-contention/
  baseline-stability/
  host-copy-matrix/
  main-effect/
  per-gpu-injection/
  pressure-sweep/
  hbm-l2-ablation/
doc/traces/gpu-contention/
doc/reports/gpu-internal-copy-contention-results.md
```

每个结果目录必须包含环境快照、原始 JSON/CSV、完整命令、退出状态和未加工 log。
最终报告中的每个表格单元都应能追溯到一个原始文件。

## 14. 停止条件

满足以下任一条件时停止继续细分并报告当前证据：

- 主效应在 7 次正式运行中不稳定；
- 单 GPU 注入没有任何 GPU-local 对应关系；
- 效应可由降频、PCIe 路径、NUMA 或测量结构解释；
- HBM/L2 消融结论互相矛盾且无法用计数器确认；
- 公开工具只能支持“GPU-local internal arbitration”，不能再可靠细分。

“无法严格区分 HBM 与 L2”不是失败。只要方向、局部性、压力响应和时间线共同
排除了主要外部因素，就可以合理地证明竞争发生在 GPU 内部的数据移动路径。
