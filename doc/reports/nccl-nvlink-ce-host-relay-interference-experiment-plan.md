# NCCL NVLink 与 CE Host Relay 双路并行干扰实验方案

## 1. 研究目标

本文设计一套四卡 V100 环境下的分阶段实验，用于评估以下双路通信结构是否能够带来
有效的并行加速，以及两条路径并发时是否发生性能干扰：

```text
主路：NCCL Ring+Simple P2P kernel -> NVLink
旁路：CUDA Copy Engine -> D2H -> pinned host relay -> H2D -> 目标 GPU
```

第一阶段不立即实现真实的双路 collective。CE Host Relay 只搬运独立缓冲区，作为可控
的 PCIe aggressor；完整的 NCCL collective 作为 NVLink victim。这样可以先回答：

1. 非流水化 CE Host Relay 在不同 payload 下的端到端带宽、方向带宽和延迟是多少；
2. 官方 `.sys` NCCL 单独走 NVLink 时，AG、AR、RS 的 clean 性能是多少；
3. CE Host Relay 提前进入 steady state 后，是否拖慢同时运行的 NCCL；
4. NCCL 与 relay 各自下降多少，双路 aggregate useful bandwidth 是否仍高于单路；
5. 干扰与 D2H、H2D、完整 relay、GPU 放置、payload、active duty 的关系是什么；
6. `NCCL_P2P_DISABLE=1` 的 SHM 路径与显式 CE relay 是否产生相同的干扰签名；
7. 若存在稳定下降，NCCL primitive 中的 `waitPeer`、copy/reduce、system fence、step
   store 分别贡献多少；
8. `.sys -> .gpu` 作用域消融能够解释多少额外时间，以及仍未解释的 residual 是多少。

只有完成以上定位并满足本文停止条件后，才进入真实 payload 分流、双路 completion 和
collective 结果合并设计。

## 2. 术语和结论边界

### 2.1 四种不同路径

本文严格区分：

| 名称 | 数据执行者 | 数据路径 | 用途 |
| --- | --- | --- | --- |
| NCCL P2P | NCCL GPU kernel | GPU -> NVLink -> peer GPU | NVLink victim/主路 |
| NCCL SHM | NCCL GPU kernel | GPU -> mapped host memory -> GPU | NCCL-managed PCIe 对照 |
| NCCL P2P CE | CUDA Copy Engine | GPU -> P2P fabric -> peer GPU | 当前 NV2 环境通常仍走 NVLink，不作为 PCIe relay |
| 显式 CE Host Relay | CUDA Copy Engine | D2H -> pinned host -> H2D | 最终方案 B 的 PCIe 旁路 |

`NCCL_P2P_DISABLE=1` 只禁用 direct GPU P2P，单机内通常回退到 SHM。SHM 使用 GPU 可映射
的 host buffer，但它不是显式的 `cudaMemcpyAsync(D2H)+cudaMemcpyAsync(H2D)`。

NCCL 2.31.2 源码中的 `NCCL_P2P_USE_CUDA_MEMCPY=1` 会令 P2P proxy 使用 D2D
`cudaMemcpyAsync`，但 CUDA 会利用可用的 NVLink P2P 路径，因此它不能作为本实验中
独立的 PCIe 旁路。

### 2.2 fence 与 barrier

本文所称 `.sys/.gpu` 是 memory fence scope，不等同于线程执行屏障：

```text
barrier()                         worker/线程组执行同步
fence_acq_rel_sys()              system-scope memory ordering
fence_acq_rel_gpu()              device-scope memory ordering
st_relaxed_sys_global(step, ...) system-scope progress publication
```

`.sys -> .gpu` 只作为因果消融，不预设为正确修复。peer GPU、host、proxy、network/GDR
是否需要 system scope，必须由具体 connection 的生产者、消费者和 completion 协议决定。
实验中不得同时修改 fence 和 step store，否则无法隔离作用域贡献。

### 2.3 允许与禁止的结论

公开工具和本方案打点可以支持：

> 并发 PCIe relay 使某 collective/rank/channel 的某 primitive 阶段变长；只修改目标
> post fence scope 后，额外时间减少了某一比例。

不能仅凭吞吐或 `MEMBAR.SYS` 的存在声称：

> 已确定物理 CE、descriptor FIFO、L2 slice、HBM partition、NVLink scheduler 或
> firmware credit 是根因。

## 3. 当前硬件与源码树职责

### 3.1 当前机器

| 项目 | 当前值 |
| --- | --- |
| GPU | 4 x Tesla V100-SXM2-32GB |
| GPU 架构 | Volta `sm_70` |
| GPU 拓扑 | GPU0/1/2/3 任意两卡均为 `NV2` |
| Host NUMA | 四卡均位于 NUMA node 0 |
| Driver | 580.178.04 |
| PCIe 拓扑 | 四卡到 host 路径显示为同 NUMA node 内 `NODE` |

正式实验前必须重新采集 `nvidia-smi -q`、`nvidia-smi topo -m`、CPU/NUMA、CUDA、编译器
和 GPU clock/power 状态，不能只复用本文记录。

### 3.2 唯一职责划分

```text
/home/songxb26/HyLink/nccl
  官方 NCCL v2.31.2-1
  原始 .sys 实现
  仅 sm_70 构建
  建立基线后不再修改

/home/songxb26/HyLink/Cu-Hylink-Analyse/third_party/nccl
  实验 NCCL v2.31.2-1
  sys 等价构建、打点、.gpu 消融都只发生在这里

/home/songxb26/HyLink/Cu-Hylink-Analyse/third_party/nccl-tests
  唯一 nccl-tests 源码树
  只构建一套测试前端
  运行时通过 LD_LIBRARY_PATH 切换 NCCL 后端
```

禁止在结果生成后用“当前源码状态”推断历史 case 使用的库。每个 case 必须保存独立的
commit、diff、library realpath、hash 和 `ldd`。

## 4. Stage 0：恢复并冻结官方 `.sys` 基线

### 4.1 当前已知不满足项

截至方案编写时，外部 `/home/songxb26/HyLink/nccl` 为 `v2.29.2-1-dirty`，并存在两个
`fence_acq_rel_sys() -> fence_acq_rel_gpu()` 的未提交修改。因此它当前不能作为正式
baseline。

`third_party/nccl` 位于 `v2.31.2-1`，目标 Ring+Simple 主路径仍为 `.sys`；其
`makefiles/common.mk` 有仅生成 `sm_70` 的本地构建修改，并带有 Inspector 相关未跟踪
文件。这些状态必须写入构建清单，不能默认为 clean tree。

### 4.2 外部 NCCL 恢复流程

1. 在外部 NCCL 中用带日期和说明的 `git stash` 保存现有 `.gpu` diff；记录 stash ID；
2. `git fetch --tags`，确认官方 `v2.31.2-1` tag；
3. 从该 tag 建立固定 baseline branch 或 detached worktree；
4. 确认目标 `prims_simple.h` 为原始 `.sys`；
5. 确认 `git status` clean；
6. 只生成 `sm_70`，不包含其他 SASS 架构；
7. 输出到独立目录，不复用任何旧 `build/`；
8. 构建完成后不再修改该源码树。

建议输出：

```text
build/nccl-official-v2.31.2-sm70-sys/
```

### 4.3 实验树 sys 等价构建

在 `third_party/nccl` 先生成未打点 `.sys` release：

```text
build/nccl-experiment-v2.31.2-sm70-sys/
```

它必须在 clean NCCL-only 条件下与 official sys 做等价性校准。若 AG/AR/RS 的 paired
性能差异超出 baseline 噪声范围，不得进入 `.gpu` 消融。

### 4.4 唯一 nccl-tests 构建

从 `third_party/nccl-tests` 生成：

```text
build/nccl-tests-hybrid-sm70/
```

nccl-tests 二进制不得写入固定 NCCL RUNPATH。每次运行前强制：

```text
readelf -d <perf binary>
ldd <perf binary>
realpath resolved libnccl.so.2
sha256sum resolved libnccl.so.2
NCCL version check
```

运行脚本只接受命名后端：

```text
--backend=official-sys
--backend=experiment-sys
--backend=experiment-gpu
--backend=experiment-<instrumentation>
```

禁止要求执行者手工拼接 `LD_LIBRARY_PATH`。

### 4.5 SASS 和构建契约

对每个 NCCL build 保存：

```text
source commit/tag
git status --porcelain
git diff --binary
nvcc version
NVCC_GENCODE
libnccl.so realpath/SHA256
fatbinary/cubin architecture list
目标 kernel SASS fence scope
build command
```

official sys 与 experiment sys 必须包含目标 system-scope fence；experiment gpu 必须只在
patch manifest 指定的位置改变 scope，并保留 `st_relaxed_sys_global`。

## 5. 公共测量规则

### 5.1 CPU、GPU 和 NUMA 固定

- 所有 host relay pinned buffer 分配在 GPU 所在 NUMA node 0；
- background worker 的 CPU affinity 固定并写入 metadata；
- nccl-tests launcher affinity 固定，避免与 relay worker 争用同一 CPU core；
- 记录 persistence mode、application clock、power limit、temperature 和 throttle reason；
- 若不能固定 clock，则每个 case 保存采样，并使用 paired clean-before/after；
- 任何 GPU Xid、ECC、CUDA、NCCL async error 都使 case 失败。

### 5.2 重复和统计

- smoke：1 次，只验证功能和路径；
- screening：至少 3 次；
- formal：至少 7 次独立进程级重复；
- 报告 median、p10/p90、MAD 或 bootstrap confidence interval；
- 不用单次最好值代表性能；
- 所有 treatment 使用 clean-before 和 clean-after 配对；
- clean-after 未恢复到 baseline 容差时，case 标记为 contaminated，不进入根因统计。

正式 slowdown 同时报告两种口径：

```text
before-only slowdown = 1 - concurrent / clean-before
bracket slowdown     = 1 - concurrent / center(clean-before, clean-after)
```

其中 `center` 默认使用算术平均；若 before/after 跨越两个稳定性能档位，则该 repetition 标记
为 baseline-mode-shift，单独报告，不得用 median-of-medians 掩盖。低于 3% 的 slowdown 必须
给出逐次 paired 分布和置信区间，不能只凭聚合中位数宣称干扰成立。

建议 baseline 稳定性门限：连续 7 次的 CV 不高于 2%；若平台噪声更高，必须先建立新的
经验阈值，而不是放宽 treatment 判定。

### 5.3 混合 case 的强制生命周期

nccl-tests 运行较短，因此所有混合流量必须由 PCIe/SHM aggressor 提前启动：

```text
T0  启动 aggressor
T1  aggressor 完成 warmup，并连续两个报告窗口达到 steady state
T2  runner 收到 READY；记录 background start/bytes/throughput
T3  启动 nccl-tests victim
T4  nccl-tests 结束；background 仍保持运行
T5  额外保留 background 1-2 个报告窗口
T6  请求 background 正常退出并回收
T7  运行 clean-after
```

禁止以下顺序：

```text
先启动 nccl-tests，再临时启动 background
background 使用固定少量 iterations，可能在 NCCL 结束前提前退出
通过固定 sleep 猜测 background 已就绪，但不检查实际吞吐
NCCL 一结束就 kill -9，导致背景结果和 flush 丢失
```

background 应支持无限循环模式、READY 文件/pipe、周期 JSON 和 SIGTERM 正常退出。
runner 必须保存 host monotonic timestamps，并验证：

```text
background_steady_begin <= nccl_begin
background_active_end    >= nccl_end
background bytes 在 NCCL window 内持续增加
```

代表 trace 中还需用 GPU activity 时间戳验证实际 overlap。若 background 未覆盖完整 NCCL
measured window，case 必须失败，不能只在备注中说明。

### 5.4 NCCL 运行时长校准

对每个 collective/size 先测单次迭代时间，再自动选择 `-n`，使正式 measured window
达到约 5-10 秒；建议：

```text
warmup >= 20
100 <= measured iterations <= 5000
```

不得为了延长窗口改变 treatment 与 clean 的 `-n/-w`。Inspector/NSys case 可以减少
iterations，但不参与正式吞吐统计。

background 的最终 JSON 若只覆盖从 READY 到 STOP 的整个生命周期，只能用于进程级健康
检查，不能代表 NCCL 并发窗口内的吞吐。relay 必须以 100-500 ms 窗口记录 monotonic
timestamp、completed bytes、D2H/H2D/e2e latency；runner 再按 victim measured begin/end 截取
overlap-only throughput。若暂时无法增加窗口计数，则必须延长 victim，使其占 background
active interval 的绝大部分，并把该限制写入结果。

## 6. Stage 1：非流水化 CE Host Relay microbenchmark

### 6.1 数据路径

第一版严格串行：

```text
cudaMemcpyAsync(host, srcGpu, bytes, D2H, d2hStream)
等待 D2H 完成
cudaMemcpyAsync(dstGpu, host, bytes, H2D, h2dStream)
等待 H2D 完成
下一次 iteration
```

不允许同一 payload 的 H2D 在 D2H 完成前开始；不做 chunk 分片、double buffer 或多 buffer
pipeline。CPU 负责提交和 completion 管理，不执行数据 memcpy。

### 6.2 buffer 与正确性

- host buffer 必须是 pinned/page-locked；
- 每条同时运行的 edge 使用独立 host buffer，避免伪共享；
- source/destination device buffer 不与 NCCL victim buffer 重叠；
- source 写入 iteration/edge 可辨识 pattern；
- warmup 后、正式运行后分别做 destination 校验；
- 校验失败时输出首个错误 offset、expected/actual 和 edge ID；
- 正式计时区禁止逐 iteration 全量校验。

### 6.3 size 扫描

```text
64K, 128K, 256K, 512K,
1M, 2M, 4M, 8M, 16M,
32M, 64M, 128M, 255M, 256M,
512M, 1G
```

512M/1G 先只运行单 edge。四 edge ring 是否运行这些尺寸取决于 pinned-memory 预算和系统
稳定性；runner 必须在分配前检查物理内存和 memlock 限制，不能无条件锁定数 GiB 内存。

### 6.4 拓扑矩阵

```text
单边 forward：0 -> Host -> 1
单边 reverse：1 -> Host -> 0
单边其他 peer：0 -> Host -> 2/3
两条 disjoint：0 -> 1 与 2 -> 3
两条 shared-source：0 -> 1 与 0 -> 2
四卡 ring：0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0
四卡 reverse ring：0 -> 3, 3 -> 2, 2 -> 1, 1 -> 0
```

screening 先做单边、disjoint 和四卡 ring；shared-source 与 reverse ring 用于出现异常后的
局部性诊断。

### 6.5 必须分开的模式

对相同 payload 分别测：

```text
D2H-only
H2D-only
完整 D2H -> H2D relay
```

完整 relay 的一次 iteration 必须记录：

```text
D2H duration
D2H->H2D transition gap
H2D duration
end-to-end duration
iteration interval
actual duty
bytes completed
```

### 6.6 带宽定义

设原始 payload 为 `S`：

```text
D2H bandwidth = S / T_D2H
H2D bandwidth = S / T_H2D
relay useful bandwidth = S / T_end_to_end
relay PCIe traffic rate = 2*S / T_end_to_end
```

最终双路 aggregate 使用 `relay useful bandwidth`，不能使用 `2*S` 人为放大有效数据量。
报告中必须同时给出 traffic rate，便于解释 PCIe 压力。

### 6.7 Stage 1 输出与选择点

输出 payload 曲线、latency 分布、方向带宽、端到端有效带宽和实际 duty。选出：

1. latency/cadence 区的小 payload；
2. throughput knee；
3. steady bandwidth plateau；
4. 大 payload 长 active-window 点。

主矩阵优先使用约 `4M/16M/64M/255M` 中实际落入上述区域的点，而不是预先把某个 size
命名为硬件阈值。

## 7. Stage 2：official `.sys` NCCL NVLink 基线

### 7.1 固定配置

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
NCCL_ALGO=RING
NCCL_PROTO=SIMPLE
NCCL_P2P_DISABLE=0
```

第一轮关闭可能改变 transport/algorithm 的其他调优变量。保存 `NCCL_DEBUG=INFO` 且至少
包含 `INIT,GRAPH,TUNING,P2P,SHM` 的路径日志。

### 7.2 collective 与 size

```text
AllGather
AllReduce(sum)
ReduceScatter(sum)
```

完整 size sweep：

```text
1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1G
```

主干扰矩阵先固定已经有对比度的 `64M`，并用 `256M` 做吞吐区对照；若四卡新环境的
size sweep 显示敏感点已改变，以四卡结果重新选择，不能沿用三卡结论。

### 7.3 路径验收

每个正式 case 必须确认：

- 四个 rank 均建立预期 P2P transport；
- ring edge 与 GPU mapping 符合预期；
- 没有 SHM/NET fallback；
- 算法为 Ring、协议为 Simple；
- channel/thread 配置已记录；
- correctness error count 为零。

### 7.4 基线指标

保存 nccl-tests 的 algorithm bandwidth、bus bandwidth、time 和 error count，同时计算
不同 repetition 的 median/CV。不同 collective 的 busbw 定义不同，不能直接把三者的
busbw 相加作为双路 aggregate goodput。

## 8. Stage 3：official NCCL + CE Relay 并行主矩阵

### 8.1 筛选矩阵

先以 `AG 64M` 为 victim、Stage 1 的 steady relay point 为 aggressor：

| 背景 | placement | 目的 |
| --- | --- | --- |
| none | none | clean baseline |
| D2H-only | GPU0 | source-local 方向对照 |
| H2D-only | GPU1 | destination-local 方向对照 |
| full relay | 0 -> 1 | 单 edge 完整路径 |
| full relay | 2 -> 3 | 与部分 victim edge 的相对位置变化 |
| two relay | 0 -> 1, 2 -> 3 | disjoint 并行 |
| four relay | 0 -> 1 -> 2 -> 3 -> 0 | 全卡 steady stress |

筛选全部使用本文 5.3 的 background-first 生命周期。显著性、恢复性和方向性通过后才扩大
到 AR/RS 和更多尺寸。

### 8.2 正式矩阵

```text
collective = AG, AR, RS
victim size = 64M, 256M
relay size = Stage 1 选出的 knee/steady/long-window 三点
relay placement = single 0->1, disjoint pair, four-GPU ring
regime = clean-before, relay-standalone, concurrent, clean-after
repetitions = 7
```

若矩阵过大，执行顺序为：

1. 所有 collective 的 64M + steady relay；
2. 256M 对照；
3. relay size 三点；
4. 其他 NCCL size extension。

不得只保留出现最大下降的 case；所有预注册 case 都要保留结果和失败原因。

### 8.3 主要指标

```text
NCCL slowdown
  = 1 - NCCL_concurrent / paired_NCCL_clean

relay slowdown
  = 1 - relay_concurrent / relay_standalone

aggregate useful throughput
  = NCCL payload useful throughput + relay useful throughput

aggregation efficiency
  = aggregate concurrent useful throughput
    / (standalone NCCL useful throughput + standalone relay useful throughput)
```

NCCL 的 algorithm bandwidth 可用于 payload useful throughput；busbw 单独报告，不与 relay
直接相加。

上述 aggregate 只用于判断“两个独立满尺寸工作负载并发时是否存在容量余量”，不能直接
当作真实双路 collective 加速比。真实 payload 分片必须使用：

```text
T_total = max(T_NVLink_partition, T_host_relay_partition)
```

并按照 AG/AR/RS 的实际数据语义计算；特别是 AR/RS 的 relay lane 还缺少 reduction ownership
和 partial-result merge，因此当前 `NCCL algbw + relay useful bandwidth` 只能写为 feasibility
proxy。

还要报告：

```text
NCCL slowdown 是否小于 relay 带来的增量
双路 aggregate 是否超过 NCCL-only
CE relay 是否在 NCCL 活跃期间降速
clean-after 恢复率
不同 rank/channel 的 tail
```

### 8.4 进入 profile 的门限

满足以下任一项进入 Stage 5/6：

- NCCL median slowdown 稳定超过 5%；
- aggregate gain 小于 3% 或为负；
- 某 collective/rank/channel 出现明显长尾；
- `.sys` 敏感性的已有预测与方向/placement 结果不一致，需要定位 residual。

若下降低于 2% 且 aggregate gain 稳定为正，先扩展 size/duty，不立即修改 NCCL。

## 9. Stage 4：NCCL SHM 对照

### 9.1 双进程结构

两个独立 nccl-tests 进程组使用相同四张 GPU 和独立 device buffer：

```text
victim process:
  official sys NCCL
  NCCL_P2P_DISABLE=0
  Ring+Simple

aggressor process:
  official sys NCCL
  NCCL_P2P_DISABLE=1
  NCCL_SHM_DISABLE=0
  Ring+Simple
```

环境变量在进程启动前固定，不能在一个进程内动态切换 communicator 的 P2P 属性。

### 9.2 启动顺序

SHM aggressor 必须先启动并持续循环/运行足够多 iterations。日志确认全部预期连接为
`via SHM/direct` 后，再启动 P2P victim。victim 结束后 aggressor 继续保留 1-2 个报告
窗口，然后正常退出。

### 9.3 对照解释

比较：

```text
P2P NCCL + CE D2H-only
P2P NCCL + CE H2D-only
P2P NCCL + CE full relay
P2P NCCL + NCCL SHM
```

若 CE relay 干扰小而 SHM 干扰大，优先考虑第二套 NCCL kernel、mapped-host access、polling
和 fence；若两者签名相近，优先考虑 PCIe/system-memory traffic 对主 NCCL 的共同影响。

NCCL SHM 只作为对照，不作为最终方案 B 的性能替代。

## 10. Stage 5：低扰动宏观 profile

### 10.1 工具职责

| 工具 | 用途 | 不允许的解释 |
| --- | --- | --- |
| NCCL Inspector | collective/rank/channel 时长与 straggler | 直接定位到某条 fence 指令 |
| Nsight Systems | NCCL kernel 与 D2H/H2D 的真实 overlap、CUDA API、memcpy 时间线 | 直接命名内部 FIFO/credit |
| CUPTI memcpy activity | direction、bytes、stream、HW channel、activity overlap | 将 channel ID 等同于物理 CE instance |
| Nsight Compute | 选定 kernel 的公开计数器 | 与正式性能结果混用 replay 数据 |

正式吞吐 case 默认关闭所有 profiler。每个工具独立重跑代表 case，不叠加 Inspector、NSys、
NCU 和自定义打点。

### 10.2 代表 case

至少采集：

```text
AG/AR/RS 64M clean
AG/AR/RS 64M + full four-GPU relay
最弱干扰 case
最强干扰 case
P2P victim + SHM aggressor
256M clean/concurrent 对照
```

NSys/CUPTI 必须检查 background activity 覆盖完整 victim measured window。若 trace 注入导致
background 或 NCCL 吞吐改变超过校准阈值，只将其用于定性时序。

### 10.3 宏观问题

1. NCCL kernel 是否与 D2H、H2D 分别真实重叠；
2. slowdown 只发生在 D2H active window、H2D active window，还是整个 relay 周期；
3. 最慢 rank 是否为 relay source/destination，还是经 collective 同步传播到其他 rank；
4. 最慢 channel 是否稳定；
5. NCCL kernel 变长还是 kernel launch/start 被延迟；
6. CE HW channel 分配是否随 topology/stream 改变；
7. SHM 对照是否出现相同 rank/channel 签名。

## 11. Stage 6：third_party NCCL primitive 打点

### 11.1 前置等价性

先比较 official-sys 与 experiment-sys：

```text
AG/AR/RS
64M/256M
clean/concurrent
7 repetitions
```

algorithm、protocol、channel/thread、correctness 必须一致。若性能差异超过各自 baseline
噪声，不得用 experiment build 解释 official build 的根因。

### 11.2 独立构建变体

```text
sys-release
sys-inst-wait
sys-inst-fence
sys-inst-store
sys-inst-post
sys-inst-copy
gpu-release
gpu-inst-wait
gpu-inst-fence
gpu-inst-store
gpu-inst-post
gpu-inst-copy
```

每个 build 使用独立目录、build ID、patch manifest 和 library hash。instrumented build
只用于阶段组成，release build 用于吞吐真值。

### 11.3 打点对象

#### waitPeer

记录 wait-recv/wait-send、spin load 次数、目标 step、初始 step 和包围时间。

#### copy/reduceCopy

记录 primitive signature、slice bytes、copy/reduction 属性和协作线程组阶段时间。

#### fence

仅由实际执行 post-send fence 的角色线程记录：

```cpp
t0 = globaltimer();
fence_acq_rel_sys();
t1 = globaltimer();
```

#### step store

独立 build 记录：

```cpp
t0 = globaltimer();
st_relaxed_sys_global(connStepPtr, step);
t1 = globaltimer();
```

#### postPeer

另一个独立 build 包围 fence+store 整体。不能在一个 build 中同时加入所有 timer，再把绝对
时间作为原始关键路径。

### 11.4 低扰动记录

- 不使用 device `printf`；
- 每 rank/channel/group 使用独立固定 ring buffer；
- 只由目标角色线程写入；
- 支持 `1/N` step 采样；
- 采样率扫描 `N=1,8,32,128`；
- buffer 满后停止并报告 overflow，不静默覆盖；
- communicator teardown 后一次性导出；
- 期望 profiler overhead 不高于约 2%，超出则降低采样率并只做定性证据。

记录字段至少包括：

```text
rank, channel, collective, primitive, group, step,
slice_bytes, phase, start_ns, duration_ns,
spin_load_count, sample_sequence, overflow
```

### 11.5 贡献量级

对相同 collective/size/rank/channel：

```text
delta_coll  = Tcoll(concurrent)  - Tcoll(clean)
delta_wait  = Twait(concurrent)  - Twait(clean)
delta_fence = Tfence(concurrent) - Tfence(clean)
delta_store = Tstore(concurrent) - Tstore(clean)
delta_copy  = Tcopy(concurrent)  - Tcopy(clean)
```

沿关键 rank/channel 的串行 primitive 路径估计：

```text
critical-path increment
  ~= effective fence count * delta_fence
   + critical wait increment
   + copy/reduce increment
   + unexplained residual
```

不得对并行 channel 的所有样本直接求和。必须同时关注 p50/p90/p99/max，防止均值掩盖
active-window 触发的长尾。

## 12. Stage 7：`.sys -> .gpu` 作用域消融

### 12.1 patch 约束

只修改 Ring+Simple 正式路径中预注册的 post-send fence：

```text
fence_acq_rel_sys() -> fence_acq_rel_gpu()
```

保持：

```text
st_relaxed_sys_global(connStepPtr, step)
wait/load 语义
algorithm/protocol/channel/thread
其他 transport/path 的 fence
```

若 v2.31.2 存在多个 template/PAT/NVLS 路径，必须通过编译实例、NCCL 日志和 SASS 确认
正式 Ring+Simple case 实际命中的位置；不能批量替换全文件所有 `.sys`。

### 12.2 性能矩阵

```text
official-sys
experiment-sys
experiment-gpu

x clean
x CE full relay
x D2H-only
x H2D-only
x SHM aggressor
```

对 AG/AR/RS 的 64M 敏感点和 256M 对照点分别运行 7 次。

定义：

```text
scope-removable fraction
  = (delta_sys - delta_gpu) / delta_sys

residual slowdown
  = delta_gpu
```

只有当细粒度 `delta_fence`、collective critical-path 数量级和 release `.gpu` 消融方向一致
时，才能把 system-scope fence 写为主要贡献者。

### 12.3 correctness 和安全边界

`.gpu` build 必须额外执行：

- nccl-tests correctness；
- 随机 pattern；
- AG/AR/RS；
- 小/中/大 size；
- 长迭代和重复 communicator 创建销毁；
- clean、CE relay、SHM background；
- 不同 rank order/ring；
- CUDA/NCCL async error 检查。

一次正确性失败即停止性能结论。即使当前单机 V100 case 全部通过，也只能说明该实验范围
未观察到错误，不能把 `.gpu` 无条件推广到 host/network/GDR 或任意 peer-GPU path。

## 13. Stage 8：真实双路 collective 的进入条件

满足全部条件后，另行设计实现：

1. CE relay standalone 已找到稳定 knee/plateau；
2. official NCCL + relay 的 aggregate useful throughput 稳定高于 NCCL-only；
3. 干扰具有可重复性且 clean-after 恢复；
4. 已区分 D2H、H2D、完整 relay 和 NCCL SHM；
5. 已量化 wait/fence/store/copy 与 residual；
6. 已确定 `.gpu` 仅是性能消融还是具有受限正确性使用条件；
7. 已得到 relay chunk size 和目标 NVLink/PCIe 分流比例的初始范围。

真实实现再引入：

```text
collective payload partition
NVLink lane
PCIe relay lane
double/multi buffering
cross-device completion
result merge/reduction ownership
load balancing
failure fallback
```

AG 可以最先实现；RS/AR 还需定义 reduction ownership 和两路 partial result 的合并，不能
直接照搬 AG 数据分片。

## 14. 阶段门限与决策表

| 结果 | 解释优先级 | 下一步 |
| --- | --- | --- |
| relay standalone 很低 | 非流水 host relay 自身受限 | 先找 knee/NUMA/方向瓶颈，不进入 NCCL 修改 |
| NCCL 不降、aggregate 上升 | 初始双路可行 | 扩展 size/duty，再设计真实分流 |
| NCCL 降但 aggregate 仍上升 | 存在干扰但可能值得使用 | profile 并优化分流比例 |
| NCCL 降且 aggregate 不升 | 双路收益被干扰抵消 | 进入 Stage 5-7 根因消融 |
| D2H-only 强、H2D-only 弱 | source-read/outbound 路径优先 | 对齐 active window、rank/channel |
| H2D-only 也强 | 非 D2H-only 机制 | 检查全卡方向、CE/HBM/fence 共同作用 |
| SHM 强、CE relay 弱 | NCCL kernel/SHM polling/fence 优先 | 比较 primitive 与 mapped-host access |
| CE relay 与 SHM 都强 | PCIe/system-memory 共因优先 | 检查 overlap、方向和 scope 消融 |
| `.gpu` 消除大部分 delta | system scope 为主要贡献候选 | 用 fence timing 和 correctness 交叉验证 |
| `.gpu` 仅消除少量 delta | 多因素或 fence 非主导 | 分析 wait/copy/store/residual |
| `.gpu` 出错 | 作用域不满足正确性 | 停止将其作为优化，仅保留因果实验 |

## 15. 结果目录与元数据

建议：

```text
doc/results/nccl-hybrid-path/
  environment/
  relay-standalone/
  nccl-official-baseline/
  ce-relay-interference/
  shm-control/
  backend-equivalence/
  primitive-instrumentation/
  fence-scope-ablation/

doc/traces/nccl-hybrid-path/
  inspector/
  nsys/
  cupti/
  ncu/
```

每个 case 保存：

```text
case.json
command.txt
environment.txt
topology.txt
cpu-affinity.txt
backend-manifest.txt
ldd.txt
nccl-debug.log
victim stdout/stderr
background stdout/stderr
background periodic JSON
timestamps.json
result.json
validation.json
profiler-overhead.json（若适用）
```

case ID 必须显式编码：

```text
backend
collective
victimSize
aggressorType
relaySize
relayEdges
direction
warmup/iterations
repetition
```

必须把 `victimSize` 和 `relaySize` 分列，避免再次出现两个 size 同时变化的混杂。

## 16. 推荐执行顺序

### 第一批：基线和 standalone

1. 恢复并冻结 official v2.31.2 `.sys` sm_70；
2. 构建 experiment sys 和唯一 nccl-tests；
3. 校验动态链接和 official/experiment sys clean 等价性；
4. 完成 CE relay 单边/四卡 size sweep；
5. 完成 official NCCL 四卡 AG/AR/RS size sweep。

### 第二批：最小干扰筛选

1. AG 64M + D2H-only；
2. AG 64M + H2D-only；
3. AG 64M + single full relay；
4. AG 64M + disjoint pair；
5. AG 64M + four-GPU ring relay；
6. 每个 background 都提前 steady，再启动 nccl-tests。

### 第三批：正式矩阵和 SHM 对照

1. AG/AR/RS 64M；
2. AG/AR/RS 256M；
3. relay knee/steady/long-window；
4. P2P victim + SHM aggressor；
5. 7 次 paired repetitions。

### 第四批：只对代表点 profile

1. Inspector rank/channel；
2. NSys overlap；
3. CUPTI CE channel/activity；
4. third_party primitive 独立打点；
5. `.sys/.gpu` release 消融；
6. correctness/stress。

## 17. 最终交付问题

实验报告必须回答：

1. 非流水 CE Host Relay 的端到端有效带宽和 PCIe traffic rate 曲线是什么；
2. 四卡 official `.sys` NCCL 的 AG/AR/RS clean 曲线是什么；
3. background-first 并发时，NCCL 和 relay 分别下降多少；
4. aggregate useful throughput 是否真正超过 NCCL-only；
5. 干扰是 D2H、H2D、完整 relay，还是全卡并发才触发；
6. CE relay 与 NCCL SHM 的干扰签名是否一致；
7. slowdown 落在哪些 rank/channel/active window；
8. wait/fence/store/copy 各自的增量和 critical-path 贡献是多少；
9. `.gpu` 消除了多少额外时间，是否通过限定范围的 correctness；
10. residual 是否足以阻止真实双路 collective；
11. 下一阶段应选择的 chunk size、relay placement 和初始分流比例是什么。

在这些问题得到回答前，不进入流水化优化，也不把 `.sys -> .gpu` 当作已经成立的产品修复。

## 18. 2026-08-30 初步结果复盘与下一轮实验

本节基于
[`nccl-nvlink-ce-host-relay-interference-results.md`](nccl-nvlink-ce-host-relay-interference-results.md)
和对应原始日志，对 Stage 1-4 的阶段性证据作重新归纳。它覆盖前面尚未执行时的预测；后续
实验以本节优先级为准。

### 18.1 已经能够确认的现象

64M、7 次进程级 paired repetition 显示，干扰随 relay 覆盖的 GPU/edge 数量增加而增强：

| background | AG median slowdown | AR median slowdown | RS median slowdown | 证据强度 |
| --- | ---: | ---: | ---: | --- |
| single relay | 约 1-2% | 约 1-2% | 约 1% | 未稳定高于当前基线波动 |
| disjoint pair | 6.1% | 2.7% | 4.6% | 中等，逐次结果均为正 |
| four-ring relay | 12.2% | 7.8% | 10.9% | 强，跨 collective/重复一致 |

four-ring 的逐次 paired slowdown 范围为：AG `8.13%-12.88%`、AR
`4.49%-7.98%`、RS `7.62%-12.13%`；disjoint pair 也在所有重复中为正。因此可以确认：

1. official `.sys` NCCL NVLink victim 与显式 CE Host Relay 并发时存在真实干扰；
2. 干扰主要随参与 GPU 数、并发 CE/host-memory traffic 强度和 topology 扩大；
3. four-ring 结果远高于基线噪声，不能用随机抖动解释；
4. clean-after 没有 correctness failure 或不可恢复的持久性退化。

当前 clean NCCL 存在约 4%-5% 的性能档位变化，例如 AG/RS 会在约 `130` 和
`137 GB/s` 两档之间切换。因此 single relay 的约 1%-2% 下降只能写为“可能存在”，不能
写成已经与噪声分离；后续必须采用 5.2 的 bracket baseline、时钟遥测和随机化执行顺序。

### 18.2 当前最合理的机制解释

four-ring standalone aggregate useful bandwidth 约 `10.154 GB/s`，混合 case 的典型值约
为 `8.3-8.6 GB/s`，同时 NCCL 下降约 `8%-12%`。这表明两条路径是双向竞争，而不是 CE
单方面抢占 NCCL：

```text
CE DMA
  -> GPU local HBM/L2 read/write
  -> PCIe/host-memory transaction
  -> I/O/system coherence completion
  <-> NCCL kernel 的 HBM/L2/NVLink access 与 system-scope ordering
```

NVLink payload 不经过 PCIe，因此“PCIe 链路带宽耗尽”不足以单独解释 NCCL slowdown。
当前优先假设是：CE、L2/HBM 和 I/O/system-domain transaction 形成基础竞争，NCCL primitive
中的 `.sys` fence 可能等待这些 system-domain transaction 或放大其长尾。

该解释仍不是根因定论。现有实验没有直接测量 fence latency，也没有完成只改变 scope 的
`.sys/.gpu` release 对照；所以不能排除 CE/HW channel 仲裁、L2/HBM 回压、PCIe coherent
transaction 或多个因素叠加。后续报告应使用：

```text
基础资源竞争 + system-scope fence 放大
```

作为待验证模型，而不能写成“已经证明全部下降由某一条 `.sys` fence 造成”。

### 18.3 当前双路容量判断

把并发 NCCL algbw 与 relay useful bandwidth 相加，仅作为独立工作负载的容量 proxy，可得：

| relay topology | AG proxy gain | AR proxy gain | RS proxy gain |
| --- | ---: | ---: | ---: |
| single | +2.9% | +6.9% | +3.6% |
| disjoint pair | -0.8% | +7.0% | +1.1% |
| four-ring | -5.3% | +3.5% | -4.6% |

由此可确定：朴素地跑满四个 relay edge 不是 AG/RS 的最优策略；single 或 disjoint pair 更接近
可用区域。AR 的 proxy 更有余量，但当前 relay lane 没有执行 reduction，不能将该数字写成
AR 的实际加速。真实实现必须按 8.3 的 partition makespan 建模，并通过 chunk/edge/duty
搜索找到两路同时完成的比例。

### 18.4 SHM 结果的证据边界

一次 AG 64M screening 中，P2P victim 从 `132.88 GB/s` 降至 `45.87 GB/s`，slowdown
`65.48%`；SHM aggressor 日志确认连接为 `via SHM/direct`，其自身约 `3.12 GB/s`、单次约
`21.5 ms`。该签名明显强于显式 CE relay，但不能直接归因于 fence，因为 SHM aggressor
同时引入：

- 第二套 NCCL kernel 和 CUDA context；
- SM 调度/占用；
- mapped-host load/store 与 GPU polling；
- system-scope fence/progress publication；
- PCIe/host-memory transaction。

同一次 concurrent 日志中 out-of-place 为 `45.87 GB/s`、in-place 为 `63.32 GB/s`，差异
较大；当前又只有一次 screening。因此该结果只用于选择代表 case，不能进入 fence 贡献量
计算。

SHM formal 必须使用显式 READY/STOP 和时间戳，并增加以下必需对照：

```text
P2P victim alone
P2P victim + P2P NCCL aggressor       # 扣除双 kernel/context 通用竞争
P2P victim + SHM NCCL aggressor       # kernel + mapped-host/polling/fence
P2P victim + 等强度显式 CE relay       # 无第二套 NCCL primitive
```

四组都使用相同 victim `-n/-w`、完整 measured-window overlap 和至少 7 次 paired repetition。
只有 `P2P+SHM` 相对 `P2P+P2P` 的增量，才可作为 mapped-host/polling/fence 组合贡献。

### 18.5 下一轮 P0：先修正 overlap 和统计可信度

在任何 `.sys/.gpu` 结论前完成：

1. 给 relay 增加 100-500 ms 周期计数，记录 monotonic timestamp、每 edge completed bytes、
   D2H/H2D/e2e latency；
2. runner 记录 victim measured begin/end，并只计算精确 overlap 窗口内的 relay throughput；
3. 把正式 victim window 延长到 5-10 秒，clean 和 treatment 固定相同 `-n/-w`；
4. 对 64M 代表点复测，确认修改测量方式不会改变 four-ring 干扰签名；
5. 将 256M 从单次 screening 提升为 7 次 formal；
6. 随机化 background 顺序，同时记录 GPU clocks、P-state、温度、功率和 throttle reason；
7. 同时报告 before-only、bracket slowdown、逐次范围和置信区间。

当前 relay JSON 覆盖约 8 秒的整个 background 生命周期，而 nccl-tests measured window 更短，
所以其中的 aggregate relay bandwidth 不能当作精确 concurrent throughput。P0 完成前，18.3
中的 proxy gain 只保留为方向性筛选。

### 18.6 下一轮 P1：`.sys/.gpu` 单变量 release 消融

official NCCL 保持冻结；所有修改只发生在 `third_party/nccl`。首先建立与 official 等价的
experiment-sys，然后生成只修改预注册 post-send fence scope 的 experiment-gpu。最小矩阵：

| backend | background |
| --- | --- |
| official-sys | clean / D2H / H2D / single / disjoint / four-ring / SHM |
| experiment-sys | 同上 |
| experiment-gpu | 同上 |

固定 AG/AR/RS、64M，优先 four-ring 和 SHM 代表点；256M 作为尺寸对照。计算：

```text
delta_sys = slowdown(experiment-sys, background)
delta_gpu = slowdown(experiment-gpu, background)

scope-removable fraction = (delta_sys - delta_gpu) / delta_sys
residual slowdown        = delta_gpu
```

判定规则：

- `.gpu` 只在 host background 下改善且 clean 不变：支持 fence 放大模型；
- clean/concurrent 同比例变化：更像普通 kernel 实现差异；
- `.gpu` 后仍有明显 residual：继续归因 CE/L2/HBM/PCIe 等共享资源；
- 任一 correctness/stress 失败：停止将 `.gpu` 作为优化，只保留为因果消融。

### 18.7 下一轮 P2：profile 与 primitive 打点顺序

只对 `clean / four-ring / SHM / P2P+P2P` 代表点执行：

1. Nsight Systems：先确认 NCCL kernel 与 D2H/H2D 的精确 overlap，以及 slowdown 是 kernel
   变长还是 launch/start 延迟；
2. NCCL Inspector：定位 collective/rank/channel straggler 和 active window；
3. CUPTI activity：记录 memcpy direction、stream、bytes、公开 HW channel/activity；
4. third_party primitive 打点：分别构建 wait、fence、store、post、copy 变体，不在同一 build 堆叠；
5. fence 前后使用 `globaltimer/clock64` 低采样率记录，沿关键 rank/channel 估算 critical-path
   increment；
6. Nsight Compute 仅用于已经缩小的单 kernel/site，不参与原始并发吞吐真值，因为 replay 和
   serialization 会改变干扰关系。

最终只有当 release scope 消融、细粒度 fence timing 和 Inspector/NSys 关键路径三者方向及
数量级一致时，才将 system-scope fence 写为主要贡献者；否则继续报告 residual，不向更底层
不可观测的具体物理 FIFO、CE instance 或 firmware credit 过度归因。

### 18.8 更新后的执行门限

下一阶段按以下顺序执行，不并行跨越证据门限：

```text
P0 overlap/statistics 修正
  -> 64M 代表点复现 + 256M formal
  -> SHM 四组控制矩阵
  -> official/experiment sys 等价性
  -> sys/gpu release scope 消融
  -> NSys/Inspector/CUPTI 代表点
  -> primitive fence/wait/store/copy 打点
  -> 再决定真实 AG 双路分片实现
```

进入真实双路 AG 的最低条件为：使用 overlap-only 数据后，至少一种 single/disjoint relay
配置的真实 partition 模型仍给出稳定正收益；并且 residual、正确性边界和初始 relay duty
已经量化。four-ring 在当前 AG/RS 数据下不作为默认实现起点。

### 18.9 2026-08-31 执行记录与当前门限

本轮已完成前述 P2 primitive 计划的第一阶段：

1. `sys-release` 已使用 `PRIMITIVE_TRACE_FENCE_SCOPE=sys`、`PRIMITIVE_TRACE=0` 独立构建，
   64M disjoint AG/AR/RS 的 7-rep release 结果已完成；
2. system-scope 的 `sys-inst-wait/fence/store/post/copy` 和 GPU-scope 的对应独立
   instrumented build 均已完成，均为 V100 `sm_70`、`-j80`；
3. 每个 primitive 在 64M disjoint pair 上完成 AG/AR/RS 的 N=128 代表点，以及 AG 的
   N=1/8/32/128 sample sweep；代表点 `wrong=0`、窗口 marker 唯一、relay telemetry
   完整、victim/relay/helper 返回码为 0，N=128 无 overflow；
4. wait 原始记录已包含 wait-recv/send 的 spin-load，post 包含 fence+store，fence、store、
   copy 保持独立 build，满足不能在单一 build 中堆叠绝对计时的约束。

上述记录随后发现存在显著 observer effect：当前 trace buffer 位于 host-mapped memory，且
每次发布采样记录都会额外执行 `__threadfence_system()`。因此 18.9 中直接继续 critical-path
数量级核对和根据 single/disjoint proxy 进入真实 AG 的顺序作废，由第 19 节的新计划覆盖。

## 19. 2026-08-31 primitive trace 审计后的新实验计划

### 19.1 新计划要回答的问题

已有 release、Inspector、NSys 和方向/拓扑结果支持以下待验证模型：

```text
CE/本地内存/I/O 基础竞争
  -> 某些 rank/channel 的数据推进产生长尾
  -> Ring wait/polling 将局部停顿传播到 collective
  -> system-scope post-send fence 在高 system traffic 下进一步放大长尾
```

新实验必须回答：

1. 去掉 tracer 自身的 host-mapped write 和额外 `.sys` fence 后，production NCCL 中的
   fence/wait 长尾是否仍存在；
2. `.sys/.gpu` release 差异是否在同一随机化 block、相同 relay 强度下复现；
3. collective 长尾与 D2H-active、H2D-active、mixed、transition 哪个时间窗口相关；
4. `.gpu` 后的 residual 主要随方向、参与 GPU 数、CE duty 还是 aggregate traffic 变化；
5. 是否存在一个覆盖完整四卡 AG 语义、而非 single-edge proxy 的正收益 operating point。

在这些问题完成前，不运行 NCU replay，不进入产品化 `.gpu` 修改，也不把 single-edge 的正
proxy gain 写成四卡 AG 可加速证明。

### 19.2 Stage 9：重构为 device-only primitive tracer

#### 19.2.1 当前 tracer 为什么不能用于定量归因

当前实现通过 `ncclCudaHostCalloc()` 分配 host-mapped trace ring。被采样线程写完 record 后
执行：

```cpp
__threadfence_system();
record->valid = 1;
```

这会在被研究的 NCCL kernel 内重新引入 mapped-host transaction 和 system-scope fence；
即使被测 build 的 protocol fence 已改成 `.gpu`，tracer 仍然包含额外 `.sys`。现有 disjoint
release slowdown 约为 3%-6%，而多个 instrumented case 的 treatment 吞吐下降达到
40%-50%，说明 tracer 已改变干扰机制，不能视为低扰动观察者。

现有 duration 还有口径问题：一条 record 会聚合一个 sampled work 内的多个事件。例如 AG
fence record 常见 `eventCount=6`，表中 `durationNs` 是 6 次 fence 的和，不是单条 fence。
旧数据可以保留为定性线索，但不得用于 production critical-path 百分比。

#### 19.2.2 新 tracer 的存储与导出契约

新 tracer 必须满足：

1. trace ring 使用 device memory，不使用 pinned/mapped host memory；
2. NCCL kernel 活跃期间不执行任何为 trace 发布服务的 `.sys` fence；
3. host 仅在被测 kernel/communicator 已完成同步后，通过一次 D2H copy 导出 trace；
4. slot 由 `rank/channel/work/group/role/eventIndex` 确定性寻址，不在关键路径上使用全局 atomic
   分配；
5. buffer 不使用会静默覆盖仍需样本的 modulo ring；越界只设置 device overflow flag；
6. `startNs/endNs/durationNs/eventCount/sliceBytes/spinLoadCount` 全部保留；
7. fence、wait、store、post、copy 继续使用独立 build，禁止在一个 build 中堆叠计时；
8. trace 导出过程不计入 nccl-tests measured window。

fence 正式模式优先每个 slot 保存一次独立 fence event。若仍需按 work 聚合，输出必须同时
提供：

```text
per-work duration sum
eventCount
per-work mean = duration sum / eventCount
```

`per-work mean` 不能冒充单事件 p99；只有逐事件记录才能生成单 fence 的 p50/p90/p99。

按当前 `-n5000`、`samplePeriod=128`、channel/group/role 和每 work 最多事件数预计算容量，
正式运行前必须证明 buffer 足够，不能等运行结束后才发现 overflow。N=128 是起始值，不因
旧 host-mapped tracer 的 N=128 无 overflow 就自动成为新 tracer 的最终采样率。

#### 19.2.3 observer-effect 验收矩阵

每个 primitive 先只跑 AG 64M、disjoint `0:1,2:3`：

```text
release trace-off
instrumented compiled but runtime-disabled
instrumented N=512
instrumented N=256
instrumented N=128
```

每组执行 clean-before/concurrent/clean-after，至少 7 次，顺序随机化。验收条件：

- runtime-disabled 相对 release 的 clean 差异 95% CI 包含 0，绝对均值不超过 1%；
- N=128 的 clean overhead 不超过 2%；
- N=128 测得的 concurrent slowdown 与对应 release 相差不超过 `max(1 percentage point,
  release slowdown 的 20%)`；
- N=128 无 overflow/dropped record；
- N=128 与 N=256 的 per-event p50 相差不超过 10%，p99 相差不超过 20%；
- 所有 case `#wrong=0`、window marker 唯一、relay telemetry 完整。

任一条件失败时，不扩大到 AR/RS；先降低采样密度、扩大 device buffer 或减少 record 字段。
不得通过放宽验收阈值让 tracer 进入正式根因统计。

### 19.3 Stage 10：matched `.sys/.gpu` release 因果矩阵

#### 19.3.1 为什么需要重跑

现有 four-ring `.sys/.gpu` 分别成批运行，虽然所有 collective/size 都同向改善，但仍可能受
跨批次状态影响。新完成的 disjoint release 原始结果为：

| collective | `.sys` disjoint | `.gpu` disjoint | 绝对改善 |
| --- | ---: | ---: | ---: |
| AG | 5.46% | 5.29% | 0.17 pp |
| AR | 3.35% | 2.85% | 0.50 pp |
| RS | 6.09% | 4.85% | 1.24 pp |

它提示 fence scope 的贡献可能随 system traffic 强度非线性增加：AG disjoint 中几乎不可见，
four-ring 中才达到明显比例。必须在相同 block 内直接验证这一交互。

#### 19.3.2 正式矩阵

```text
backend = experiment-sys-release, experiment-gpu-release
collective = AG, AR, RS
victim size = 64M
background = none, D2H-only, H2D-only, single, disjoint, four-ring
repetitions = 7
```

每个 repetition 内随机决定 sys/gpu 顺序；每个 backend 都独立执行
`clean-before -> treatment -> clean-after`。同一 paired block 固定 relay payload、edge、CPU
affinity、victim `-n/-w` 和时钟策略。除 slowdown 外，必须检查两 backend 在 victim window
内的 relay useful/traffic rate 是否匹配；差异超过 3% 的 pair 标记为 intensity-mismatch，
不进入 scope interaction 主统计。

主统计量：

```text
scope interaction = slowdown_sys - slowdown_gpu
scope-removable fraction = scope interaction / slowdown_sys
```

同时报告 absolute percentage-point difference，不只报告 fraction；当 `slowdown_sys < 2%`
时 fraction 不稳定，只报告 interaction 和 CI。256M 只重跑 four-ring 代表点，除非 64M
出现与已有结果相反的方向。

### 19.4 Stage 11：CE phase 与 collective 长尾精确对齐

#### 19.4.1 时间线要求

100 ms relay throughput telemetry 足以做带宽统计，但不足以判断某次约 0.5-1 ms collective
踩中了 D2H 还是 H2D。代表 case 需要为每次 CE activity 记录：

```text
edge, direction, stream
submit timestamp
GPU activity begin/end
bytes
iteration/phase sequence
```

优先使用同一 Nsight/CUPTI 时间域；若使用 host marker，必须保存时间域映射和误差。Inspector
或低开销 operation marker 提供每个 collective/rank/channel 的 begin/end。所有 profiler
先做 off/on 开销校准，正式吞吐仍来自 profiler-off release case。

#### 19.4.2 代表矩阵

```text
backend = experiment-sys, experiment-gpu
collective = AG, RS；AR 作为 scope 高占比补充点
background = D2H-only, H2D-only, single, disjoint, four-ring
size = 64M
```

对每条 collective operation 按其与 CE activity 的重叠分类：

```text
none
D2H-active
H2D-active
D2H+H2D mixed
transition/queue gap
```

分别计算 p50/p90/p99/max，并定位最慢 rank/channel 与本地 relay source/destination 的关系。
预注册假设是 H2D-active 的 tail 高于 D2H-active；若数据不支持，必须放弃“H2D ingress 是
residual 主导因素”的解释，而不是重新选择时间窗口。

### 19.5 Stage 12：方向、覆盖 GPU 数与 duty 的 residual 消融

现有结果中，single relay traffic 与单向 copy 接近，但 slowdown 更小；disjoint traffic 只比
single 高约 13%，slowdown 却放大约 2-3 倍。因此 aggregate PCIe GB/s 不是充分解释变量。

在 `.gpu` release 上执行：

```text
direction = D2H-only, H2D-only, full relay
active GPU count = 1, 2, 4
target duty = 25%, 50%, 75%, 100%
relay payload = 16M, 64M, 255M
collective = AG 64M；确认后补 RS/AR 代表点
repetitions = 7
```

需要两类扫描：

1. 固定每 edge throughput，改变 active GPU count，观察局部覆盖和 Ring 传播；
2. 固定 aggregate traffic，改变 edge 数/每 edge duty，区分“总带宽”与“覆盖 GPU 数”。

每个 case 输出 per-GPU/per-edge traffic、D2H/H2D active fraction、NCCL bracket slowdown 和
tail latency。若固定 aggregate traffic 后多 GPU 仍显著更差，证据优先支持 per-GPU local
CE/memory pressure 经 Ring 传播，而不是 host/PCIe 总带宽饱和。

### 19.6 Stage 13：用新 tracer 闭合 critical path

只有 Stage 9 验收通过后才执行。对 matched sys/gpu 的 AG/AR/RS 64M disjoint 和 four-ring：

1. 逐事件比较 fence `.sys/.gpu` 的 clean、D2H-active、H2D-active 和 concurrent 分布；
2. 对 wait-recv/wait-send 比较 duration 与 spinLoadCount，并关联上游 peer 的 phase；
3. store 若继续保持亚微秒且无同向放大，从候选主因中降级；
4. copy 只计算同 rank/channel 的 phase delta，不对并行 channel 求和；
5. 以 Inspector 找到的最慢 operation/rank/channel 为入口，只沿其串行依赖链估算增量。

数量级闭合使用：

```text
release collective delta
  ~= critical serialized fence delta
   + critical wait delta
   + copy/work delta
   + residual
```

允许区间由重复误差和 trace overhead 给出，不能要求虚假的精确相等。只有同时满足以下条件，
才能把 system-scope fence 写成已量化贡献者：

- matched release scope interaction 显著为正；
- device-only trace 中 `.sys` per-event delta 显著大于 `.gpu`；
- 增量发生在 Inspector/phase alignment 指出的关键 rank/channel 和不利 CE phase；
- 估算数量级不超过 release collective delta，且能够解释其稳定非零部分。

### 19.7 Stage 14：SHM 辅助分支

SHM formal 已显示：P2P NCCL aggressor 使 victim 下降约 62%，SHM aggressor 约 70%，SHM
相对 P2P 的额外 paired slowdown 约 7.8 pp；显式 CE relay 仅约 13%。因此 SHM 的主体是
第二套长驻 NCCL kernel/context 竞争，不是 CE 方案的直接替代。

若继续研究 SHM，只执行独立的 2x2：

```text
victim backend = sys, gpu
aggressor backend = sys, gpu
aggressor transport = P2P, SHM
```

victim 和 aggressor 是独立进程，必须分别设置各自 `LD_LIBRARY_PATH`，不能继续让一次实验
同时、不可区分地修改两侧 scope。保存 victim throughput、aggressor collective useful
throughput、kernel residency/overlap 和 transport 审计。该分支优先级低于 CE phase/residual
实验，不阻塞 CE 双路可行性判断。

### 19.8 Stage 15：完整 AG 旁路的进入门限

single `0->1` 的正 proxy gain 只证明一条独立 CE edge 可能有容量余量，不能证明四卡
AllGather 可由它加速。四卡 AG 的 CE lane 必须覆盖所有 rank 的数据传播；朴素 ring 至少需要
四条边和多个 step，而当前 four-ring proxy 对 AG 为负。因此真实实现前先做 duty operating
point 搜索：

```text
four-ring relay target duty = 10%, 25%, 50%, 75%, 100%
relay chunk = 4M, 16M, 64M, 255M
victim = AG 64M, 256M
backend = official-sys, experiment-gpu
repetitions = 7
```

每点使用 overlap-only 数据计算：

```text
capacity proxy = NCCL concurrent payload rate + relay useful rate
aggregation efficiency
predicted partition makespan
```

但 capacity proxy 仍是上界，因为 background relay 只重复单 hop，不包含完整 AG 的 `N-1`
step、chunk ownership 和 completion。进入四卡真实 AG 必须同时满足：

1. 至少一个 four-rank-complete relay topology/duty 点的 proxy gain 95% CI 下界大于 0；
2. 依据完整 AG step/bytes 建模后的 predicted makespan 优于 NCCL-only 至少 3%；
3. sys 和允许范围内的 gpu backend 都分别报告，不用 `.gpu` 掩盖 official baseline；
4. 新 tracer 已通过 observer-effect gate，或真实实现明确不依赖 tracer 结论；
5. `.gpu` correctness 仅限当前单机 NVLink path 的实验消融，不作为通用 transport 修复。

如果所有 four-rank-complete 点都不满足门限，则停止四卡性能实现；可以另做两卡双向
NVLink+Host Relay 的功能原型验证数据分片和 completion，但不得将两卡结果外推为四卡 AG
加速。

### 19.9 新的执行顺序和停止条件

```text
Stage 9  device-only tracer + observer-effect gate
  -> Stage 10 matched sys/gpu release matrix
  -> Stage 11 CE phase / collective tail alignment
  -> Stage 12 direction / active-GPU / duty residual matrix
  -> Stage 13 device-only primitive critical-path closure
  -> Stage 15 four-rank-complete AG operating-point gate
  -> 通过后才设计真实双路 AG

Stage 14 SHM 2x2 为低优先级旁支，可独立执行，不阻塞主线
```

出现以下任一情况立即停止对应方向：

- device-only tracer 经过两轮降采样仍无法满足 overhead/slowdown gate：放弃定量 trace，只保留
  release+Inspector+phase 证据；
- matched scope interaction 的 CI 包含 0：该 case 不宣称 fence 贡献；
- `.gpu` correctness/stress 出错：停止把 `.gpu` 作为性能候选，仅保留 `.sys` 正式路径；
- four-rank-complete AG operating point 的 gain 下界均不大于 0：停止真实四卡加速实现；
- profiler 改变 relay direction/duty 或 NCCL slowdown 超过预注册阈值：trace 只用于定性时序。

### 19.10 2026-09-01 后续实验重复策略

从本条记录之后启动的新实验批次，默认使用 **2 对交错 paired repetitions**，即每对包含
一个 release case 和一个 treatment case；每个 case 仍完整执行
`clean-before -> concurrent -> clean-after`，并保留返回码、校验、窗口、时钟和 relay
telemetry。已经启动的 7 对批次继续完成，不中途缩短或丢弃。

2 对结果仅用于快速筛选和下一轮决策，报告逐对差值、均值、范围和探索性 95% CI；由于样本数
少于原 formal 的 7 次，不把单个 2 对批次的 CI 当作最终因果证明。只有在结果接近 2% 停止
阈值、方向相互矛盾、clean-after 未恢复或出现 correctness/telemetry 异常时，才启动下一
个独立的 2 对批次，并在报告中把批次分开，禁止把重复批次选择性合并来制造显著性。

后续每轮循环固定为：

```text
读取上一轮结果 -> 写入本节的下一轮设置 -> 运行 2 对交错批次
  -> 核验完整 case -> 记录逐对/聚合结果 -> 判断是否达到 <=2% -> 决定下一轮
```

该重复策略只改变后续实验的时间预算，不改变 `#wrong=0`、窗口唯一、telemetry 完整、
无 CUDA/NCCL/Xid 错误和 clean-after 恢复等硬性安全条件；任一硬性条件失败仍立即停止
性能结论。

### 19.11 已启动批次设置：copy v7/v8 直接配对控制

copy v8 在 release 配对中仍出现负偏差，而 clean-before/after 也同步偏低，不能仅凭
release 对比判断新增缓存 gate 是否改善了 copy hook。因此下一批只做实现间直接控制：

```text
left/right build = copy-v7、copy-v8
trace mode       = runtime-disabled
collective       = AG
victim size      = 64M
background       = disjoint 0:1,2:3 full relay
victim           = -n 5000 -w 100
relay            = 255M, warmup 20, 100ms telemetry
repetitions      = 3 paired blocks, order alternating (在 2 对规则生效前已启动)
```

每个 block 仍分别运行两个独立进程级 case，并各自保存 clean-before、concurrent、
clean-after。主指标为 v8 相对 v7 的 treatment busbw，同时报告 clean 两端差值、逐对
差值、探索性 95% CI、wrong、窗口和 telemetry。若 v8 没有相对 v7 的改善，则撤销该
假设并以 v7 为 copy tracer 基准；若 v8 改善，则只保留能够被直接配对数据支持的改动，
不把 release 批次间差异解释为 hook 优化收益。

### 19.12 下一轮设置：Stage 10 AG four-ring matched scope pilot

copy v7/v8 的直接配对未显示 v8 有可确认改善，且 copy runtime-disabled 的两轮 observer
结果仍未满足“95% CI 包含 0”的严格条件；因此不再为 copy 继续增加观测器改动。wait、
fence、store、post 仍保留 device-only 定量结果，copy 只用于 release/Inspector/phase 的
定性辅助。

Stage 10 先用最能放大 scope interaction 的代表点验证 matched 流程：

```text
backend          = experiment-sys-release, experiment-gpu-release
collective       = AG
victim size      = 64M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks, sys/gpu order alternating
```

每个 backend 的每个 case 均保存 clean-before、concurrent、clean-after、精确 overlap
telemetry、GPU 状态、返回码和校验。计算 before-only 与 bracket slowdown，并以同一 paired
block 的 `slowdown_sys - slowdown_gpu` 作为 scope interaction；2 对只作快速筛选，不把
探索性 CI 写成最终显著性证明。若 ring interaction 方向稳定，再补 AR/RS ring 代表点；若
interaction 接近 0，则直接转向 Stage 12 的 direction/active-GPU/duty residual 搜索。

### 19.13 下一轮设置：Stage 12 four-ring duty screen（`.sys`）

Stage 10 pilot 显示 full-duty four-ring 下 `.sys` 的 AG bracket slowdown 约为
`11%-17%`，`.gpu` 仍约为 `9%`，scope 消融不足以达到目标。因此先在实际 `.sys`
backend 上降低 CE relay active duty，寻找“干扰不超过 2% 且仍有可测 relay useful rate”
的 operating point：

```text
backend          = experiment-sys-release
collective       = AG
victim size      = 64M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.25, 0.50, 0.75
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per duty, order alternating
```

每个 duty 点独立建结果目录；每个 case 使用 `--relayDutyCycle`，并检查 relay overlap
窗口内的 useful/traffic rate、active duty、NCCL before-only/bracket slowdown、clean-after
恢复和全部正确性条件。若某点 bracket slowdown 不超过 2% 且 relay useful rate 非零，
再用同 duty 做 `.gpu` scope 对照和 aggregate/makespan 估算；若三个点都超过 2%，下一轮
转向 active-GPU count 与方向拆分，而不继续盲目降低 duty。

### 19.14 下一轮设置：Stage 12 four-ring duty bracket

上一轮 `.sys` 结果为：duty `0.25/0.50/0.75` 的 bracket slowdown 分别约
`3.04%/7.54%/10.92%`，三个点的 relay overlap useful rate 均非零。下一批围绕目标
阈值向下搜索：

```text
backend          = experiment-sys-release
collective       = AG
victim size      = 64M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.10, 0.15, 0.20
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per duty, fresh case directories
```

每个点报告逐对 before-only/bracket slowdown、relay overlap useful/traffic rate 和实际
active duty。若某点达到 `<=2%` 且 relay useful rate 保持非零，立即把该点作为候选
operating point，进入 `.gpu` matched 对照和完整 AG makespan/proxy 检查；若仍未达到，
转向 active-GPU count/方向拆分，避免只靠降低 duty 获得没有实际吞吐价值的结果。

### 19.15 下一轮设置：duty 0.15 candidate cross-collective validation

four-ring、255M relay 的 `.sys` AG duty bracket 中，duty `0.15` 的两对平均 bracket
slowdown 约为 `1.66%`，overlap useful rate 约为 `2.48 GB/s`，是当前干扰与旁路流量的
折中候选。下一批验证该候选是否只对 AG 有效，并同时校准 scope residual：

```text
backend          = experiment-sys-release（AR、RS）以及 experiment-gpu-release（AG）
collective       = AR 64M、RS 64M、AG 64M
victim size      = 64M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.15
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per collective/backend, fresh directories
```

每个点报告 before-only/bracket slowdown、NCCL algbw/busbw、relay overlap useful/traffic
rate 和完整安全检查。AG 的 sys/gpu 顺序交错；AR/RS 先用 `.sys` 判断 candidate 是否跨
collective 成立。若至少 AG/AR/RS 中两个 collective 均达到 `<=2%` 且 relay useful rate
非零，则以 duty 0.15 进入 Stage 15 的四卡 AG capacity/makespan 审查；若只有 AG 达标，
下一轮改做 direction/active-GPU residual，而不宣称通用双路收益。
### 19.16 下一轮设置：RS duty threshold search

duty `0.15` 下 AG/AR 已分别达到 `1.43%/1.19%`，但 RS 仍为 `2.64%`；因此只对 RS
沿 duty 轴向下做边界搜索：

```text
backend          = experiment-sys-release
collective       = RS
victim size      = 64M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.10, 0.125
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per duty, fresh directories
```

每个点继续使用 bracket baseline，报告 overlap useful/traffic rate 和完整安全检查。
RS duty `0.15` 的约 `2.64%` 已被接受为当前范围内的可用结果，不再阻塞主线；duty
`0.125/0.10` 仅用于记录 RS 的趋势和可选优化空间，不用 AG/AR 结果替代 RS 结论。

主线下一步转向 Stage 15 的四卡 AG capacity/makespan 审查：AG duty `0.15` 作为默认
候选，AG duty `0.10` 作为低干扰对照；RS 的进一步降低 duty 作为非阻塞旁支。

### 19.17 下一轮设置：Stage 15 四卡 AG capacity/makespan size audit

四卡 full-ring 的 AG 64M 结果已经给出一个低干扰 operating point，但 64M 仍可能受
启动和短消息效应影响，不能直接代表完整 AG 的分片 makespan。因此下一批固定四条 relay
边，改用 256M AG 做容量审查；duty `0.15` 为默认候选，`0.10` 为更低干扰的对照。

此处 duty 的语义按 `host_relay` 的实际实现记录：每次迭代先完成一组 D2H→H2D copy，
再按

```text
idle_time = active_copy_time * (1 / duty - 1)
```

进行 host-side sleep。因此 duty 只改变 relay 发射 cadence/平均 PCIe 压力，不改变单次
payload，也不代表真实 AG 已经完成数据分片；relay useful rate 与 predicted makespan
仍然只能作为 feasibility proxy。

```text
backend          = experiment-sys-release
collective       = AG
victim size      = 256M
background       = four-ring full relay, edges 0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.15, 0.10
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per duty, fresh directories, alternating order
```

每个 duty 点报告逐对 clean-before/after bracket slowdown、NCCL algbw/busbw、relay
useful/traffic rate、实际 active duty、aggregate capacity proxy 和依据 AG payload/step
字节计算的 predicted partition makespan。必须保持 `#wrong=0`、窗口唯一、relay 返回码为
零、telemetry 覆盖 overlap 且 clean-after 恢复。若 256M 的 duty `0.15` 仍满足 2% 约束并
且 proxy/makespan 显示容量余量，再补同一点的 `.gpu` scope 对照；若容量余量不足，则转向
更小 relay chunk 或 active-GPU/方向拆分，不进入真实四卡 AG 分流实现。

### 19.18 下一轮设置：快速根因隔离矩阵（方向 × 覆盖 × 双向性）

用户要求优先快速确定根因并解决，因此从下一批开始暂停继续扩展 size/duty 网格，先做最小
因果矩阵。前面的 duty 曲线只证明“压力越大，干扰通常越强”，本批要把方向和覆盖 GPU
数固定/拆开，直接区分方向性、per-GPU 局部资源和双向 transition 的贡献。

当前 256M AG 容量审查完成后，使用更敏感但仍可控的 AG 64M、duty `0.50`，保持每个
active edge 的 payload 和 CPU placement 相同：

```text
backend          = experiment-sys-release
collective       = AG 64M
relay size       = 255M
relay duty       = 0.50
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per point

A: mode=d2h,   edges=0:0,1:1,2:2,3:3   # 四卡 D2H-only
B: mode=h2d,   edges=0:0,1:1,2:2,3:3   # 四卡 H2D-only
C: mode=relay, edges=0:1,1:2,2:3,3:0   # 四卡双向 full relay
```

P0 runner 增加显式 `--relayMode/--relayEdges`，以便三个点使用同一套窗口、校验、GPU
状态和 telemetry 记录而只改变 relay 机制。首轮判别规则如下：

1. A 与 B 的 slowdown 明显不对称：优先锁定 D2H source-read 或 H2D destination-write
   方向，随后只在该方向检查 CE channel/内存路径，不再先改 NCCL primitive。
2. A、B 均明显而接近，且 C 不超过单向控制太多：优先考虑 D2H/H2D 共同的 GPU
   memory/L2/CE 资源或 PCIe/system transaction 竞争。
3. C 显著高于 A/B：优先检查 D2H→H2D transition gap、双向 CE 仲裁或 relay 的两阶段
   串行化，而不是把总 traffic 作为唯一解释。

若 A/B 方向差异不大，再补同 duty 的 D2H `0:0` 单卡与 `0:0,1:1,2:2,3:3` 四卡两对
对照：固定每 edge 参数观察 slowdown 是否按 active GPU 数增长。按 GPU 数增长支持本地
CE/HBM/L2 压力及 Ring 传播；按 aggregate traffic 而非 GPU 数增长才支持 host/PCIe
总带宽解释。只有方向/覆盖矩阵完成后，才用严格匹配的 `.sys/.gpu` 两对对照确认
system-scope ordering 是否是可修复的放大项；不再用 observer-effect tracer 作为根因证据。

执行结果（2 对探索性批次，全部 `#wrong=0`、窗口唯一、telemetry 覆盖）为：

| 点 | bracket slowdown 均值 | useful / traffic 均值 |
| --- | ---: | ---: |
| 四卡 D2H-only | 5.14% | 8.85 / 8.85 GB/s |
| 四卡 H2D-only | 10.58% | 8.21 / 8.21 GB/s |
| 四卡 full relay | 7.83% | 5.10 / 10.19 GB/s |

H2D-only 比 D2H-only 高约 5.44 个百分点；full relay 的 traffic 更高却没有超过 H2D-only，
因此当前优先假设不是单一 aggregate PCIe 带宽瓶颈，而是 H2D destination-write 路径及其
可能的 system-domain ordering/回压更敏感。

### 19.19 下一轮设置：H2D 主导方向的 `.sys/.gpu` 因果闭合

对上一个矩阵中最强的 H2D-only 点做严格 scope 对照，保持 relay 方向、四卡覆盖、duty、
payload、CPU placement、NCCL iteration 和 paired order 完全一致：

```text
backend          = experiment-sys-release, experiment-gpu-release
collective       = AG 64M
background       = H2D-only, edges=0:0,1:1,2:2,3:3
relay size       = 255M
relay duty       = 0.50
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per backend, alternating backend order
```

主指标为同一 relay 强度下 `slowdown_sys - slowdown_gpu`，同时保留两端 clean recovery、
H2D active duration/p99、每 GPU bytes、NCCL algbw/busbw 和所有硬性安全检查。判别规则：

1. `.gpu` 显著降低 H2D slowdown 并接近 2%：system-scope ordering 是可直接验证的主要
   放大项；随后只做该 scope 修改的 correctness/stress 回归，不继续扩大物理资源猜测。
2. `.sys` 与 `.gpu` 都接近 10%：scope 不是主因，优先做 H2D active-GPU 数（1 vs 4）
   对照，判断是 destination-local CE/HBM/L2 压力还是跨卡/host aggregate 效应。
3. `.gpu` 只消除一部分：将 scope interaction 记为可修复子项，剩余部分进入 H2D active-GPU
   数与 CE timing 的最小诊断，不再把全部干扰归给 fence。

本批加入 CE standalone 后的同配置结果为：D2H-only 的 NCCL/CE slowdown 均值约为
`5.42%/0.06%`，H2D-only 约为 `10.62%/-3.50%`，full relay 约为 `7.32%/-21.34%`。
负的 CE slowdown 表示 concurrent relay rate 高于 standalone rate，不是 CE 下降；但
full relay 的差值偏大，故 standalone 仍按探索性 paired 分母处理，不能把短批次的速度差
写成 CE 加速证明。NCCL 的方向性结论仍稳定：H2D-only 是最强敏感点。

### 19.21 下一轮设置：H2D 双侧目标下的 `.sys/.gpu` matched 对照

对 H2D-only 主导点同时验证 scope interaction 和双侧性能目标。每个 case 在 clean-before
之后先运行 5 秒同配置 CE standalone，再运行 concurrent NCCL，保证 CE slowdown 有明确
分母：

```text
backend          = experiment-sys-release, experiment-gpu-release
collective       = AG 64M
background       = H2D-only, mode=h2d, edges=0:0,1:1,2:2,3:3
relay size       = 255M
relay duty       = 0.50
relay standalone = 5s, same mode/edges/size/duty/CPU
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per backend, alternating backend order
```

验收同时计算 `NCCL slowdown` 与 `CE slowdown`；`.gpu` 若令 NCCL 从约 10% 降至 2% 内且
CE 仍不下降，则优先保留为候选修复方向；若 `.gpu` 仍约 10%，则停止把 fence scope 当主
因，转向 H2D active-GPU 数/CE timing。无论哪种结果，都必须保留 standalone 波动、逐对
数据和 clean-after 恢复，不能只用 NCCL 一侧的改善下结论。

执行结果为：`.sys`/`.gpu` 的 NCCL bracket slowdown 均值约为 `10.29%/5.98%`，scope
interaction 约 `4.31` 个百分点；`.gpu` 仍有约 `5.98%` residual。两个 backend 的 CE
slowdown 均值均为负，未见 CE 相对同配置 standalone 的下降。由此确认 system-scope
ordering 是放大项但不是唯一根因，下一步固定 `.gpu` 以隔离剩余的 H2D destination/CE/GPU
memory 资源效应。

### 19.22 下一轮设置：H2D active-GPU coverage 因果对照

在已去除 system-scope 放大项的 `.gpu` backend 上，比较同一 H2D payload/duty 下单卡与
四卡 destination 覆盖；每个点同时测同配置 CE standalone，避免只看 NCCL：

```text
backend          = experiment-gpu-release
collective       = AG 64M
relay mode       = h2d
relay size       = 255M
relay duty       = 0.50
point A          = edges=0:0
point B          = edges=0:0,1:1,2:2,3:3
relay standalone = 5s, same mode/edges/size/duty/CPU per point
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks per point
```

若四卡相对单卡的 residual 按覆盖 GPU 数明显放大，根因优先落在每 GPU destination CE/
HBM/L2 压力或其传播；若只按总 traffic 变化，则优先处理 host/PCIe aggregate；若两者接近，
则继续检查 H2D stream/CE channel 调度和 destination mapping。该点完成后再决定是保留
`.gpu` scope 修复、降低/重排 H2D relay cadence，还是需要两者组合；最终候选必须同时满足
NCCL 与 CE 相对各自单跑均不超过 2%。

执行结果为：`.gpu` 单卡 H2D 的 NCCL/CE slowdown 均值约 `1.66%/1.54%`，四卡约
`5.74%/-3.56%`。四卡覆盖相对单卡使 NCCL residual 增加约 `4.08` 个百分点，但 CE
相对 standalone 没有下降；这支持“目的 GPU 的 H2D/本地资源压力随覆盖数叠加并沿 NVLink
critical path 传播”，不支持把 host/PCIe aggregate 作为唯一根因。

### 19.23 下一轮设置：四卡 H2D duty shaping 的双侧回归

先验证不改 NCCL、只改变 CE relay cadence 是否足以解决官方 `.sys` 的四卡残余；使用
已经确认的四卡 H2D 主导方向，duty `0.10` 作为低干扰候选，和此前 duty `0.50` 的高
信号结果比较。每个 case 继续有同 duty CE standalone 分母：

```text
backend          = experiment-sys-release
collective       = AG 64M
background       = H2D-only, mode=h2d, edges=0:0,1:1,2:2,3:3
relay size       = 255M
relay duty       = 0.10
relay standalone = 5s, same mode/edges/size/duty/CPU
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks
```

若 `.sys` 的 NCCL slowdown 与 CE slowdown 均不超过 2%，该点成为无需 NCCL 代码改动的
可用 operating point；随后用 `.gpu` 同 duty 做 scope 交叉，判断是否还需要 scope 修复。
若 NCCL 仍超过 2% 而 CE 不降，说明只降平均 cadence 不足，下一轮改为四卡 edge phase
stagger/stream 调度；若 CE 也下降，则先修 relay 调度或 host/PCIe 路径，而不是修改 NCCL。

执行结果：四卡 H2D-only、官方 `.sys`、duty `0.10` 的两对均通过硬性检查，NCCL
bracket slowdown 均值约 `1.40%`，CE slowdown 均值约 `-44.18%`。因此该点同时满足
双侧 2% 目标（负值代表 concurrent CE useful rate 未下降），并确认只降低 relay cadence
即可把 `.sys` residual 压到目标内；后续仍需用实际 full relay 做最终验证。

### 19.24 下一轮设置：实际四卡 full relay 双侧最终候选验证

H2D-only 已验证 cadence shaping 的下界；最终候选改回真实 D2H→H2D host relay。沿用
此前 AG 64M `.sys` duty `0.15` 的 NCCL 候选点，并为每个 case 加入同配置 CE standalone：

```text
backend          = experiment-sys-release
collective       = AG 64M
background       = full relay, mode=relay, edges=0:1,1:2,2:3,3:0
relay size       = 255M
relay duty       = 0.15
relay standalone = 5s, same mode/edges/size/duty/CPU
victim           = -n 5000 -w 100
relay            = warmup 20, telemetry 100ms, background-first
repetitions      = 2 paired blocks
```

每对计算 NCCL bracket slowdown 和 CE useful-rate slowdown；只有两者均不超过 2%、
`#wrong=0`、relay/NCCL 返回码为零、窗口唯一、telemetry 覆盖且 clean-after 恢复时，才把
`duty=0.15` 记录为实际 full-relay operating point。若 full relay 不满足，则用 duty `0.10`
作为保守候选，并把 H2D 目的端压力 + system-scope 放大记录为根因，而不是继续盲目增加
tracer 或 primitive 观测。

5 秒 standalone 的初步结果中，full relay 每条边有效迭代数较少，CE 分母的短窗口波动
明显大于 NCCL 侧。因此在宣布双侧闭环前做一次相同配置、仅延长 standalone 时长的确认：

```text
same full-relay candidate as above
relay standalone = 15s instead of 5s
repetitions      = 2 paired blocks
```

该确认只用于稳态 CE rate 的分母质量，不改变 duty、payload、拓扑或 NCCL workload；若
NCCL 和 CE 两个 slowdown 都仍不超过 2%，即可结束性能干扰主线并转入结果归档/修复说明。

### 19.25 当前双侧目标闭环与根因结论

15 秒 standalone 确认的实际四卡 full relay 结果为：NCCL bracket slowdown 两对均值
`1.52%`（pair 为 `1.60%/1.44%`），CE useful-rate slowdown 均值 `-57.39%`
（pair 为 `-97.32%/-17.47%`）。所有 case 均满足 `#wrong=0`、NCCL/relay 返回码为零、
marker 唯一、overlap telemetry 覆盖和 clean-after 恢复。因此在当前 V100、AG64M、255M
relay、固定 placement 的实验范围内，NCCL 与 CE 相对各自单跑均未出现超过 2% 的下降。

根因链闭合到以下工程层级：H2D destination-write 对 NCCL NVLink critical path 比 D2H
source-read 更敏感；覆盖从单 GPU 扩到四 GPU 后 residual 明显放大，说明目的 GPU 的
CE/GPU-memory/L2 资源压力会沿 collective 传播；`.sys`→`.gpu` 又可移除约 4.31 个
百分点，说明 system-scope ordering 是额外放大项。当前解决方案是对 full relay 使用
duty `0.15` 做 CE cadence shaping，避免 burst/ordering 压力超过 NCCL critical path；
该方案在实际 full relay 上完成双侧验收，不依赖 observer-effect tracer，也不需要把
不安全的 `.gpu` scope 改动作为默认修复。

主线在此 operating point 达到停止条件；后续若要扩展到其他 collective/size，只能作为
新的独立 2 对批次，并继续使用 CE standalone 分母，不能把本点外推为所有 workload 均
无干扰。详细根因和双侧指标见
`doc/results/nccl-hybrid-path/20260901-root-cause-dual-side-summary.md`。

### 19.20 双侧性能目标修正与 CE standalone 记录规范

本实验的最终目标不是只让 NCCL 在并发时保持稳定，而是同时满足两条相对于各自单跑的
性能约束：

```text
NCCL slowdown = 1 - NCCL_concurrent / NCCL_clean
CE slowdown   = 1 - CE_concurrent / CE_standalone

目标：NCCL slowdown <= 2%
      CE slowdown   <= 2%（理想情况下同样必须满足）
```

其中 CE standalone 必须使用与 concurrent 完全相同的 `mode/edges/relay size/duty/CPU`
配置；`duty` 的人为 sleep 不能算作干扰。`CE_concurrent` 使用 NCCL measured window
内的 relay useful rate，`CE_standalone` 使用同一 case 中先运行的 relay-only 固定时长
基线；relay 模式下两者都使用 useful rate，traffic 只作为附加报告，避免单双向因子不一致。

P0 runner 新增 `--relayStandaloneSec` 后，正式双侧验收 case 必须开启该选项，并在 case
JSON 中保存 standalone relay JSON、返回码、GPU 状态和
`relaySlowdownPct = 1 - useful_concurrent / useful_standalone`。若缺失 standalone、
overlap telemetry 未覆盖窗口、任一返回码/校验/marker 不满足，case 只能用于定性诊断，
不能作为双侧 2% 证据。前面没有 CE standalone 分母的结果保留为方向/机制筛选证据，
不追溯改写为 CE 已达标。

### 19.26 标准满负载 D2H 与 NCCL AllGather 尺寸 A/B

针对需要区分“纯 D2H 背景”与“D2H+H2D relay”的要求，补充一组最直接的单次 A/B：
先运行干净的 4 卡 NCCL AllGather，再启动一个标准 D2H 进程并运行完全相同的
`nccl-tests` 命令；NCCL 结束后立即终止 D2H 进程。本批不使用 relay、不使用
`duty<1`，也不把 D2H 流量拆成多个降载子任务。

```text
victim          = all_gather_perf, CUDA_VISIBLE_DEVICES=0,1,2,3
victim sizes    = 16M,32M,64M,128M,256M,512M,1G (-f 2)
victim iters    = -n 30 -w 10
background      = host_copy_background, GPU0 -> Host, --size=255M
background duty = --dutyCycle=1.0
background loop = cudaMemcpyAsync(D2H) -> cudaStreamSynchronize -> repeat
measurement     = clean once, D2H-concurrent once
```

该批是直接 A/B 探索结果，不提供多次 repetition 的置信区间。两次有效 NCCL 运行均
报告 `nccl-library=23102`，返回码为 0，所有数值行 `#wrong=0`。原始输出和逐 size
结果保存在 `doc/results/nccl-hybrid-path/standard-d2h-ag-size-sweep-20260901/`。
