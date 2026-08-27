# NCCL 集合通信与 D2H 背景流量干扰分析实验方案

## 1. 目标与结论边界

本文设计一套基于 `nccl-tests` 的分层实验，用于分析同机多 GPU 场景下持续 D2H
背景流量对 NCCL 集合通信性能的影响。研究对象限定为以下三种常见 collective：

- AllGather（AG）；
- AllReduce（AR）；
- ReduceScatter（RS）。

现有 CUDA 层实验已经证明，在当前四卡 Tesla V100-SXM2-32GB 环境中，D2H
背景流量能够引发 GPU-local、source-side 的数据移动干扰；但 `cudaMemcpyPeerAsync`
和 NCCL kernel 的执行机制不同，因此 CUDA 层结论不能直接外推到 NCCL。

参考文档
`/home/songxb26/HyLink/nccl-tests/D2H背景流量导致NCCL性能下降-根因分析.md`
提出了一个重要候选机制：NCCL Simple primitive 的 `postPeer` 路径中包含
system-scope 内存栅栏，D2H DMA 可能显著增加该栅栏的完成延迟。该文档的测试不完全
基于当前环境，因此本方案不把这一判断当作已知结论，而是将其作为待验证假设。

本实验需要回答：

1. AG、AR、RS 在当前 V100 环境中是否都受到 D2H 干扰，干扰量是否依赖消息大小、
   运行时长和物理 GPU 放置；
2. 性能下降发生在 host/API、NCCL kernel、某些 rank/channel，还是背景流量启动与
   清理过程；
3. `waitPeer`、数据 copy/reduce、system-scope fence、system-scope step store
   分别贡献多少额外延迟；
4. AG、AR、RS 的差异是否与其 Ring+Simple primitive 序列和 send-bearing step
   数量相符；
5. 将 fence scope 从 `.sys` 降为 `.gpu` 能消除多少 D2H 额外开销，以及仍未消除的
   部分来自哪里。

本文允许得到“多因素共同导致下降”的结论。`.sys -> .gpu` patch 只是作用域消融，
不能单独证明 system-scope fence 是唯一根因。若公开工具和安全打点只能把竞争定位到
GPU-local synchronization/data-movement path，也不应继续声称已确定某个未公开的
L2 slice、memory partition 或 Copy Engine 仲裁单元。

## 2. 当前环境和已有证据

当前机器的关键条件为：

| 项目 | 当前值 |
| --- | --- |
| GPU | 4 x Tesla V100-SXM2-32GB |
| GPU 拓扑 | 任意 GPU 对均为 `NV2` |
| Host NUMA | GPU0-3 均位于 NUMA node 0 |
| Driver / CUDA | 580.178.04 / CUDA 12.6.85 |
| PCIe | Gen3 x16 |
| GPU 架构 | Volta，`sm_70` |

CUDA 层实验已经支持以下判断：

- allpairs D2D 与 D2H 并发时存在明显下降，ring 和 H2D 是负对照；
- D2H 只注入 GPU k 时，主要是 source=k 的 D2D stream 变慢；
- 20-repeat 与 500-repeat 的下降幅度不同，说明短批次和 steady-state 必须分开；
- Nsight Systems 能确认 P2P memcpy 与 D2H memcpy 的真实重叠；
- 现有证据尚不能把 CUDA 层现象细分为 HBM、L2、CE 或内部 fabric。

NCCL 层实验复用这些测量纪律，但不复用 CUDA 层的根因结论。

## 3. 总体策略：五层证据链

采用假设驱动、逐层收敛的方式，不在第一轮执行 collective、size、protocol、algorithm、
channel 和背景强度的全笛卡尔积。

```text
L0 版本与测量校准
  -> L1 nccl-tests 宏观性能
    -> L2 NCCL Inspector rank/channel 观测
      -> L3 NCCL primitive 细粒度打点与必要采样
        -> L4 .sys/.gpu 作用域消融
```

各层的证据边界为：

| 层次 | 能回答的问题 | 不能单独回答的问题 |
| --- | --- | --- |
| L0 | 库、算法、协议、channel 和插件开销是否一致 | D2H 的具体慢点 |
| L1 | collective 是否变慢、慢多少、能否恢复 | kernel 内部慢在哪里 |
| L2 | 哪次 collective、哪个 rank/channel 的 kernel 区间变长 | 是否卡在某条 fence 指令 |
| L3 | wait/copy/fence/store 的延迟分布和次数如何变化 | 修改 fence scope 是否安全通用 |
| L4 | 缩小 fence scope 能消除多少额外开销 | 剩余开销是否不存在 |

Inspector、Nsight Systems、Nsight Compute 和细粒度 instrumented NCCL 不在同一次
正式运行中叠加使用。每种观测手段都必须有独立的无背景对照和开销校准。

## 4. 双 NCCL 源码树与构建管理

### 4.1 源码树角色

```text
nccl-tests（唯一 benchmark 入口）
  |
  +-- /home/songxb26/HyLink/nccl
  |     官方 NCCL 2.31.2、正常 .sys 实现、性能真值
  |
  +-- third_party/nccl
        NCCL 2.31.2 基线、Inspector、细粒度打点和实验 patch
```

`/home/songxb26/HyLink/nccl` 不用于实验性源码修改。`third_party/nccl` 在开始修改前
必须先建立与官方源码树等价的 `sys-release` 基线。

当前 `/home/songxb26/HyLink/nccl/src/device/prims_simple.h` 存在未提交修改。更新源码前
必须先保存以下信息：

```bash
git -C /home/songxb26/HyLink/nccl status --short
git -C /home/songxb26/HyLink/nccl diff -- src/device/prims_simple.h
git -C /home/songxb26/HyLink/nccl describe --tags --always --dirty
```

不得直接覆盖该修改。保存 diff 及其 SHA256 后，再将官方源码树更新并固定到
`v2.31.2-1`。官方树最终必须为 clean worktree，并确认 `prims_simple.h` 使用
`fence_acq_rel_sys()`。

### 4.2 V100 专用构建

两套 NCCL 均使用：

```bash
NVCC_GENCODE='-gencode=arch=compute_70,code=sm_70'
```

这样直接生成 V100 的 `sm_70` SASS，避免运行时 PTX JIT 和多架构构建差异，也缩短
instrumented NCCL 的反复构建时间。该产物必须标记为 `v100-sm70-only`，不能用于
其他 GPU 架构。

除了检查源码，还要用 `cuobjdump` 或 `nvdisasm` 检查生成的 cubin。Volta 上最终
SASS 的助记符可能表现为 `MEMBAR.SYS` 一类形式，因此是否为 system scope 以实际
反汇编为准，不能只依据 C++ 函数名。

### 4.3 每次运行的构建身份

每个结果目录必须保存：

```text
NCCL source path
NCCL git commit / describe
NCCL dirty status
applied patch
patch SHA256
libnccl.so SHA256
Inspector plugin SHA256
NVCC_GENCODE
CUDA / driver / GPU
nccl-tests commit
ldd output
NCCL runtime version
```

不得仅依赖 `LD_LIBRARY_PATH` 的期望值。Runner 启动测试前必须检查 `ldd`，并在日志中
记录实际加载的 `libnccl.so.2` 绝对路径。不同 NCCL build 的结果不能写入同一 case
目录。

## 5. D2H 背景流量维护方式

背景流量的代码、脚本、配置和结果入口统一维护在 `nccl-tests` 内。建议结构为：

```text
nccl-tests/
  interference/
    d2h_background/
      d2h_background.cu
      README.md
    scripts/
      run_case.sh
      run_matrix.sh
      switch_nccl.sh
      capture_env.sh
    analysis/
      parse_nccl_tests.py
      parse_inspector.py
      summarize_instrumentation.py
    configs/
      v100-ring-simple.yaml
    results/
      <date>-<run-id>/
```

### 5.1 背景程序接口

D2H 工具至少支持：

```text
--devices 0,1,2,3
--bytes 267386880
--direction d2h|h2d
--duration-sec N
--queue-depth N
--sync-every N
--duty-cycle PERCENT
--ready-file PATH
--stop-file PATH
--json-output PATH
```

每张参与 GPU 使用独立 device buffer、pinned host buffer、stream 和 worker，并输出：

```text
GPU id
实际传输字节数
operation count
elapsed time
per-GPU GB/s
aggregate GB/s
CUDA error
退出原因
```

第一轮保持与现有 CUDA 实验相同的 255 MiB payload 和每次 memcpy 后同步的 cadence，
以便跨层比较。后续强度扫描再改变 queue depth、同步频率和 duty cycle。

### 5.2 生命周期协议

单个 treatment case 的顺序为：

```text
启动 background
-> 等待所有目标 GPU ready
-> 再等待 1-2 秒进入稳态
-> 运行 nccl-tests
-> 创建 stop-file
-> 等待 background 正常退出
-> 检查 background JSON 和 exit code
-> 运行 clean-after
```

禁止仅使用固定 `sleep` 判断背景是否就绪，也禁止使用宽泛的 `pkill -f` 清理进程。
Runner 应保存准确 PID，并在异常路径中先发送正常停止信号，超时后再终止对应 PID。

## 6. Ring+Simple 源码假设

第一轮固定：

```bash
NCCL_ALGO=RING
NCCL_PROTO=SIMPLE
CUDA_VISIBLE_DEVICES=0,1,2,3
```

当前 NCCL 2.31.2 源码中：

```text
NCCL_STEPS = 8
ALLGATHER_SLICESTEPS = NCCL_STEPS / 4 = 2
ALLGATHER_CHUNKSTEPS = NCCL_STEPS / 2 = 4
ALLREDUCE_SLICESTEPS = 2
ALLREDUCE_CHUNKSTEPS = 4
REDUCESCATTER_SLICESTEPS = 2
REDUCESCATTER_CHUNKSTEPS = 4
```

因此 AG、AR、RS 的 Ring+Simple 主路径均使用：

```text
SlicePerChunk = 2
StepPerSlice = 2
```

四卡 Ring 每个 chunk 的逻辑 primitive 序列为：

| Collective | 每个 chunk 的 primitive 序列 | 含 Send 的逻辑阶段数 |
| --- | --- | ---: |
| AG | `Send -> 2 x RecvCopySend -> Recv` | 3 |
| RS | `Send -> 2 x RecvReduceSend -> RecvReduceCopy` | 3 |
| AR | RS 段：`Send -> 2 x RecvReduceSend -> RecvReduceCopySend`；AG 段：`2 x RecvCopySend -> Recv` | 6 |

Simple primitive 的 system-scope fence 由 `RolePostSend` 线程执行，并且仅在存在有效
数据写入或启用 connection FIFO 时触发。由此得到待验证预测：

- 四卡 Ring 中，AR 每个 chunk 的 send-bearing primitive 数约为 AG/RS 的两倍；
- 若 system-scope fence 是主要贡献者，控制其他因素后 AR 可能表现出更高的同步敏感性；
- 实际 fence 次数仍受 chunk 数、channel 数、tail slice、direct mode、registration
  和 connection 配置影响，必须用打点计数验证，不能由源码循环次数直接替代。

AR 还包含 reduction 数据路径，AG/RS 的数据量定义和读写组合也不同，因此不能只按
fence 数预测最终 slowdown。

## 7. 阶段 A：版本、路径与正确性校准

### A1. 官方 NCCL 2.31.2

验收以下项目：

1. 官方源码树为 clean；
2. `prims_simple.h` 为正常 `.sys` 实现；
3. cubin 包含预期 system-scope barrier；
4. 三个 `nccl-tests` 二进制实际链接到官方库；
5. AG、AR、RS 的 correctness validation 全部通过；
6. NCCL 日志确认 GPU 间使用预期 P2P/NVLink transport。

### A2. third_party sys-release 等价性

使用同一编译器、同一 `sm_70` 参数构建 third_party 的未打点 `.sys` release。对 AG、
AR、RS 各选择 64 MiB 和 256 MiB，分别执行至少 5 次 clean 对照。

要求：

- algorithm、protocol、channel 和 thread 配置一致；
- correctness 一致；
- 性能差异位于各自基线噪声范围内；
- 若不等价，必须先解释源码、构建或 Inspector 集成差异，不能进入 patch A/B。

### A3. Inspector 自身开销

比较：

| NCCL build | Inspector | 用途 |
| --- | --- | --- |
| 官方 2.31.2 | OFF | 性能真值 |
| 官方 2.31.2 | ON | Inspector 开销 |
| third_party sys-release | OFF | 双源码树等价性 |
| third_party sys-release | ON | 后续宏观观测入口 |

建议 Inspector 配置：

```bash
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_ENABLE_P2P=0
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=-1
export NCCL_INSPECTOR_DUMP_VERBOSE=1
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=1
export NCCL_INSPECTOR_DUMP_DIR=<case-specific-path>
```

仅在 communicator teardown/finalization 时 dump，避免周期性写文件进入测量窗口。
运行时使用：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL,PROFILE
```

记录 algorithm、protocol、channel count、thread count、transport、buffer registration
和插件初始化信息。若 Inspector ON 相对 OFF 的开销超过约 2%，Inspector 仍可用于
定性和分布分析，但不能替代无插件吞吐真值。

## 8. 阶段 B：运行 regime 校准

当前 CUDA 层实验已经发现 short-burst 和 steady-state 可能进入不同性能状态。NCCL
正式矩阵前，先对 AG、AR、RS 各选择 64 MiB，扫描：

```text
warmup: 5, 20
measured iterations: 20, 100, 500
```

记录每个 cycle 的时间和 Inspector per-operation 时间序列，检查：

- 前若干 collective 是否系统性偏慢；
- D2H 是否只影响启动阶段或持续影响稳态；
- 增加 warmup 是否改变 slowdown；
- clean-after 是否立即恢复；
- 迭代数是否改变算法、协议或 channel 数。

最终冻结两个正式 regime：

1. short-burst：默认起点为 `-w 5 -n 20`；
2. steady-state：按 collective/size 选择迭代数，使 measured window 达到约 3-5 秒。

两种 regime 的结果分别统计，不合并成一个 slowdown。

## 9. 阶段 C：AG、AR、RS 宏观主矩阵

### C1. 消息大小

第一轮使用：

```text
4 MiB    同步/延迟占比高
64 MiB   参考报告和当前 V100 测试中的敏感区
256 MiB  中等大消息
1 GiB    吞吐主导区
```

每个结果同时记录 nccl-tests 输入参数、per-rank data size、aggregate data size、
`time`、`algbw` 和 `busbw`，避免不同 collective 的 size 语义混淆。

### C2. 主效应筛选

```text
3 collectives
x 4 sizes
x 2 regimes
x {clean-before, D2H-all, clean-after}
x 7 paired repetitions
```

每个 repetition 内使用随机化或 ABBA 顺序。核心指标为：

```text
paired_clean = mean(T_clean_before, T_clean_after)

slowdown = T_D2H / paired_clean - 1

recovery_error = T_clean_after / T_clean_before - 1
```

若 `clean-after` 超出 clean baseline 的预设置信区间，先检查背景进程、温度、时钟、
P-state、context 和缓存/队列状态。该 repetition 不直接用于根因归因。

### C3. GPU-local 和方向负对照

只对 C2 中 slowdown 明显的 size 扩展：

```text
D2H GPU0 only
D2H GPU0-3 all
H2D GPU0 only
H2D GPU0-3 all
```

再执行两卡物理隔离：

```text
NCCL GPU0,1 + D2H GPU0,1    同卡 treatment
NCCL GPU0,1 + D2H GPU2,3    远端负对照
```

判读：

- 同卡下降、远端不下降：支持 GPU-local 机制；
- GPU0-only D2H 引发特定 rank/channel 尾延迟：支持局部 straggler；
- D2H 明显、H2D 很弱：支持方向相关的 system/DMA 或源端路径；
- D2H/H2D 相同：不能继续使用 D2H 特有的 fence 解释；
- clean-after 不恢复：需要先解释残留状态。

### C4. 背景强度响应

主效应稳定后，对每种 collective 的一个敏感点扫描实际 D2H 压力：

```text
0%, 10%, 25%, 50%, 75%, 100% duty cycle
```

x 轴必须使用背景程序实际测得的 per-GPU D2H GB/s，而不是配置百分比。若 fence 或
post 延迟随 D2H GB/s 单调增长，支持资源压力响应；若主要随 operation rate 变化，
则还需区分 DMA in-flight 深度、提交频率和字节率。

## 10. 阶段 D：NCCL Inspector 宏观分析

Inspector 不覆盖完整主矩阵。每种 collective 从 C 阶段选择：

- 一个 slowdown 最大或同步最敏感的 size；
- 一个大消息、吞吐主导的 size。

每个点采集：

```text
clean
D2H GPU0 only
D2H GPU0-3 all
clean-after
```

### D1. Collective 维度

分析：

```text
coll_exec_time_us
coll_algobw_gbs
coll_busbw_gbs
coll_timing_source
operation sequence number
operation latency p50/p90/p99/max
```

只接受 `kernel_gpu` timing 进入正式比较。检查 D2H 是否影响所有 operation，还是只
影响启动阶段或少量尾部 operation。

### D2. Rank 维度

对 GPU0-only D2H，比较每个 rank 的 collective 和 kernel event：

- GPU0 对应 rank 是否首先变慢；
- 其他 rank 是否随后被同步依赖拖住；
- 最慢 rank 是否决定 collective 完成时间；
- rank 差异是否在 clean-after 消失。

### D3. Channel 维度

使用 verbose event trace 分析每个 channel 的 kernel start/stop：

```text
channel duration
max/min channel duration
channel CV
p90/p99 channel tail
最慢 channel id 的稳定性
```

若 D2H 只使部分 channel 成为 straggler，细粒度打点必须优先覆盖这些 channel，而不是
只输出全局平均值。

Inspector 能支持的结论为：

> D2H 使某 collective、rank、channel 对应的 NCCL kernel 执行区间变长。

Inspector 不能单独支持：

> kernel 时间增长发生在 `fence.acq_rel.sys`。

## 11. 阶段 E：NCCL primitive 细粒度打点

### E1. 打点对象

在 `prims_simple.h` 中拆分观察以下阶段。

#### waitPeer

记录：

- `waitPeer` 包围时间；
- `loadStepValue()` 执行次数；
- 是否进入 spin；
- wait-recv / wait-send；
- conn step 的初值和目标值差距。

#### 数据处理

记录：

- copy 或 reduceCopy 阶段包围时间；
- primitive signature；
- 本 slice 有效字节数；
- direct recv/send、Src/Dst 和 reduction 属性。

该时间是协作线程组阶段时间，不应被解释为单条 load/store 指令延迟。

#### fence

```cpp
t0 = globaltimer();
fence_acq_rel_sys();
t1 = globaltimer();
```

仅由实际执行 fence 的 `RolePostSend` 线程记录。

#### step store

```cpp
t0 = globaltimer();
st_relaxed_sys_global(connStepPtr, step);
t1 = globaltimer();
```

按 post-send 和 post-recv 角色区分。

#### postPeer 整体

```cpp
t0 = globaltimer();
fence_acq_rel_sys();
st_relaxed_sys_global(connStepPtr, step);
t1 = globaltimer();
```

必须单独保留整体测量，因为在 fence 和 store 之间插入 timer 会改变原始关键路径。
不能在同一个 build 中同时打开全部细分打点，再把其绝对时间作为正式性能结果。

### E2. 独立构建变体

third_party 建议生成：

```text
sys-release       原始 .sys，无打点
sys-inst-wait     只测 waitPeer
sys-inst-post     只测 fence + step store
sys-inst-fence    只测 fence
sys-inst-store    只测 step store
sys-inst-copy     只测 primitive 数据阶段
gpu-release       .gpu scope 消融，无打点
gpu-inst-post      .gpu scope 下测 fence + step store
```

每个变体使用独立输出目录、build id 和 `libnccl.so` hash。release build 用于吞吐真值，
instrumented build 只用于解释时间组成。

### E3. 低扰动记录机制

不使用 device `printf`，不在每个 step 上竞争同一个全局 atomic counter。推荐：

- 使用 NCCL 已有的 `%globaltimer`；
- 为每个 rank/channel/group 分配独立固定长度 ring buffer；
- 只由对应角色线程写入；
- 默认只记录选定 channel；
- 支持按 `1/N` step 采样；
- buffer 满后停止记录并设置 overflow 标志，不静默覆盖旧数据；
- communicator teardown 时一次性导出。

每条记录至少包含：

```text
rank
channel_id
collective_type
primitive_signature
group
step
slice_bytes
phase
start_ns
duration_ns
spin_load_count
sample_sequence
```

先扫描采样率：

```text
N = 1, 8, 32, 128
```

选择在延迟分布稳定的前提下，相对对应 release build 扰动最低的采样率。约 2% 是期望
上限，不是自动通过线；若敏感场景无法做到低扰动，应继续降低采样频率，并将打点数据
限制为定性证据。

### E4. 打点实验矩阵

每种 collective 从 C/D 阶段选择一个同步敏感点和一个吞吐主导点。分别运行：

```text
sys-inst-wait
sys-inst-post
sys-inst-fence
sys-inst-store
sys-inst-copy
```

每个 build：

```text
clean x 5
D2H GPU0-only x 5
D2H GPU0-3 all x 5
clean-after x 5
```

该阶段禁止同时启用 Inspector、Nsight Systems 或 Nsight Compute。

输出：

```text
sample count
event count
p50/p90/p99/max duration
spin load count distribution
per-rank distribution
per-channel distribution
per-primitive distribution
per-slice-byte bucket
overflow/dropped sample count
```

### E5. 贡献量级核对

对同一个 collective、size、rank 和 channel 定义：

```text
delta_coll  = Tcoll(D2H)  - Tcoll(clean)
delta_wait  = Twait(D2H)  - Twait(clean)
delta_fence = Tfence(D2H) - Tfence(clean)
delta_store = Tstore(D2H) - Tstore(clean)
delta_copy  = Tcopy(D2H)  - Tcopy(clean)
```

以单 channel 的串行 primitive 路径估计：

```text
critical-path increment
approximately
  effective fence count x delta_fence
  + critical wait increment
  + copy/reduce increment
  + unexplained residual
```

不能把所有 channel、所有角色线程的时间直接求和。channel 间并行，collective 完成
时间通常由关键 channel 或 rank 决定。除 p50 外必须关注 p99/max，因为 D2H 可能主要
制造 fence 或 wait 的长尾。

预测贡献率可写为：

```text
fence contribution estimate
= effective critical-path fence count x delta_fence
  / delta_coll
```

该值用于数量级核对，不要求 wait、copy、fence、store 四项机械相加为 100%。

## 12. 阶段 F：采样工具

### F1. Nsight Systems

Nsight Systems 用于代表场景的必要时序证据：

```text
AG clean / D2H
AR clean / D2H
RS clean / D2H
```

每种 collective 使用阶段 C 中的一个敏感 size。Wrapper 必须同时启动背景程序和
`nccl-tests`，保证两者属于 profiler 的 process tree。使用 NVTX 标记：

```text
background-startup
background-steady
nccl-warmup
nccl-measured
background-stop
clean-after
```

用途：

- 证明 NCCL kernel 与 D2H memcpy 真正重叠；
- 判断下降体现为 kernel duration、API submission 还是 host synchronization；
- 检查 D2H 停止后的恢复时间；
- 排除初始化、allocation、page fault 和意外 device-wide synchronization；
- 对齐 background steady 和 measured collective 区间。

Nsight Systems 与 Inspector 分轮采集。

### F2. Nsight Compute

默认不使用 kernel replay 直接 profile 多 GPU NCCL collective。NCCL kernel 会在设备内
轮询对端 flag，隔离重放单张 GPU kernel 可能导致死锁。

Nsight Compute 只在以下场景条件使用：

1. 单 GPU fence/wait/copy 微基准；
2. HBM/L2 压力 kernel；
3. 已通过小规模 smoke test 证明 application replay 能完整重放多 GPU collective，
   并且每次重放都能建立相同 D2H 背景。

即使 application replay 可用，也只选择最少指标和最少 pass。其聚合 metric 作为数据
通路佐证，system-scope fence 的细粒度时间仍以 `%globaltimer` 打点为主。

### F3. PC sampling 或指令级采样

若当前工具链能在不 replay collective 的条件下提供 GPU PC sampling，则只对 AG/AR/RS
各一个代表点采集 clean/D2H 对照。采样目标为：

- fence/membar 附近的样本占比是否增加；
- `loadStepValue` 轮询路径的样本是否增加；
- reduceCopy/load-store 路径是否增加 stall 样本。

采样结果只能定位热点，不能直接等价为单次指令延迟。必须与源码打点结果交叉验证。

## 13. 阶段 G：`.sys -> .gpu` 作用域消融

该阶段在 third_party 中进行，并且位于 `.sys` 打点之后。

第一轮只修改 Ring+Simple 主路径相关的 post fence：

```diff
- fence_acq_rel_sys();
+ fence_acq_rel_gpu();
```

`st_relaxed_sys_global(connStepPtr, step)` 继续保持 `.sys`。这样隔离 fence scope，避免
同时改变 step publication 的可见性语义。

比较：

```text
third_party sys-release
third_party gpu-release
third_party sys-inst-post
third_party gpu-inst-post
```

对 AG、AR、RS 分别记录：

```text
clean 绝对性能
D2H slowdown
fence/post 延迟
waitPeer 延迟
channel imbalance
未解释 residual
correctness
```

至少执行 nccl-tests correctness validation、不同消息大小和足够迭代次数。当前单机
NVLink P2P 路径上 validation 通过，只能说明该实验场景未观察到错误，不能证明把
`.gpu` 无条件用于 host、network、GDR 或多机场景是安全的。

推荐结论形式：

> `.gpu` scope 消除了约 X% 的 D2H 额外时间；剩余 Y% 与 waitPeer、数据搬运或
> channel imbalance 相关。因此 system-scope fence 是主要贡献者之一，但不是完整
> 根因。

只有当 `.sys` 细粒度打点、collective 增量数量级和 `.gpu` 消融三者共同吻合时，才能
把 system-scope fence 表述为主要贡献者。

## 14. 分析决策表

| 观测组合 | 支持的解释 | 仍不能声称的内容 |
| --- | --- | --- |
| Inspector 显示 kernel/channel 变长；fence 打点显著变长；数量级吻合 | system-scope fence 是主要贡献者 | fence 是唯一根因 |
| `.gpu` 消除大部分 slowdown，但 wait/copy 仍变长 | fence 主导，另有数据或同步路径影响 | `.gpu` 已解决全部问题 |
| fence 变化小，waitPeer 的时长和 spin load 数增加 | 对端发布或流水线等待传播更重要 | wait 中每次 load 都变慢 |
| copy/reduce 阶段显著变长 | L2/HBM/SM load-store 路径需要进一步消融 | 已定位到具体 HBM channel/L2 slice |
| GPU0-only D2H 只产生 GPU0 rank/channel straggler | GPU-local source-side 机制 | 系统内其他共享资源完全无影响 |
| 同卡下降、远端不下降 | 排除主要 host/NUMA/global 因素 | 已排除 GPU 内所有非 fence 因素 |
| D2H/H2D 同程度下降 | 非 D2H 特有机制，应重新评估 | system-scope fence 假设成立 |
| clean-after 不恢复 | 存在残留状态、清理或热状态问题 | treatment slowdown 可直接归因 |
| profiler 下有差异、无 profiler 无差异 | profiler artifact | 原始干扰成立 |

## 15. 统计与重复规则

1. 正式主效应至少 7 个 paired repetitions；打点诊断至少 5 次；
2. 保存每次独立结果，不只保存平均值；
3. 报告 median、mean、min/max、标准差或 MAD，以及 paired slowdown；
4. 对 operation/fence/wait 分布报告 p50、p90、p99 和 max；
5. 同时报告 effect size 和基线变异，不能只报告百分比；
6. 不混合 short-burst 与 steady-state；
7. 不混合 Inspector ON/OFF；
8. 不混合官方、third_party、instrumented 和 patched build；
9. 不混合 profiler trace 运行和无 profiler 性能运行；
10. 背景压力使用实际 per-GPU GB/s，不只使用配置档位。

## 16. 结果目录与可追溯性

建议在 `nccl-tests/interference/results/<run-id>/` 保存原始数据，在当前分析仓库的
`doc/results/nccl-contention/` 保存经过选择的长期结果和摘要。

```text
doc/results/nccl-contention/
  environment/
  build-calibration/
  regime-sweep/
  main-effect/
  locality-direction/
  inspector/
  instrumentation/
  nsys/
  ncu-microbench/
  scope-ablation/
```

每个 case 目录至少包含：

```text
command.txt
environment.txt
build-identity.json
nccl-tests.log
nccl-tests.json
background.log
background.json
inspector/              # 仅 Inspector case
instrumentation.bin     # 仅打点 case
instrumentation.csv
nsys.log                # 仅 nsys case
status.json
```

最终报告中的每个表格单元必须能够追溯到一个原始 case 目录。分析脚本、字段定义和
过滤条件必须随结果一起保存，避免依赖手工选择时间区间。

## 17. 推荐执行顺序

### 第一批：建立可信基线

1. 保存 `/home/songxb26/HyLink/nccl` 当前 dirty diff；
2. 固定官方 NCCL 2.31.2 `.sys` 和 `sm_70` build；
3. 构建 third_party sys-release 和 Inspector；
4. 验证两套库、SASS、correctness 和 `ldd`；
5. 将受控 D2H 工具和 runner 放入 `nccl-tests/interference/`；
6. 完成 Inspector ON/OFF 开销校准。

### 第二批：宏观收敛

1. 完成 AG/AR/RS regime sweep；
2. 执行 4 MiB、64 MiB、256 MiB、1 GiB 主矩阵；
3. 对显著点执行 GPU0-only、all-GPU、H2D 和远端负对照；
4. 选出每种 collective 的同步敏感点和吞吐主导点；
5. 采集代表性 Inspector 和 Nsight Systems 数据。

### 第三批：底层归因

1. 实现低扰动 per-channel instrumentation buffer；
2. 分别构建 wait、post、fence、store、copy 诊断版本；
3. 校准采样率和打点扰动；
4. 完成三种 collective 的细粒度分布分析；
5. 用单 GPU 微基准和条件采样工具补充指令级证据；
6. 计算 fence/wait/copy 对 collective 增量的量级贡献。

### 第四批：作用域消融

1. 构建 `.gpu` release 和 `.gpu` post instrumentation；
2. 重跑选定的 AG/AR/RS clean/D2H case；
3. 做 correctness 和 SASS 检查；
4. 分析被消除部分和 residual；
5. 形成最终根因报告和适用边界。

## 18. 停止条件

满足以下任一条件时暂停向更底层扩展，先解释当前异常：

- 主效应在 7 次 paired repetitions 中不稳定；
- clean-after 无法回到 clean-before 的基线范围；
- Inspector 开关改变 algorithm、protocol 或 channel 数；
- 官方和 third_party sys-release 无法建立等价基线；
- 打点 build 的扰动过大且无法通过 channel/step 采样控制；
- `.sys` fence 延迟不随 D2H 变化；
- wait、copy、fence 和 store 都不能解释 collective 增量；
- 采样工具改变现象或造成 collective 挂起；
- `.gpu` validation 出错或出现不稳定结果。

最后一种情况下不能通过减少 correctness 检查来继续性能比较，应首先恢复 `.sys`
基线并检查作用域假设。

## 19. 最终报告应回答的问题

最终结果报告至少明确回答：

1. AG、AR、RS 的 clean、D2H 和恢复性能分别是多少；
2. short-burst 与 steady-state 是否得到相同结论；
3. 干扰是否具有 GPU-local 和 D2H 方向性；
4. Inspector 看到的慢点位于哪些 rank/channel；
5. wait、copy、fence、store 的延迟和长尾分别如何变化；
6. 源码预测的 primitive/fence 次数与实测是否一致；
7. `.gpu` scope 消除了多少额外时间，剩余时间如何解释；
8. Nsight Systems、采样结果和内部打点是否互相支持；
9. 哪些结论只适用于当前 V100、单机、NVLink、Ring+Simple 路径；
10. 下一步工程化修改需要满足哪些 memory-ordering 安全条件。

本方案的最终目标不是证明一个预设结论，而是形成一条可复现、可反驳的证据链：

```text
稳定性能现象
-> 物理局部性和方向负对照
-> Inspector 的 collective/rank/channel 宏观定位
-> primitive 内部 wait/copy/fence/store 细粒度时间
-> 必要的时序和采样佐证
-> .sys/.gpu 作用域消融及 residual 分析
```
