# D2H 背景流量导致 NCCL AllGather 性能下降 —— 根因分析报告

> 调查日期：2026-06-07
> 分析源码：`/root/sxb/nccl`（NCCL 2.30.7）；运行时实际链接 `libnccl.so.2`（2.26.2），prims 同步路径机制一致
> 背景流量：`/root/sxb/Adaptive-CCL/tests/pcie_test/mem_test/d2h_test.cpp`（二进制 `d2h_test_orig`）
> 测试程序：`/root/sxb/nccl-tests`（all_gather_perf）
> 微基准：`/tmp/sysfence_bench.cu`

---

## 0. 一句话结论

NCCL 性能下降的根因是 **NCCL 每个流水线 step 必发的系统作用域内存栅栏 `fence.acq_rel.sys`**：当同一张物理 GPU 上有 D2H DMA 在飞时，该栅栏的延迟从 **588 ns 翻倍到 1430 ns（+143%）**。这个栅栏处在 NCCL 生产者-消费者流水线的串行关键路径上，因此它一变慢，整个 collective 就被节流。

普通 `cudaMemcpy`（D2D/CE 拷贝）没有这种细粒度同步栅栏，所以只有在真正打满 NVLink/NVSwitch 带宽（fullmesh）时才掉，而 2 卡 / ring / 小数据都不掉 —— 这正是最初观察到的"NCCL 与 D2D 行为不一致"的原因。

---

## 1. 问题描述（来自现象观察）

- 开启 D2H 背景拷贝后，nccl-test（AllGather，单进程多线程、非 MPI）性能**小幅下降**。
- 用简单的 **D2D 拷贝**做类似测试：性能下降**只在特殊 case**（4 卡 fullmesh）出现；2 卡不掉，4 卡 ring 几乎不掉。
- 无论 NCCL 还是 D2D，nsys 采样都只看到 **kernel 本身变慢**。
- NCCL 在 2 卡 / 4 卡、各种数据 size 下**都**有下降。
- 怀疑：NCCL kernel 的实现与 D2H 拷贝产生了竞争。

---

## 2. 硬件环境（关键前提）

- **8 × NVIDIA H200**，GPU 间 **NV18（NVLink/NVSwitch 全互联）**。
- PCIe **Gen5 x16**。
- GPU0–3 在 NUMA0，GPU4–7 在 NUMA1。
- 测试时每卡仅 ~2.1 GB 空闲显存（其余被 vLLM 占用）。

**推论**：NCCL 走 **NVLink**，D2H 走 **PCIe**，两者**物理链路完全分离**。所以下降的根源不可能是链路带宽竞争，必然在 GPU 片内的共享资源。

---

## 3. 源码分析：NCCL kernel 与普通 memcpy 的本质区别

单机多卡 AllGather 默认走 `NCCL_ALGO_RING` + `SIMPLE` 协议（实测默认即 Simple），核心拷贝在
`src/device/all_gather.h:57-74`（`directCopySend` / `directRecvCopyDirectSend` / `directRecv`），
底层是 `src/device/prims_simple.h` 的生产者-消费者流水线。它与 `cudaMemcpy` 有三处本质区别：

### 3.1 忙等自旋轮询对端步进计数器（延迟敏感）
`src/device/prims_simple.h:107-114` `waitPeer`：
```cpp
while (connStepCache + (...) < step + StepPerSlice) {
    connStepCache = loadStepValue(connStepPtr);   // ld_volatile_global，反复轮询
    if (checkAbort(flags, Aborted, spins)) break;
}
```
LL 协议更极端：`src/device/prims_ll.h:113-119` `readLL` 反复 `ld.volatile.global.v4.u32` 轮询内联 flag。

### 3.2 每个 step 一次系统作用域内存栅栏（**主因**）
`src/device/prims_simple.h:165-172` `postPeer`：
```cpp
if (Send && (flags & RolePostSend) && (dataStored || (flags & ConnFifoEnabled)))
    fence_acq_rel_sys();                 // ← 系统作用域栅栏
st_relaxed_sys_global(connStepPtr, step);// ← 系统作用域 store
```
`src/device/op128.h:393-397`：
```cpp
__device__ __forceinline__ void fence_acq_rel_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");   // 老架构为 membar.sys
}
```

### 3.3 数据拷贝经 L2/HBM
`src/device/common_kernel.h` 的 `reduceCopy` 用 SM 的向量化 ld/st 经 L2 搬数据。

### 3.4 缓冲区/flag 位置
`src/transport/p2p.cc:123-124` `P2P_USE_CUDA_MEMCPY` 默认 0 → 默认 NVLink P2P 为 SM 直驱，数据缓冲与同步 flag 都在 **GPU 显存**，但访问使用 **system scope**（`.sys`）。因此竞争不是"轮询 flag 走 PCIe"，而是 **system-scope 内存序操作对在飞 PCIe 流量敏感**。

---

## 4. 实验与数据

测试命令（单进程 2 线程各 1 卡，对应"单进程多线程、非 MPI"）：
```
./build/all_gather_perf -b 4M -e 16M -f 2 -t 2 -g 1
```
D2H 背景：`./d2h_test_orig`（写死跑在 GPU0/1，每卡 ~586MB，GPU util 100%）。**参数未改动。**

### 4.1 实验 1 —— 复现（nccl 与 d2h 同在 GPU0/1，默认 Simple 协议）

噪声基线：连跑 3 次，波动 < 1%。

out-of-place 单次耗时：

| size | baseline | +D2H | 下降 |
|---|---|---|---|
| 4M  | 26.97 μs | 28.50 μs | **+5.7%** |
| 8M  | 36.78 μs | 37.80 μs | +2.8% |
| 16M | 54.57 μs | 55.35 μs | +1.4% |

✅ 复现"小幅下降、小数据更明显"。**相对下降随数据增大而减小** —— 与 HBM 带宽竞争（应随 size 加剧）相反，指向延迟/同步机制。

### 4.2 实验 2 —— 隔离对照（nccl 移到 GPU2/3，d2h 留在 GPU0/1）

| size | baseline | nccl@GPU2/3（d2h@0/1） |
|---|---|---|
| 4M  | 26.97 μs | 26.94 μs |
| 16M | 54.57 μs | 54.54 μs |

✅ **下降完全消失**。证明竞争是**单 GPU 片内**的，排除 host 内存控制器 / PCIe root complex / NUMA / NVSwitch 等系统级因素。

### 4.3 实验 3 —— 协议敏感度（同卡竞争 vs 干净卡，相对下降，out-of-place）

| 协议 | 4M | 8M | 16M |
|---|---|---|---|
| **Simple**（默认） | +8.8% | +2.8% | +2.5% |
| **LL128** | +5.3% | +3.6% | +3.6% |
| **LL** | +7.0% | +0.3% | +1.1% |

✅ 三种协议**都是 4M 掉得最多、随 size 收敛**。默认协议经核对即 Simple（无 `NCCL_PROTO` 时数字与强制 Simple 完全一致）。

### 4.4 ncu 尝试失败（记录在案）

`ncu` 对 NCCL 多卡 collective 做 kernel-replay 会**死锁挂起**：NCCL kernel 在 kernel 内部自旋等待对端 GPU 的 flag，而 ncu 串行隔离重放两个 GPU 的 kernel，对端 flag 永不更新（超时被 kill，exit 143）。**结论：ncu kernel-replay 不适用于 NCCL collective。** 改用单 GPU 微基准。

### 4.5 实验 4 —— 单 GPU 微基准（决定性，拆解机制）

`/tmp/sysfence_bench.cu`，单块 128 线程，循环 200000 次，各操作分别计时。
对照 NCCL 中的对应操作。结果（ns/iter，3 次完全一致）：

| 操作（NCCL 对应） | 无 D2H | 同卡 D2H | 变化 |
|---|---|---|---|
| 纯计算（绝对对照） | 2.357 | 2.359 | +0.1% |
| `membar.cta` 本地栅栏（廉价对照） | 9.111 | 9.113 | +0.02% |
| `ld.volatile.global` flag 轮询（`waitPeer`） | 18.83 | 20.90 | +11% |
| **`fence.acq_rel.sys`（`postPeer`）** | **588.6** | **1430.2** | **+143%（2.43×）** |

关键解读：
- 纯计算 / 本地栅栏**完全不受影响** → 排除 SM 算力、时钟降频、调度因素。
- flag 轮询仅 +11% → L2/HBM 访存延迟轻微竞争，**次要因素**。
- 系统栅栏 **+143%** → **主因**；且其绝对开销本就极高（588ns，是本地栅栏的 65 倍）。


### 4.6 实验 5 —— V100 大消息场景与 `NCCL_BUFFSIZE` 扫描

补充测试日期：2026-06-08。

平台从 H200 切换到 **4 × Tesla V100-SXM2-32GB**，GPU 间拓扑为 `NV2`。测试程序为当前目录 `nccl-tests` 的 `all_gather_perf`，版本信息：

```text
nccl-tests version 2.17.9
nccl-headers=22105
nccl-library=22105
```

测试命令形态：

```bash
cd /home/songxb26/HyLink/nccl-tests

# clean baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  [NCCL_BUFFSIZE=<bytes>] \
  ./build/all_gather_perf -b <size> -e <size> -g 4 -n 20 -w 5

# D2H 背景流量
/home/songxb26/HyLink/Adaptive-CCL/tests/pcie_test/mem_test/d2h_test_orig

# +D2H
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  [NCCL_BUFFSIZE=<bytes>] \
  ./build/all_gather_perf -b <size> -e <size> -g 4 -n 20 -w 5
```

下表为 out-of-place `time`，单位为 μs。箭头左侧为 clean baseline，右侧为叠加 D2H 背景流量，括号中是 D2H 相对 clean 的时间增幅。

| `NCCL_BUFFSIZE` | 64M | 128M | 256M | 512M | 1G |
|---:|---:|---:|---:|---:|---:|
| default | 426.38 → 524.86 (+23.1%) | 817.76 → 1015.17 (+24.1%) | 1583.10 → 1746.28 (+10.3%) | 3129.52 → 3447.38 (+10.2%) | 6155.00 → 6642.34 (+7.9%) |
| 8M | 417.64 → 521.57 (+24.9%) | 785.17 → 890.36 (+13.4%) | 1541.58 → 1681.73 (+9.1%) | 3035.12 → 3207.53 (+5.7%) | 6036.14 → 6314.63 (+4.6%) |
| 16M | 418.73 → 526.55 (+25.7%) | 785.90 → 889.40 (+13.2%) | 1520.84 → 1620.00 (+6.5%) | 3013.48 → 3151.90 (+4.6%) | 5975.64 → 6149.21 (+2.9%) |
| 24M | 418.11 → 526.34 (+25.9%) | 787.10 → 891.69 (+13.3%) | 1521.72 → 1621.51 (+6.6%) | 2992.43 → 3096.30 (+3.5%) | 5966.56 → 6093.90 (+2.1%) |
| 32M | 418.74 → 526.87 (+25.8%) | 790.67 → 890.22 (+12.6%) | 1527.38 → 1625.03 (+6.4%) | 2992.40 → 3093.29 (+3.4%) | 5955.00 → 6092.21 (+2.3%) |
| 48M | 417.62 → 527.18 (+26.2%) | 786.06 → 889.25 (+13.1%) | 1522.85 → 1624.74 (+6.7%) | 2992.68 → 3095.86 (+3.4%) | 5937.84 → 6040.50 (+1.7%) |

关键观察：

1. **大消息下 D2H 影响仍然存在，但会被摊薄**：默认配置下，D2H 增幅从 64M/128M 的约 +23%～+24%，下降到 1G 的 +7.9%。
2. **增大 `NCCL_BUFFSIZE` 对 128M 以上消息有效**：128M 从 default 的 +24.1% 降到约 +13%；256M 从 +10.3% 降到约 +6%～+7%；512M/1G 的收益更明显。
3. **64M 消息不受益**：64M 在各个 `NCCL_BUFFSIZE` 下仍有约 +25% 的 D2H 增幅，说明这个 size 下同步/启动/流水线填充开销仍占较大比例，单纯增大 buffer 无法掩盖。
4. **较大 buffer 会降低 D2H 相对影响，但并非越大越好**：24M、32M、48M 在 512M/1G 上接近，48M 的 1G 最好（+1.7%），但 24M/32M 更稳健。
5. **`NCCL_BUFFSIZE=64M` 在该 V100 环境不可用**：clean baseline 与 +D2H 均多次出现只打印表头、不产生数据行的长时间自旋/卡住现象，因此不纳入性能比较，不建议使用。


### 4.7 实验 6 —— 将 Simple 协议 post fence 从 `.sys` 降为 `.gpu`

基于 4.5 的微基准结论，进一步验证源码级缓解方案：将 NCCL Simple primitive 中每个 step 的 post fence 从 system scope 降为 GPU scope。

实验 patch：

```diff
diff --git a/src/device/prims_simple.h b/src/device/prims_simple.h
@@
-        fence_acq_rel_sys();
+        fence_acq_rel_gpu();  // EXPERIMENT: reduce fence scope for local NVLink P2P
@@
-          fence_acq_rel_sys();
+          fence_acq_rel_gpu();  // EXPERIMENT: reduce fence scope for local NVLink P2P
```

只替换 `prims_simple.h` 中两个 Simple post fence，**没有修改** `st_relaxed_sys_global(connStepPtr, step)`，即 step counter 的 store 仍保持 `.sys` scope。这样可以先隔离验证 fence scope 本身对 D2H 干扰的贡献。

编译时只针对当前 V100 架构生成 `sm_70`：

```bash
cd /home/songxb26/HyLink/nccl
make -j src.build NVCC_GENCODE='-gencode=arch=compute_70,code=sm_70'
```

`nccl-tests` 运行时确认链接到本地重编译库：

```text
libnccl.so.2 => /home/songxb26/HyLink/nccl/build/lib/libnccl.so.2
```

#### 4.7.1 `.gpu` fence + default `NCCL_BUFFSIZE`

下表为 out-of-place `time`，单位为 μs。

| size | clean | +D2H | D2H 增幅 | `.sys` default 增幅 |
|---:|---:|---:|---:|---:|
| 64M | 424.97 | 486.12 | +14.4% | +23.1% |
| 128M | 816.47 | 868.52 | +6.4% | +24.1% |
| 256M | 1579.84 | 1624.36 | +2.8% | +10.3% |
| 512M | 3117.04 | 3149.90 | +1.1% | +10.2% |
| 1G | 6136.02 | 6190.02 | +0.9% | +7.9% |

关键结论：

1. `.gpu` fence patch 显著降低 D2H 背景流量造成的额外开销。
2. 128M 以上收益尤其明显：128M 从 +24.1% 降到 +6.4%，512M 从 +10.2% 降到 +1.1%，1G 从 +7.9% 降到 +0.9%。
3. validation 未报错，说明在当前 4 卡 V100 单机 AllGather 路径下，该实验 patch 可以正确跑通。
4. 该 patch 仍是实验性质：它隐含假设当前通信路径是单机 GPU P2P/NVLink，不需要 host/network/GDR 观察到 data-store 与 step-store 之间的 system-scope ordering。

### 4.8 实验 7 —— `.gpu` fence 下继续扫描 `NCCL_BUFFSIZE`

在 4.7 的 `.gpu` fence patch 基础上，继续扫描 `NCCL_BUFFSIZE`。每个 buffer 均测试 clean baseline 和 +D2H。下表为 out-of-place `time`，单位为 μs。箭头左侧为 clean，右侧为 +D2H，括号中为 D2H 相对 clean 的时间增幅。

| `NCCL_BUFFSIZE` | 64M | 128M | 256M | 512M | 1G |
|---:|---:|---:|---:|---:|---:|
| default | 424.77 → 482.44 (+13.6%) | 817.28 → 870.24 (+6.5%) | 1575.69 → 1623.57 (+3.0%) | 3105.13 → 3149.65 (+1.4%) | 6174.38 → 6191.89 (+0.3%) |
| 16M | 425.48 → 481.81 (+13.2%) | 794.65 → 851.73 (+7.2%) | 1528.16 → 1585.34 (+3.7%) | 3015.09 → 3072.81 (+1.9%) | 5972.94 → 6038.28 (+1.1%) |
| 24M | 418.20 → 482.19 (+15.3%) | 786.47 → 850.14 (+8.1%) | 1520.82 → 1585.53 (+4.3%) | 2991.08 → 3057.73 (+2.2%) | 5951.00 → 6021.34 (+1.2%) |
| 32M | 418.14 → 482.71 (+15.4%) | 786.18 → 851.32 (+8.3%) | 1520.80 → 1585.95 (+4.3%) | 2991.20 → 3055.72 (+2.2%) | 5952.20 → 6017.90 (+1.1%) |
| 48M | 417.91 → 483.93 (+15.8%) | 785.95 → 850.22 (+8.2%) | 1521.05 → 1585.47 (+4.2%) | 2992.06 → 3058.83 (+2.2%) | 5936.53 → 6002.98 (+1.1%) |

关键观察：

1. 在 `.gpu` fence patch 下，D2H overhead 已经很低，继续调 `NCCL_BUFFSIZE` 的主要收益转为改善 clean 性能和 +D2H 绝对时间，而不是进一步大幅降低相对 overhead。
2. 512M/1G 场景下，`16M`～`48M` 的 +D2H 绝对时间均优于 default。
3. 1G 下 `48M` 最快：default +D2H 为 6191.89 μs，`48M` +D2H 为 6002.98 μs。
4. 512M 下 `32M` 略快：`32M` +D2H 为 3055.72 μs，`24M/48M` 基本持平。
5. 64M 下 `NCCL_BUFFSIZE` 调整仍无明显收益，+D2H 绝对时间都在约 482～484 μs。

阶段性结论：**源码级 `.gpu` fence patch 是主收益，`NCCL_BUFFSIZE` 是辅助收益**。如果 patch 可被安全限定到纯 GPU P2P/NVLink 路径，大消息场景下可优先使用 `.gpu` fence；随后再按消息大小选择 `32M` 或 `48M` buffer。

---

## 5. 根因机制

`fence.acq_rel.sys` 的 `.sys` 作用域要求保证本线程之前的访存对**整个系统一致性域（含通过 PCIe 连接的 host）可见**。硬件实现必须把 GPU 内尚未到达系统可见点的访存"排空/定序"到一致状态。当 Copy Engine 持续将数据经 PCIe 灌向 host 时，系统侧 in-flight 写事务增多，栅栏需要等待/定序的对象更多 → 完成延迟翻倍。

这是一个**纯硬件、单 GPU 片内**的效应，与 NVLink 带宽、host 内存控制器、NUMA 无关 —— 与实验 2"换卡即消失"完全自洽。

NCCL 在 `postPeer` 中**每个 step 都发一次**该栅栏，处于流水线串行关键路径，因此其延迟翻倍直接节流整个 collective。

---

## 6. 所有现象的统一解释

| 现象 | 解释 |
|---|---|
| NCCL 全 size、2/4 卡都掉 | 栅栏是 per-step、与数据量无关，故任何 size/卡数都掉（含大数据） |
| 小数据掉得最多 | 小数据每步搬字节少，栅栏开销占比最大（4M +5.7% → 16M +1.4%） |
| 量级吻合 | 每栅栏多 ~842ns，2 卡 ring 关键路径约 2–3 个串行栅栏 → 多 ~1.7–2.5μs，对上 4M 实测 +1.5～2.4μs |
| 纯 D2D 只在 fullmesh 掉 | CE 拷贝无栅栏/无 flag，纯带宽流式，只有打满 NVSwitch 才掉 |
| 换卡后不掉 | 栅栏是单 GPU 片内操作，竞争源（D2H DMA）必须在同卡 |
| nsys 只看到 kernel 变慢 | 多出的时间花在 kernel 内部卡在 `fence.acq_rel.sys`，采样只体现为 kernel 时间变长 |

---

## 7. 缓解方向

1. **隔离背景流量（最稳）**：让 D2H 别跑在 NCCL 用到的物理 GPU 上（实验 2 证明换卡即零影响）。
2. **降低 D2H 在飞深度**：减小单次 D2H 块大小、降并发/频率，缩短同一时刻 PCIe in-flight 事务，直接压低栅栏额外延迟。
3. **NCCL 调参掩盖延迟（治标）**：增大 `NCCL_MIN_NCHANNELS` / `NCCL_NTHREADS`，加深流水线、用更多 warp 掩盖单步栅栏延迟。
4. **从源码根治（Adaptive-CCL 可发力点）**：发力点为 `src/device/prims_simple.h` `postPeer` 的 `fence_acq_rel_sys()` 与 sys-scope step 更新（`src/device/op128.h` 的 `.sys` 原语）。对**同进程、纯 NVLink、无 host/网络/GDR 可见性依赖**的 P2P 连接，理论上可将作用域从 `.sys` 降为 `.gpu`（`fence.acq_rel.gpu`），代价是必须严格证明该连接不涉及任何需要系统级可见性的对端。属侵入式改动，但正是本项目可从根上动的点。

### 7.1 NCCL 配置建议

针对当前 V100 4 卡 AllGather + D2H 背景流量的测试结果：

0. **若允许修改 NCCL 源码，优先验证 `.gpu` fence patch**：在当前单机 V100 AllGather + NVLink P2P 路径中，将 Simple post fence 从 `.sys` 降到 `.gpu` 后，default buffer 下 1G 的 D2H 增幅从 +7.9% 降到 +0.9%，512M 从 +10.2% 降到 +1.1%。这是目前最接近根因的缓解手段，但必须限定在不依赖 host/network/GDR system-scope ordering 的连接上。
1. **大消息场景优先尝试 `NCCL_BUFFSIZE=33554432` 或 `NCCL_BUFFSIZE=50331648`**：
   - `32M` 在 512M/1G 上 D2H 增幅约 +3.4% / +2.3%。
   - `48M` 在 512M/1G 上 D2H 增幅约 +3.4% / +1.7%，1G 最优。
   - 若追求稳健性，先用 `32M`；若主要负载为 1G 级别大消息，可尝试 `48M`。
2. **`NCCL_BUFFSIZE=16M` 是保守可用值**：相较 default，256M/512M/1G 的 D2H 增幅从 +10.3%/+10.2%/+7.9% 降到 +6.5%/+4.6%/+2.9%，且未观察到卡住。
3. **不建议在该环境使用 `NCCL_BUFFSIZE=64M`**：多次出现 clean 和 +D2H 下均只打印表头、不输出结果的情况，疑似触发 NCCL 运行时/资源配置边界问题。
4. **`NCCL_BUFFSIZE` 主要改善 128M 以上消息**：64M 消息在 8M～48M buffer 下仍有约 +25% 的 D2H 增幅，说明瓶颈仍以同步延迟和流水线填充为主。
5. **`NCCL_BUFFSIZE` 应与 `NCCL_MIN_NCHANNELS` / `NCCL_NTHREADS` 联合调参**：增大 buffer 会减少 step 数，从而减少 per-step `fence.acq_rel.sys` 次数，但也可能增加 pipeline bubble；增大 channel/thread 则可能提高并行度、掩盖同步延迟。实际配置应按目标消息大小分段验证。

推荐起点：

```bash
# 保守配置，适合 256M～1G 大消息
export NCCL_BUFFSIZE=33554432

# 偏向 512M～1G 大消息，当前 V100 测试中 1G 最优
export NCCL_BUFFSIZE=50331648
```


若已经应用 `.gpu` fence patch，推荐起点调整为：

```bash
# .gpu fence patch 后，大消息较稳的折中配置
export NCCL_BUFFSIZE=33554432

# .gpu fence patch 后，偏向 1G 消息的配置
export NCCL_BUFFSIZE=50331648
```

注意：`.gpu` fence patch 不应作为无条件全局替换进入通用 NCCL。更合理的工程化方向是：仅对确认使用 GPU 内 P2P/NVLink、且同步对象不需要 host/network/GDR system-scope 可见性的连接，选择 GPU-scope fence；其他路径仍保留 system-scope fence。

---

## 8. 复现方法

```bash
# baseline
cd /root/sxb/nccl-tests
./build/all_gather_perf -b 4M -e 16M -f 2 -t 2 -g 1

# 开启 D2H 背景（GPU0/1）
cd /root/sxb/Adaptive-CCL/tests/pcie_test/mem_test
nohup ./d2h_test_orig > /tmp/d2h.log 2>&1 &

# 同卡复现下降
cd /root/sxb/nccl-tests
CUDA_VISIBLE_DEVICES=0,1 ./build/all_gather_perf -b 4M -e 16M -f 2 -t 2 -g 1
# 换卡对照（应无下降）
CUDA_VISIBLE_DEVICES=2,3 ./build/all_gather_perf -b 4M -e 16M -f 2 -t 2 -g 1

# 微基准（决定性证据）
nvcc -O3 -arch=sm_90 /tmp/sysfence_bench.cu -o /tmp/sysfence_bench
CUDA_VISIBLE_DEVICES=0 /tmp/sysfence_bench 200000 128 5   # 对比 d2h 开/关

# 清理
pkill -f d2h_test_orig
```
