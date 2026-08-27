# Q1 source-chain 诊断结果

本目录是 Q1 筛选矩阵的原始结果，运行时间为 2026-08-26，固定口径为 4 卡
allpairs、255M、warmup=10、重复次数 20/300，背景为无背景或四卡 D2H。每个
配置执行 3 次。完整 CSV 见 [`summary.csv`](summary.csv)，环境快照见
[`environment.txt`](environment.txt)。

## 1. 筛选结果

`aggregateGBps` 使用 D2D source makespan 计算。D2H 下降百分比在同一配置、同一
重复次数下由无背景均值计算。

| streamMode | streamDependency | repeats | 无背景均值 (GB/s) | 全卡 D2H 均值 (GB/s) | 相对下降 |
| --- | --- | ---: | ---: | ---: | ---: |
| per-source | none | 20 | 178.869 | 127.959 | 28.46% |
| per-edge | none | 20 | 581.655 | 578.011 | 0.63% |
| per-edge | source-chain | 20 | 193.816 | 192.851 | 0.50% |
| per-source | none | 300 | 188.935 | 161.017 | 14.78% |
| per-edge | none | 300 | 581.875 | 578.193 | 0.63% |
| per-edge | source-chain | 300 | 193.854 | 192.892 | 0.50% |

关键结果是：`per-edge + source-chain` 保留了每个 source 的三条 edge 的执行依赖，
但 D2H 相对下降仍只有约 0.50%，和 `per-edge + none` 的约 0.63% 同量级；它没有
恢复 `per-source` 的约 14.78% 下降。这里不应比较三个配置的绝对 GB/s：
`source-chain` 有意把同一 source 的三条 edge 串行化，绝对吞吐自然接近
per-source；判别量是各自的 D2H treatment 差值。

## 2. Nsight Systems trace

三份 trace 都是 repeats=300、全卡 D2H、source-major edge order。原始命令见
[`nsys-commands.txt`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/nsys-commands.txt)。

| 配置 | trace | SQLite 分析 |
| --- | --- | --- |
| per-source + none | [`per-source-none-d2h-v2.nsys-rep`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/per-source-none-d2h-v2.nsys-rep) | [`per-source-none-d2h-v2.json`](../trace-analysis/20260826T103320Z/per-source-none-d2h-v2.json) |
| per-edge + none | [`per-edge-none-d2h.nsys-rep`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/per-edge-none-d2h.nsys-rep) | [`per-edge-none-d2h.json`](../trace-analysis/20260826T103320Z/per-edge-none-d2h.json) |
| per-edge + source-chain | [`per-edge-source-chain-d2h.nsys-rep`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q1-source-chain/20260826T103320Z/per-edge-source-chain-d2h.nsys-rep) | [`per-edge-source-chain-d2h.json`](../trace-analysis/20260826T103320Z/per-edge-source-chain-d2h.json) |

逐条 P2P 统计如下；分析 JSON 的 `p2pRows` 保存了 source、destination、stream、
start、end、duration、round 和 queue position。

| 配置 | P2P stream 数 | P2P 数量 | duration p50/p90/p99/max (ms) | >8 ms 数量 | 慢波数 |
| --- | ---: | ---: | --- | ---: | ---: |
| per-source + none | 4 | 3720 | 5.532 / 5.544 / 13.465 / 13.502 | 260 | 68 |
| per-edge + none | 12 | 3720 | 5.586 / 5.603 / 5.603 / 5.605 | 0 | 0 |
| per-edge + source-chain | 12 | 3720 | 5.531 / 5.543 / 5.544 / 5.583 | 0 | 0 |

在 per-source trace 中，68 个慢波都包含 4 个 source GPU；慢样本的 start-time
span 为 p50=2.089 ms、p90=2.576 ms、最大 3.138 ms。因此“慢波”不要求四卡
精确同时启动，仍与有限时间窗口内的 queue/cadence coupling 一致。两个 per-edge
配置没有 >8 ms P2P，也没有慢波。

## 3. Q1 判定

结果不支持“只要恢复 source 内 FIFO 依赖，原始 D2H 竞争就会恢复”。更准确的结论是：

1. 独立 stream 本身改变了可被 copy 调度器选择的 eligible work、stream identity
   或内部队列/engine 映射；这对消除长尾是必要条件之一。
2. `source-chain` 仍然保持严格的 source 内串行关系，却没有重新产生 per-source
   的 D2H slowdown，说明 FIFO/head-of-line 不是单独充分条件。
3. 这不能证明 FIFO 完全无关：原始现象可能需要“单 stream 的队列形态”与特定
   stream/engine 映射共同出现。当前证据仍不能把根因具体归因到某个 CE、NVLink
   path 或单一硬件队列。

因此下一步按报告进入 Q2：在 `streamDependency=none` 下扫描每个 source 的
eligible stream 数量 1/2/3，并继续用逐条 P2P 的 p99、>8 ms 数量和慢波统计判定
是否存在并行度阈值。
