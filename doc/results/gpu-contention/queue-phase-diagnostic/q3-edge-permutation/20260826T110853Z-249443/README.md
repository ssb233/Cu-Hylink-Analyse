# Q3 edge-permutation 诊断结果

本目录执行了报告第 8.4 节的六种单 stream edge permutation：4 卡 allpairs、
255M、warmup=10、`streamsPerSource=1`、`streamDependency=none`，每个点 3 次，
背景为无背景或四卡 D2H。执行脚本为
[`run_q3_edge_permutation_matrix.sh`](../../../../../../scripts/run_q3_edge_permutation_matrix.sh)，完整结果见 [`summary.csv`](summary.csv)。

## 1. Aggregate 结果

300-repeat 的 3 次均值如下；D2H 下降相对同一 permutation 的无背景均值计算：

| permutation | 无背景 (GB/s) | 全卡 D2H (GB/s) | 相对下降 |
| --- | ---: | ---: | ---: |
| 0,1,2 | 188.963 | 160.430 | 15.10% |
| 2,1,0 | 188.860 | 159.829 | 15.37% |
| 1,2,0 | 188.797 | 160.868 | 14.79% |
| 2,0,1 | 188.848 | 159.733 | 15.42% |
| 0,2,1 | 188.839 | 160.285 | 15.12% |
| 1,0,2 | 188.957 | 160.185 | 15.23% |

六种顺序的 aggregate 效应基本相同，说明 permutation 不会消除现象；关键要看
慢边是否跟随队列位置。

## 2. Trace 结果

六份 repeats=300、全卡 D2H trace 的命令见
[`nsys-commands.txt`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q3-edge-permutation/20260826T111616Z/nsys-commands.txt)，逐条分析 JSON 见
[`trace-analysis/`](../trace-analysis/20260826T111616Z/)。

| permutation | >8 ms P2P | queue position 1 占慢样本 | 慢波数 | duration p99/max (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0,1,2 | 257 | 92.22% | 68 | 13.467 / 13.503 |
| 2,1,0 | 266 | 92.48% | 67 | 13.460 / 13.511 |
| 1,2,0 | 267 | 92.51% | 67 | 13.467 / 13.506 |
| 2,0,1 | 263 | 92.40% | 67 | 13.466 / 13.503 |
| 0,2,1 | 261 | 92.34% | 68 | 13.463 / 13.517 |
| 1,0,2 | 268 | 92.54% | 67 | 13.464 / 13.501 |

六份 trace 都有 4 条 source stream、3720 个 P2P。每份 trace 的慢样本都主要位于
第二队列位置，且没有第一队列位置的慢样本；被标慢的物理 edge 会随着 permutation
中的第二个元素变化。例如 GPU0 的第二个目的地分别为：

| permutation | GPU0 的 source 顺序 | 主要慢 edge |
| --- | --- | --- |
| 0,1,2 | 1 → 2 → 3 | 0→2 |
| 1,2,0 | 2 → 3 → 1 | 0→3 |
| 2,0,1 | 3 → 1 → 2 | 0→1 |

这比“固定某个 GPU pair 总是慢”更符合 queue position/cadence 机制。每个慢波仍通常
包含 4 个 source GPU；它不要求四卡在同一时刻精确启动。

## 3. Q3 判定

Q3 强支持“慢边跟随单 source stream 内的 FIFO 位置”而不是固定物理
source-destination pair。结合 Q2 的 2→3 stream 阈值，当前最简模型是：D2H 背景
存在时，单 source stream 的第二个 eligible copy 容易进入约 13.4 ms 的慢状态；
换序会移动慢边，但不会解除单 stream 的 queue-phase coupling。

这仍不是 CE/NVLink path 的唯一定位，因为 permutation 同时改变了 host 提交顺序和
队列中的 edge 身份；下一步按报告执行 Q4 source offset 扫描，验证完全时间对齐是否
必要以及干扰窗口的量级。
