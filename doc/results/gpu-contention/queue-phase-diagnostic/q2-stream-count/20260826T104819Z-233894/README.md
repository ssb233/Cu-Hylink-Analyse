# Q2 stream-count 诊断结果

本目录执行了报告第 8.3 节的 1/2/3 streams-per-source 扫描：4 卡 allpairs、
255M、warmup=10、`streamDependency=none`，每个点 3 次，背景为无背景或四卡
D2H。执行脚本为 [`run_q2_stream_count_matrix.sh`](../../../../../../scripts/run_q2_stream_count_matrix.sh)，完整矩阵见 [`summary.csv`](summary.csv)，环境见 [`environment.txt`](environment.txt)。

## 1. Aggregate 结果

每个 stream 数的 D2H 下降在同一 repeats 下相对无背景均值计算：

| streams/source | repeats | 无背景均值 (GB/s) | 全卡 D2H 均值 (GB/s) | 相对下降 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 178.632 | 127.842 | 28.43% |
| 2 | 20 | 260.476 | 162.320 | 37.68% |
| 3 | 20 | 581.649 | 577.990 | 0.63% |
| 1 | 300 | 188.945 | 160.254 | 15.19% |
| 2 | 300 | 260.516 | 162.653 | 37.57% |
| 3 | 300 | 581.872 | 578.164 | 0.64% |

2-stream 的绝对吞吐位于 1-stream 和 3-stream 之间，但 D2H 下反而比 1-stream
的相对下降更大；因此“增加一个 stream 就足够解除竞争”不成立。只有每个 source
的三条 outgoing edge 都有独立 eligible stream 时，D2H 下降才降到约 0.63%。

## 2. Trace 结果

三份 repeats=300、全卡 D2H trace 的命令见
[`nsys-commands.txt`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q2-stream-count/20260826T105130Z/nsys-commands.txt)，逐条分析 JSON 见
[`trace-analysis/`](../trace-analysis/20260826T105130Z/)。

| streams/source | P2P stream 数 | P2P p50/p90/p99/max (ms) | >8 ms 数量 | 慢波数 |
| ---: | ---: | --- | ---: | ---: |
| 1 | 4 | 5.533 / 5.544 / 13.467 / 13.504 | 259 | 68 |
| 2 | 8 | 5.549 / 13.444 / 13.482 / 13.509 | 1236 | 309 |
| 3 | 12 | 5.586 / 5.603 / 5.603 / 5.604 | 0 | 0 |

2-stream trace 中，慢样本全部落在每个 source 的第三条 edge（source-major 下为
GPU0/1/2 的 `->3` 和 GPU3 的 `->2`），共 1236 个；这条 edge 与第一条 edge
共享同一个 stream，且在该共享 stream 中是 queue position 1。另一条独立 stream
承载的中间 edge 没有出现同样的长尾。每个 2-stream round 都出现 4 个 source GPU
的慢样本，说明它仍是稳定的 wave/cadence 现象，而非偶发单边抖动。

1-stream 的慢样本仍约 92.3% 位于每个 source stream 的第二队列位置；3-stream
没有 >8 ms operation。该结果同时说明：

1. 1→2 的增加没有打破 source 内所有 edge 的队列耦合，剩余共享 stream 中的后继
   edge 仍可持续触发长尾；
2. 2→3 才消除长尾，当前 255M workload 需要覆盖三条 outgoing edge 的独立 eligible
   stream；
3. 这是 stream 数量和 queue layout 的强阈值证据，但仍不能由此确定具体 CE instance、
   NVLink path 或硬件 scheduler。

## 3. 后续

Q2 支持“需要 3 个独立 eligible stream 才能避开当前慢波”的判定。下一步进入 Q3，
在保持单 source 单 stream 的条件下执行 6 种 edge permutation；若慢点跟随 FIFO
位置，则继续支持 queue/HOL；若跟随固定 destination，则需要把重点转向 pair/path
或 engine mapping。
