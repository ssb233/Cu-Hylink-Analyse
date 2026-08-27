# Q2b two-stream copy-assignment matrix

本目录是 Q2b 的无 profiler 消融矩阵。实验固定为四卡 allpairs、255M、warmup=10、
`streamMode=per-source`、`streamsPerSource=2`、`streamDependency=none`、
`edgeOrder=source-major`；每个配置 3 次，背景为无背景或四卡 D2H。

## Assignment 定义

assignment 按每个 source 的三个 source-major outgoing edge 位置编号，值表示 stream
slot：

| assignment | 独立 edge 位置 | 共享 stream 上的两个 edge |
| --- | ---: | --- |
| `0,1,1` | 0 | 1、2 |
| `1,0,1` | 1 | 0、2 |
| `1,1,0` | 2 | 0、1 |

因此三种配置都保持“每个 source 两条 stream”，只改变哪一条 edge 被单独放置。

## 矩阵均值

`drop` 定义为 `1 - D2H / none`，均值由 `summary.csv` 的 3 次重复计算。

| assignment | 独立 edge 位置 | 20-repeat 无背景 | 20-repeat 全卡 D2H | 20-repeat drop | 300-repeat 无背景 | 300-repeat 全卡 D2H | 300-repeat drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 0 | 260.492 | 162.686 | 37.55% | 260.422 | 162.674 | 37.53% |
| `1,0,1` | 1 | 260.455 | 162.397 | 37.65% | 260.495 | 162.620 | 37.57% |
| `1,1,0` | 2 | 260.482 | 162.350 | 37.67% | 260.421 | 162.647 | 37.54% |

单位为 GB/s。三种 assignment 的 clean 带宽约 260.4 GB/s，D2H 下约 162.4–162.7
GB/s；改变独立 edge 的位置没有带来有意义的 aggregate 差异。20 与 300 repeats
也几乎一致，说明在这个两 stream 配置下，D2H 干扰不会像三 stream 情况那样在长批次
后恢复。

## Trace 证据

三份 300-repeat、全卡 D2H trace 的原始 `.nsys-rep` 和导出的 SQLite 位于
[`trace 目录`](../../../../../traces/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154647Z/)，
逐条统计 JSON 位于
[`trace analysis`](../trace-analysis/20260826T154647Z/)。对应命令记录在 trace 目录的
`nsys-commands.txt`。

| assignment | stream 数 | 全部 P2P | >8 ms P2P（含 warmup） | measured >8 ms | 慢波数 | 慢样本在 queue position 1 的比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 8 | 3720 | 1236 | 1200 | 309 | 100% |
| `1,0,1` | 8 | 3720 | 1236 | 1200 | 309 | 100% |
| `1,1,0` | 8 | 3720 | 1236 | 1200 | 309 | 100% |

这里的 `queue position 1` 是实际 CUDA stream 内的第二个 operation，不是三个
source-major edge 位置本身。慢 edge 随共享 stream 的第二个 operation 改变：

| assignment | trace 中的慢 edge |
| --- | --- |
| `0,1,1` | `0->3`、`1->3`、`2->3`、`3->2` |
| `1,0,1` | `0->3`、`1->3`、`2->3`、`3->2` |
| `1,1,0` | `0->2`、`1->2`、`2->1`、`3->1` |

这说明慢点不是固定的 source-destination pair：被单独隔离的 edge 没有出现该长尾，
而共享 stream 中排在第二位的 edge 出现慢长尾。仅把一个 copy 从共享 stream 中移出，
仍然保留了另一个两-copy FIFO 队列，因此不能消除竞争；当前 workload 需要三条独立
eligible stream 才能进入此前观察到的无长尾区间。

## 可复现实验

完整 36 个 case 的逐 case command、日志、JSON、后台流量结果和环境快照都在本目录；
入口脚本为：

```bash
bash scripts/run_stream_assignment_matrix.sh
```

trace 采集命令见：

```text
doc/traces/gpu-contention/queue-phase-diagnostic/stream-assignment/20260826T154647Z/nsys-commands.txt
```
