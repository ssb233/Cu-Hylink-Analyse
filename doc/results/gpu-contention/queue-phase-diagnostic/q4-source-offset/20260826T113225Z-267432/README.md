# Q4 source-phase-offset matrix

本目录是 Q4 的无 profiler 结果。固定口径为 4 卡 allpairs、255M、warmup=10、
`streamsPerSource=1`、`streamDependency=none`、source-major edge order；每个 source
只在 measured batch 前执行一次 device-side `%globaltimer` delay kernel。每个点 3 次，
背景为无背景或四卡 D2H。

## 矩阵均值

`drop` 定义为 `1 - D2H / none`。均值由 `summary.csv` 的 3 次重复计算。

| sourceOffsetsUs (GPU0,1,2,3) | repeats | 无背景 GB/s | 全卡 D2H GB/s | drop |
| --- | ---: | ---: | ---: | ---: |
| 0,0,0,0 | 20 | 178.678 | 127.865 | 28.44% |
| 0,250,500,750 | 20 | 179.719 | 127.833 | 28.87% |
| 0,500,1000,1500 | 20 | 179.739 | 127.889 | 28.85% |
| 0,1000,2000,3000 | 20 | 179.807 | 127.918 | 28.86% |
| 0,2000,4000,6000 | 20 | 179.780 | 127.843 | 28.89% |
| 0,4000,8000,12000 | 20 | 179.820 | 127.764 | 28.95% |
| 0,0,0,0 | 300 | 188.862 | 160.372 | 15.09% |
| 0,250,500,750 | 300 | 189.896 | 165.790 | 12.69% |
| 0,500,1000,1500 | 300 | 187.674 | 158.147 | 15.73% |
| 0,1000,2000,3000 | 300 | 186.371 | 152.544 | 18.15% |
| 0,2000,4000,6000 | 300 | 185.105 | 149.044 | 19.48% |
| 0,4000,8000,12000 | 300 | 181.426 | 137.207 | 24.37% |

20-repeat 的下降稳定在约 28.4%–29.0%，没有因为一次性 source offset 而消失。
300-repeat 的 paired drop 在 12.7%–24.4% 之间波动，且大 offset 的 clean 本身也
下降；这说明一次性启动偏移改变了长批次的初始相位/恢复过程，但没有建立一个持续
的、可控的跨 GPU phase separation。

## Trace 校验

Nsight Systems trace 位于
[`q4 trace`](../../../../../traces/gpu-contention/queue-phase-diagnostic/q4-source-offset/20260826T113959Z/)，
SQLite 和逐条 JSON 分析位于
[`trace analysis`](../trace-analysis/20260826T113959Z/)。六份 trace 均包含 3720 个
P2P operation 和 4 条 source stream；分析器另外导出了
`sourceOffsetDelayKernels` 以及第一条 measured P2P 的 source-relative start。

| case | delay kernel duration (GPU1/GPU2/GPU3, ms) | 第一条 measured P2P 的 source-relative start (GPU0/GPU1/GPU2/GPU3, ms) | >8 ms P2P | 慢波数 |
| --- | --- | --- | ---: | ---: |
| 0,0,0,0 | — | 0.000 / 0.342 / 0.599 / 0.000 | 260 | 68 |
| 0,250,500,750 | 0.251 / 0.502 / 0.751 | 0.828 / 0.000 / 0.245 / 0.739 | 250 | 66 |
| 0,500,1000,1500 | 0.501 / 1.001 / 1.501 | 0.742 / 0.000 / 0.751 / 1.483 | 257 | 70 |
| 0,1000,2000,3000 | 1.002 / 2.001 / 3.001 | 0.216 / 0.000 / 1.489 / 2.784 | 260 | 78 |
| 0,2000,4000,6000 | 2.001 / 4.001 / 6.001 | 0.000 / 1.162 / 3.403 / 5.653 | 338 | 112 |
| 0,4000,8000,12000 | 4.001 / 8.001 / 12.001 | 0.000 / 3.498 / 7.692 / 11.925 | 441 | 165 |

delay kernel 的实际持续时间与目标值一致，证明偏移确实在设备端执行。第一条 P2P
的绝对起点不严格等于目标向量：warmup 完成时间、host enqueue 顺序以及 source0
没有 delay 都会贡献初始 jitter。因此这里应使用 trace 中的 delay kernel duration
验证“延迟被执行”，使用第一条 P2P start 验证“实际启动 span”，不能把两者直接等同。

所有 case 的慢样本仍集中在每个 source stream 的第二 queue position：约 92.0%–95.5%，
第一 position 没有慢样本。增大一次性 offset 没有消除慢波，反而在 2–12ms offset
下慢样本和慢波数量增加；因此当前证据不支持“只有四个 GPU 的 P2P 精确同时启动才
会出现问题”。它更支持 source 内 FIFO/cadence 以及 D2H 作用下的内部 queue/engine
状态，而不是一次性跨 GPU 起始相位本身。

按分析器的 round index（排除 10 个 warmup round）统计，round 内全部 P2P 的 start
span（它同时包含执行时长差和 phase drift，不是纯 source-start offset）如下：

| case | measured-round startSpan p50 / p90 / max (ms) |
| --- | ---: |
| 0,0,0,0 | 52.344 / 52.687 / 52.983 |
| 0,250,500,750 | 67.419 / 69.261 / 70.407 |
| 0,500,1000,1500 | 116.995 / 119.273 / 120.033 |
| 0,1000,2000,3000 | 197.137 / 198.535 / 198.816 |
| 0,2000,4000,6000 | 434.058 / 434.923 / 435.391 |
| 0,4000,8000,12000 | 860.915 / 910.890 / 911.187 |

这说明一次性 offset 会改变后续 round 的相对 cadence，而且偏移越大，round-level
spread 越大；它没有把系统带入“错峰后所有 edge 都恢复到 per-edge 高性能”的状态。

注意：Q4 只注入一次启动偏移，后续每个 round 没有重复注入 offset；因此不能由此
推导一个 steady-state 的 phase-tolerance 窗口。若需要测量持续错峰阈值，应增加每
round 的 device-side gate/barrier，这会引入新的同步变量，应作为后续独立实验。
