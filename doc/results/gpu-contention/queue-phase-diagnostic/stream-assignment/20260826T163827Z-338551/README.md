# Q2f：D2D=255M、D2H background=8M

本目录是 Q2b 的低一些背景强度对照。D2D workload 保持 255M，仅将全卡 D2H
background 的单次 `cudaMemcpyAsync` 设置为 8M；two-stream assignment、重复次数和
其他参数均保持不变。

固定口径：四卡 allpairs、D2D size=255M、background size=8M、warmup=10、
`streamMode=per-source`、`streamsPerSource=2`、`streamDependency=none`、
`edgeOrder=source-major`；三种 assignment、20/300 repeats、无背景/全卡 D2H，
每个配置 3 次。

## 矩阵结果

`drop` 定义为 `1 - D2H / none`，均值由 `summary.csv` 的 3 次重复计算。

| assignment | 20-repeat clean | 20-repeat D2H | 20-repeat drop | 20-repeat background | 300-repeat clean | 300-repeat D2H | 300-repeat drop | 300-repeat background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 256.113 | 178.974 | 30.12% | 13.264 | 260.432 | 179.960 | 30.90% | 12.860 |
| `1,0,1` | 260.475 | 180.783 | 30.59% | 13.681 | 260.497 | 179.558 | 31.07% | 12.780 |
| `1,1,0` | 260.494 | 179.583 | 31.06% | 13.141 | 260.433 | 179.907 | 30.92% | 12.820 |

单位为 GB/s。36 个 case 全部返回 `status=pass`。第一行 20-repeat clean 含一次约
247 GB/s 的启动样本；其余 clean 样本约为 260.4–260.5 GB/s。300-repeat 的三种
assignment 均复现约 31% 的 D2D slowdown。

## 与其他 background 强度对比

| background size | 300-repeat D2D drop |
| --- | ---: |
| 16K | 0.06%–0.12% |
| 64K | 0.32%–0.45% |
| 8M | 30.90%–31.07% |
| 16M | 33.34%–33.51% |
| 255M | 约 37.5% |

8M background 在 D2D 并行期间达到约 12.78–12.86 GB/s，已经进入明显竞争区间，且
相较 16M 的约 33.3% slowdown 略弱。结合 64K 仍基本无影响、8M 已明显下降，本轮
单独把强度转折收窄到 64K 与 8M 之间；随后 Q2g 的 4M 结果仍然明显受影响，将总体
可观测区间进一步收窄到 64K 与 4M 之间，但这不是精确阈值。

这里的 background worker 每次提交一个 D2H 后执行 `cudaStreamSynchronize`，因此改变
单次 size 同时改变了单次传输量、D2H 数据率和 operation cadence。不能把 8M/64K 之间
的区间直接解释为单一 CE 字节带宽阈值。该矩阵未采集 Nsight Systems trace，因此本
结果确认的是 aggregate D2D slowdown，不单独定位具体慢 P2P 的 queue position。

## 可复现命令

```bash
bash scripts/run_stream_assignment_matrix.sh \
  --d2dSize=255M --backgroundSize=8M
```

原始逐 case 日志、JSON、background 输出和环境快照均在本目录：

- [`summary.csv`](summary.csv)
- [`environment.txt`](environment.txt)
- 各 assignment/repeats/repetition/scenario 子目录中的 `command.txt`、`d2d.json`、
  `background.json` 和日志
