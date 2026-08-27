# Q2e：D2D=255M、D2H background=16M

本目录是 Q2b 的高背景强度对照。D2D workload 保持 255M，仅将全卡 D2H background
的单次 `cudaMemcpyAsync` 设置为 16M；two-stream assignment、重复次数和其他参数
均保持不变。

固定口径：四卡 allpairs、D2D size=255M、background size=16M、warmup=10、
`streamMode=per-source`、`streamsPerSource=2`、`streamDependency=none`、
`edgeOrder=source-major`；三种 assignment、20/300 repeats、无背景/全卡 D2H，
每个配置 3 次。

## 矩阵结果

`drop` 定义为 `1 - D2H / none`，均值由 `summary.csv` 的 3 次重复计算。

| assignment | 20-repeat clean | 20-repeat D2H | 20-repeat drop | 20-repeat background | 300-repeat clean | 300-repeat D2H | 300-repeat drop | 300-repeat background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 257.356 | 173.486 | 32.59% | 13.335 | 260.456 | 173.464 | 33.40% | 13.035 |
| `1,0,1` | 260.416 | 174.966 | 32.81% | 14.098 | 260.548 | 173.239 | 33.51% | 12.822 |
| `1,1,0` | 260.462 | 173.748 | 33.29% | 13.470 | 260.447 | 173.611 | 33.34% | 12.709 |

单位为 GB/s。36 个 case 全部返回 `status=pass`。第一行 20-repeat clean 含一次约
251 GB/s 的启动样本，但不影响 16M D2H slowdown 的判断；300-repeat clean 均稳定
在约 260.4–260.5 GB/s。

## 与低强度 background 对比

| background size | 300-repeat D2D drop |
| --- | ---: |
| 16K | 0.06%–0.12% |
| 64K | 0.32%–0.45% |
| 16M | 33.34%–33.51% |
| 255M | 约 37.5% |

16M background 在 D2D 并行期间达到约 12.7–13.0 GB/s，已经接近大传输 D2H 的稳定
平台，并重新触发 two-stream 的 aggregate slowdown。结合 16K 和 64K，本轮单独把
强度拐点定位在 64K 与 16M 之间；随后 Q2f 的 8M、Q2g 的 4M 结果将总体可观测区间
进一步收窄到 64K 与 4M 之间，尚不能由这些离散点确定更窄的阈值。

三种 assignment 的下降幅度相近，说明在高 D2H 强度下，具体哪一条 source-major
edge 被独立放置仍不是主要变量。该矩阵没有采集 Nsight Systems trace，因此本结果
确认的是 aggregate D2D slowdown，不能单独给出每条 P2P 慢样本的 queue-position
分布。

## 可复现命令

```bash
bash scripts/run_stream_assignment_matrix.sh \
  --d2dSize=255M --backgroundSize=16M
```

原始逐 case 日志、JSON、background 输出和环境快照均在本目录：

- [`summary.csv`](summary.csv)
- [`environment.txt`](environment.txt)
- 各 assignment/repeats/repetition/scenario 子目录中的 `command.txt`、`d2d.json`、
  `background.json` 和日志
