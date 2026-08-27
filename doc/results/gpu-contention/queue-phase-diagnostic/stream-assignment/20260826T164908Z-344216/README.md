# Q2g：D2D=255M、D2H background=4M

本目录是 Q2f 的进一步低强度对照。D2D workload 保持 255M，仅将全卡 D2H background
的单次 `cudaMemcpyAsync` 设置为 4M；two-stream assignment、重复次数和其他参数
均保持不变。本轮只执行 4M，不执行 1M。

固定口径：四卡 allpairs、D2D size=255M、background size=4M、warmup=10、
`streamMode=per-source`、`streamsPerSource=2`、`streamDependency=none`、
`edgeOrder=source-major`；三种 assignment、20/300 repeats、无背景/全卡 D2H，
每个配置 3 次。

## 矩阵结果

`drop` 定义为 `1 - D2H / none`，均值由 `summary.csv` 的 3 次重复计算。

| assignment | 20-repeat clean | 20-repeat D2H | 20-repeat drop | 20-repeat background | 300-repeat clean | 300-repeat D2H | 300-repeat drop | 300-repeat background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 256.820 | 186.379 | 27.43% | 13.081 | 260.460 | 184.871 | 29.02% | 12.786 |
| `1,0,1` | 260.475 | 186.666 | 28.34% | 13.478 | 260.525 | 184.572 | 29.15% | 12.853 |
| `1,1,0` | 260.504 | 184.984 | 28.99% | 13.070 | 260.466 | 184.707 | 29.09% | 12.953 |

单位为 GB/s。36 个 case 全部返回 `status=pass`。第一行 20-repeat clean 均值包含一次
约 249.6 GB/s 的启动样本；300-repeat clean 均稳定在约 260.46–260.52 GB/s。
300-repeat 的 D2D slowdown 为 29.02%–29.15%，三种 assignment 结果接近。

## 与其他 background 强度对比

| background size | 300-repeat D2D drop |
| --- | ---: |
| 16K | 0.06%–0.12% |
| 64K | 0.32%–0.45% |
| 4M | 29.02%–29.15% |
| 8M | 30.90%–31.07% |
| 16M | 33.34%–33.51% |
| 255M | 约 37.5% |

4M background 在 D2D 并行期间约为 12.79–12.95 GB/s，仍然处于明显竞争区间；相较
8M 和 16M，D2D slowdown 略有恢复，但没有回到 64K 的 clean 水平。因此当前离散
实验把明显竞争的强度转折保持在 64K 与 4M 之间；由于未执行 1M，不能继续向下收窄。

background worker 每次提交一个 D2H 后执行 `cudaStreamSynchronize`，改变单次 size
同时改变了 D2H 数据率和 operation cadence。该结果不能直接解释为单一 CE 字节带宽
阈值。本轮未采集 Nsight Systems trace，因此结论限于 aggregate D2D slowdown。

## 可复现命令

```bash
bash scripts/run_stream_assignment_matrix.sh \
  --d2dSize=255M --backgroundSize=4M
```

原始逐 case 日志、JSON、background 输出和环境快照均在本目录：

- [`summary.csv`](summary.csv)
- [`environment.txt`](environment.txt)
- 各 assignment/repeats/repetition/scenario 子目录中的 `command.txt`、`d2d.json`、
  `background.json` 和日志

