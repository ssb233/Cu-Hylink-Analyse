# Q2d：D2D=255M、D2H background=64K

本目录是 Q2b 的第二个低背景强度对照。D2D workload 保持 255M，仅将全卡 D2H
background 的单次 `cudaMemcpyAsync` 设置为 64K；two-stream assignment、重复次数和
其他参数均保持不变。

固定口径：四卡 allpairs、D2D size=255M、background size=64K、warmup=10、
`streamMode=per-source`、`streamsPerSource=2`、`streamDependency=none`、
`edgeOrder=source-major`；三种 assignment、20/300 repeats、无背景/全卡 D2H，
每个配置 3 次。

## 矩阵结果

`drop` 定义为 `1 - D2H / none`，均值由 `summary.csv` 的 3 次重复计算。负值表示
D2H 均值略高于 clean 均值，属于重复运行波动。

| assignment | 20-repeat clean | 20-repeat D2H | 20-repeat drop | 20-repeat background | 300-repeat clean | 300-repeat D2H | 300-repeat drop | 300-repeat background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0,1,1` | 257.044 | 259.467 | -0.94% | 10.312 | 260.435 | 259.598 | 0.32% | 7.354 |
| `1,0,1` | 260.435 | 259.446 | 0.38% | 11.533 | 260.482 | 259.432 | 0.40% | 7.350 |
| `1,1,0` | 260.499 | 259.294 | 0.46% | 10.560 | 260.455 | 259.276 | 0.45% | 7.743 |

单位为 GB/s。36 个 case 全部返回 `status=pass`。第一行 20-repeat clean 同样包含
一次启动阶段的约 250 GB/s 样本；其余 clean 和 D2H 结果均稳定在约 259–260 GB/s。

## 与其他 D2H size 对比

在相同 D2D=255M、two-stream assignment 口径下：

| background size | 300-repeat D2D drop |
| --- | ---: |
| 255M | 37.53%–37.57% |
| 64K | 0.32%–0.45% |
| 16K | 0.06%–0.12% |

64K 的背景 D2H 已达到约 7.35–7.74 GB/s（D2D 并行期间测得），但仍没有复现 255M
background 下的 two-stream slowdown。当前证据说明触发条件位于 64K 与 255M 之间，
或者单次 D2H 的传输粒度/同步 cadence 本身也参与了触发；不能只用总 D2H GB/s 直接
等同于 CE busy。

本轮没有采集 Nsight Systems trace，因此结论限于 aggregate D2D 性能：64K 背景下
竞争没有体现在总带宽下降上，不能单独证明每条 P2P 的长尾完全消失。

## 可复现命令

```bash
bash scripts/run_stream_assignment_matrix.sh \
  --d2dSize=255M --backgroundSize=64K
```

原始逐 case 日志、JSON、background 输出和环境快照均在本目录：

- [`summary.csv`](summary.csv)
- [`environment.txt`](environment.txt)
- 各 assignment/repeats/repetition/scenario 子目录中的 `command.txt`、`d2d.json`、
  `background.json` 和日志
