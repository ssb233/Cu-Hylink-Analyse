# D2H background memcpy size sweep

本实验用于为当前 two-stream allpairs 消融选择 D2H 背景的单次
`cudaMemcpyAsync` 大小，尚未运行正式 D2D 矩阵。背景程序对每个 GPU 使用一个 pinned
host buffer、一个 device buffer 和一个 stream，每次 D2H 后立即
`cudaStreamSynchronize`；warmup 固定为 10 次。

硬件为 4× Tesla V100-SXM2-32GB，当前驱动报告 CUDA 13.0。四卡测试使用
`--devList=0,1,2,3 --direction=d2h`，报告周期为 1 秒；每个重复运行约 3 秒。

## 四卡结果

重复均值见 [`summary.csv`](summary.csv)。单次 sweep 还观察到：4K 约 1.9 GB/s，16K
约 6.5 GB/s，64K 约 11.8 GB/s，256K 及以上约 13.1 GB/s 的 steady window；大于
256K 后没有继续明显增加。历史 255M/512M 测试的首个窗口会因启动和时钟状态出现
短暂高值，因此这里用重复测试的稳定窗口判断拐点，而不是用单次 total aggregate
的瞬时高值。

## 单 GPU0 对照

单 GPU0 的重复均值为：4K=0.485 GB/s、16K=1.797 GB/s、64K=4.947 GB/s、
256K=9.417 GB/s、1M=11.891 GB/s；已有大传输测试约为 13 GB/s。由此可见，64K
在单 GPU 路径上仍明显无法达到大传输的 CE/主机链路平台，256K 也尚未达到单 GPU
的大传输平台；四卡 aggregate 的 256K 平台主要反映当前四卡共享主机路径的上限，
不能直接当作单卡 CE 峰值。

## 选择建议

下一轮保持当前 two-stream D2D 设置，仅把 D2H background 的 `--size` 改为：

```text
首选：64K
```

理由是：

1. 它比 4K/16K 有更高的有效 payload，不容易把结果主要变成固定的 driver/同步开销；
2. 它在单 GPU0 上约 4.95 GB/s，相对大传输约 13 GB/s 明显未饱和；
3. 它仍保持连续 D2H 操作，四卡 aggregate 约 11.38 GB/s，不会把背景流量降到几乎
   不存在。

如果首选结果仍需要更强的“非 CE 饱和”对照，再增加：

```text
低压对照：16K
```

16K 的单 GPU0 只有约 1.80 GB/s，但固定开销占比更高，不建议直接用它替代 64K 作为
唯一正式点。4K 不建议作为第一选择，因为它已明显进入小传输/操作频率受限区间，
会同时改变背景数据量和单次操作粒度。

## 结论边界

这里的“饱和”是根据本机 `host_copy_background` 的可观测传输平台判断，不是直接读取
CE 硬件 busy counter。D2H 实际路径仍包含 GPU HBM 读取、copy engine、PCIe/主机内存
和同步开销；因此正式实验应将 64K 结果解释为“显著低于大传输背景强度”，而不是声称
CE 利用率精确为某个百分比。
