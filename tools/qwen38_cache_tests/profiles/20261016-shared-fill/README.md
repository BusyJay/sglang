# 2026-10-16 shared expert / fill 优化复查材料

对应代码提交 `05d6feeef7`。两份报告均由用户在线采集，取回后离线转换；
没有为整理材料发起推理、重启服务或更改部署参数。时间为 Asia/Shanghai。

| 文件 | Capture 时间 | 内容 |
| --- | --- | --- |
| [decode.nsys-rep](decode.nsys-rep) | 09:41:14 | 128 个 decode step，统计取 127 个完整区间 |
| [prefill.nsys-rep](prefill.nsys-rep) | 09:51:09 | prefill 尾段及随后约 125 个 decode step；首个 prefill chunk 不完整 |

使用 Nsight Systems 2026.4.1 或兼容的新版本打开。两份 `.nsys-rep` 合计约
43.7 MiB；它们是从原始 `.qdstrm` 导入的完整报告，未裁剪 GPU/API 时间线。
原 raw 文件、派生 SQLite 和完整日志保留在采集者本地，不重复收入 Git。
原 raw 与已提交报告的 SHA256、来源文件名、转换环境见 [metadata.json](metadata.json)。

## Decode 结果

[详细分析](decode-analysis.md)、[kernel/API 明细](decode-analysis.json)、
[三版本比较](decode-comparison.json)。比较中的前两份原报告不在本目录，
比较数字供背景参考，当前 decode 报告可独立复查。

- 整步 18.090 ms；上一版整个 MoE eager 为 22.698 ms，未并发为 17.890 ms。
- GPU 无 kernel/copy 活动从上一版 8.223 降至 1.416 ms/step。
- shared GEMV 为 0.843 ms/step，其中 0.688 ms 与 fill 重叠，约 82%。
- router/merge 恢复到主 graph，另有每层一次 shared graph replay：98 次 graph launch/step。
- 不同输入、route、cache 分配与 miss 数不是受控 A/B，不能将比较视为确定吞吐收益。

## Prefill 报告边界

[prefill-analysis.json](prefill-analysis.json) 包含全 capture、首个 decode planner
之前的区间、以及之后的完整 decode 区间三个摘要。它不是纯 prefill capture，
也没有从请求开始覆盖全部输入 token；不能用整个报告时长计算端到端 prefill 吞吐。

首个 decode planner 前约 14.519 s，其中记录 H2D 约 152.36 GiB、累计
6.049 s；无 GPU kernel/copy 活动约 6.328 s。该区间包含 prefill 尾段和
进入首个 decode planner 前的过渡，不能解释为精确的完整 prefill latency。
CPU/GPU、copy/kernel 可以重叠，累计时间不能直接相加。
随后有 6000 次 planner（约 125 step），可取 124 个完整 step，平均 17.372 ms。
该尾段还有约 1.17 GiB D2H，应结合 sampling/日志复查，不能直接当作前一份
纯稳态 decode 报告的重复测量。本轮没有修改 prefill 路径。

[context.log](context.log) 仅摘录配置、prefill/decode/profiler 时间和选定 cache stats，
帮助对齐时间；没有收入完整请求日志。KV 容量仍为 223616 tokens。

## 复现摘要

在本目录执行（需要已安装 Nsight Systems；导出无需加载模型或使用 GPU）：

```bash
nsys export --type sqlite --output /tmp/qwen38-review-decode.sqlite decode.nsys-rep
nsys export --type sqlite --output /tmp/qwen38-review-prefill.sqlite prefill.nsys-rep
python summarize.py /tmp/qwen38-review-decode.sqlite --output /tmp/qwen38-review-decode.json
python summarize.py /tmp/qwen38-review-prefill.sqlite --output /tmp/qwen38-review-prefill.json
sha256sum --check SHA256SUMS
```

脚本只依赖 Python 标准库，按此模型每步 48 次 planner 划分完整 decode 区间。
其 full/window 摘要与 decode-analysis.json 的 kernel/API 扩展明细结构不同；
同一 127-step 窗口的 wall time 已复现一致。decode-analysis.json 的 GEMV/fill
重叠仍需在 Nsight 时间线上或按 kernel 区间交集独立核验。
