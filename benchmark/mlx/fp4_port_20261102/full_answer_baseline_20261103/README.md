# NVIDIA GLM MLX 完整回答与吞吐基准

代码版本：`50f587d8bb4f4c504bd185883756ba2af3466921`。Checkpoint：`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`。设备：M5 Ultra 256 GiB / MLX 0.32.4。原生 MTP2；完整配置见 results.json。

Four explicit tasks, short/2050-token contexts, first request plus two warm repetitions. Greedy native MTP2. Task-specific checks do not establish general accuracy. TTFT is observed first SSE token; decode throughput excludes time before that token. Cold means flushed prefix cache, not cold weight pages.

每组四题，分别为清空前缀缓存后的请求及两次复用缓存的请求。生成上限 2048 token，正常 EOS 才算完整回答。短/2050 上下文的 padding 和实际输入 token 全部保存；2050 背景文本是受控合成上下文。

| 上下文 | 完整结束 | 任务检查通过 | 稳态 decode 中位数 tok/s | 稳态范围 tok/s | 稳态 TTFT 中位数 s | 重复输出一致 |
|---|---:|---:|---:|---:|---:|---|
| short | 12/12 | 12/12 | 39.139 | 35.295–41.186 | 0.264 | True |
| 2050 | 12/12 | 12/12 | 32.102 | 18.505–35.380 | 0.077 | True |

两路并发完整回答端到端合计吞吐：36.416 tok/s；该指标包含 prefill，与单请求 decode 吞吐不同。

API reasoning_content/content 完整分离检查：True。

算术/代数检查最终数值；Fibonacci 在有超时限制的子进程执行 n=0/1/2/10；中文题自动关键词 rubric 只检查覆盖，不能作为通用准确率。回答全文、reasoning、token ID、结束原因、原始 SSE 时间、缓存元数据与校验过程在 results.json。

本轮只有原生 MTP2 完整回答吞吐，未把先前 128-token AR 对照当作完整回答 A/B。首次请求含当前进程按形状编译成本；清空前缀缓存不清除权重 page cache。服务已在此次提交前启动，最后源码仅格式整理与删除未使用局部变量；AST/字面量等价证据及最终设备测试见 validation.json。

复现：

```sh
MLX_ENABLE_TF32=0 python benchmark/mlx/fp4_port_20261102/full_answer_baseline.py \
  --checkpoint /path/to/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
  --base-url http://127.0.0.1:30007 \
  --output-dir benchmark/mlx/fp4_port_20261102/full_answer_candidate
```

生成优化候选时直接比较保存的回答/token 和任务检查，保留新耗时，不必重跑已确认的参考输出。
