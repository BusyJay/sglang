# Mailbox 优化与部署记录（2026-10-15）

> 本文记录初版双映射部署；随后已修复为单份主机源权重。当前实现与内存结果见
> [MAILBOX_SINGLE_MAPPING_20261015.md](MAILBOX_SINGLE_MAPPING_20261015.md)。下文初版额外内存数据保留作为历史对照。

采用 `01-direct-mmap.patch`，并修复 RTX 5090 D 上的 CUDA host registration
兼容性与默认设备问题。没有叠加其余三个互斥 patch。方案 1 在本轮负载中
吞吐最高，并保留 all-hit 无需 CPU 确认的路径及原有 NVFP4 计算方式。

## 来源和环境

- 原始仓库基线：`44ed74ebd519621b821fbaec34f6faf0890c44f1`，已经是包要求的 mailbox 基线，无需重放 mbox。
- 输入包：`mailbox-optimizations-20261015-v2 2.tar.gz`。
- 包 SHA256：`2412934093837275ea72e403ae0c2468c08983286e021df6509d7d88571fd882`；包内 SHA256SUMS 全部通过。
- 采用的已测提交：`6398164a32`；原 patch 提交及兼容修复均保留在历史中。
- GPU：RTX 5090 D 32 GiB（SM120），驱动 615.78.08，CUDA toolkit 13.0.88。
- 模型：`nvidia/Qwen3.8-Flash-Next-NVFP4`，revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`，使用现有离线 checkpoint 和 expert pack。
- 镜像：`swallow-sglang:v0.5.21-qwen38-capacity`。
- 保留当前配置：MTP off、5140 总专家槽位、adaptive cache、lazy PLE 3072 MiB、FP32 GDN、FP8 KV、packed prefill/decode、breakable bs=1、prefill chunk=8192、mem fraction=0.91、单请求、HiCache ratio=6。

## 修复

原包四个实现都在只读 host registration 处返回 CUDA 801；线上查询
`cudaDevAttrHostRegisterReadOnlySupported=0`，确认不支持该注册标志。采用的方案使用
独立 `mmap.ACCESS_COPY` 私有可写映射，移除 `cudaHostRegisterReadOnly`，保留
`cudaHostRegisterMapped` 供 GPU 直接读取。原文件和 pack reader 的只读映射
保持不变。CUDA pinning 会物化私有 COW 页，必须为这部分主机内存准备预算。
初始化异常和正常关闭都会释放 tensor view 和私有映射；关闭前先排空 GPU reader。

初始化控制缓冲区显式使用 CPU 默认设备，解决模型加载在 CUDA 默认设备
上下文中时 pinned buffer 被错误分配到 CUDA 的问题。新增 GPU 专项测试
验证该上下文、文件不变以及映射关闭。

方案 2/3 在独立实验分支还修复了 batch API 拒绝 legacy NULL stream 的问题：
NULL stream 回退到有序 `cudaMemcpyAsync`，显式 stream 保留 batch DMA；CPU
描述符刷新时也显式指定 CPU。这些实现未合入当前部署。
CUDA API 约束见 [NVIDIA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)。

## 测量口径

每组使用同一份 `inputs.json`：512 / 4096 token 输入，temperature=0，固定输出
768 token，ignore_eos；每种长度一次预热，三轮正式测量，交替输入长度。
组顺序为基线 A1 → 1 → 2 → 3 → 4 → 基线 A2，均在独立容器内串行运行。
每次先清除 KV cache，并确认 HTTP 成功及输出 token 数正确。启动脚本来自
当前真实部署，SHA256 `eb4b0a0941758740718c4f2be8b7986298d059cda07a0f4b2e52cdd152607109`。
测试容器使用 IPC host 和 unlimited memlock；统计日志频率统一提高到 1920 层
调用 / decode 64 token，其他模型配置保持一致。

adaptive cache 在每次 prefill 后的前 512 个 decode token 采样，采样期间绕过
mailbox。最初的 256 token 试测因此不用于性能结论。正式测试分别统计：

- 整段 decode：首个与最后一个 token 的客户端接收时间，包含采样与重平衡开销。
- mailbox 段：输出位置 513–768 的客户端接收时间；每个请求都检查 mailbox 调用增量至少 8000，确保实际覆盖优化路径。
- p50/p95：各请求 mailbox 段 token 到达间隔的分位数，再取三次正式请求中位数。实际流均无合并 token chunk。
- 吞吐为三轮正式请求中位数；基线汇总增益使用前后两组共六个样本的中位数。

下表双数值依次对应 512 / 4096 token 输入；吞吐单位 tok/s，间隔单位 ms。
启动时间仅一次观察，包含当时的 JIT 和文件缓存状态，不能视为严格冷启动对照。

| 方案 | mailbox 段吞吐 | 整段 decode 吞吐 | mailbox 间隔 p50 | mailbox 间隔 p95 | 启动秒 |
| --- | --- | --- | --- | --- | --- |
| 基线 A1 | 40.35 / 42.04 | 36.27 / 37.00 | 22.93 / 22.33 | 42.56 / 38.01 | 46.0 |
| 1：direct mmap（采用） | 56.10 / 57.04 | 38.55 / 39.95 | 17.51 / 16.46 | 24.72 / 26.08 | 94.1 |
| 2：batch DMA | 51.88 / 52.93 | 36.56 / 37.17 | 18.97 / 18.65 | 22.15 / 22.15 | 87.1 |
| 3：compute pipeline | 34.46 / 31.92 | 32.90 / 32.86 | 26.99 / 29.70 | 46.03 / 45.78 | 64.1 |
| 4：single miss | 42.22 / 42.95 | 33.80 / 37.04 | 21.66 / 21.04 | 40.10 / 38.40 | 66.1 |
| 基线 A2 | 40.15 / 42.99 | 36.25 / 37.79 | 22.74 / 21.87 | 39.52 / 38.30 | 48.0 |

相对合并基线：

- 输入 512 token：mailbox 段 40.25 → 56.10 tok/s（+39.41%）；整段 decode 36.26 → 38.55 tok/s（+6.32%）；mailbox token 间隔 p95 40.19 → 24.72 ms。
- 输入 4096 token：mailbox 段 42.51 → 57.04 tok/s（+34.18%）；整段 decode 37.39 → 39.95 tok/s（+6.84%）；mailbox token 间隔 p95 38.15 → 26.08 ms。

方案 2 的 token 间隔尾部更平稳，但整段吞吐收益小于方案 1。方案 3 在该负载
出现吞吐回退，未部署，也未进一步做 Nsight 重叠证明或分组 NVFP4 质量评估。
方案 4 的收益较小。即使 temperature=0，基线自身多轮输出 ID 也不完全相同，
因此不把完整文本逐 token 相同作为本轮验收条件。

## 主机内存和适用范围

expert pack 约 63.28 GiB，需整份注册。基线从启动前 176.19 GiB MemAvailable
降至 ready 后 143.76 GiB；方案 1 从 176.53 降至 80.29 GiB，净增约 63.8 GiB
主机内存占用。CUDA 锁页通常不会完整体现在 `/proc` 的 VmLck/Mlocked 数字中。
普通 transport 的预取与 PLE / HiCache 内存仍保留。

本轮输出 768 token 时整段收益约 6–7%；不超过 512 token 的回复主要走采样
路径，不能套用 mailbox 段约 34–39% 的增益。没有验证多并发或长上下文边界。

## 验证与原始证据

- 方案 1：CPU 233 项通过；完整 GPU 回归 47 项通过，无跳过。
- 方案 2：CPU 233 项通过；完整 GPU 回归 47 项通过，无跳过。
- 方案 3：CPU 235 项通过；完整 GPU 回归 49 项通过，无跳过。
- 方案 4：CPU 232 项通过；完整 GPU 回归 48 项通过，无跳过。
- 合入主工作区后，再运行新增注册保护测试和 mailbox byte 测试，共 2 项通过。
- 四组和前后基线的完整模型请求均成功；各组独立算术 smoke 请求返回 `323`。该短 canary 不覆盖采样后的 pipeline 数值质量。
- 主工作区原有 MTP.md 未提交改动的 SHA256 在应用前后保持一致。

原始数据位于 `/home/jay/workspace/sglang/results/mailbox-optimization-20261015/`：
`benchmark.py`、`serve.sh`、`inputs.json`、`summary.json`、`validation-summary.json`、
各组 `report.json` / `server.log` / 请求流。原包失败日志保留为 `gpu-original.*`，
2/3 的 NULL stream 修复前记录保留为 `gpu-before-stream-fix.*`。
`baseline-sampling-only/` 和较早的启动/统计脚本调试数据不参与上表。

独立实验分支为 `codex/mailbox-opt1-20261015` 至 `codex/mailbox-opt4-20261015`；
主分支只 fast-forward 到方案 1 的已测提交。部署前源码备份分支为
`codex/mailbox-before-20261015`，指向原始 `44ed74ebd5`。

## 线上验收

服务通过现有 compose 的 `llm` 启动，容器 `qwen3.8-nvfp4-llm`，接口
`http://127.0.0.1:8329`。使用当前启动脚本和源码挂载，无需重建镜像。
实际启动后 54.03 秒 `/ready` 返回 200；本次使用已有 JIT cache，
比首次候选测量的 94.1 秒短。`/server_info` 确认上述配置生效。线上算术返回
`323`，固定输出 1024 token 请求完整成功，周期日志确认 mailbox 调用增量
12,815。请求后 `/ready` 仍返回 200；容器无重启、无 OOM，
主机 MemAvailable 约 80 GiB。

原始记录见 `production/ready.json`、`server-info.json`、`validation.json`、
`request-1024.json`、`server.log`、`focused-gpu.log` 和 `cuda-capabilities.json`。
验收只启动 LLM 服务，用户暂停的业务负载由用户自行恢复。

回滚（仓库根目录，先处理之后新增的代码改动）：

```bash
docker compose -f /home/jay/workspace/prod/homelab-llm/docker-compose.yml stop llm
git switch codex/mailbox-before-20261015
docker compose -f /home/jay/workspace/prod/homelab-llm/docker-compose.yml up -d --no-deps llm
```
