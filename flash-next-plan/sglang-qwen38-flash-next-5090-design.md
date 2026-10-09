> Historical scaffold, retained for reference. Apply both patches first and use
> `flash-next-plan/README-ARCHIVE.md`, `tools/qwen38_cache_tests/RUN_ON_GPU.md`,
> and `flash-next-plan/sglang-gpu-switch-workflow/README.md` for current commands.
> The old docker-job/client-canary templates are incompatible with the current
> canary and are not supported test entry points.

# SGLang v0.5.21：5090运行Qwen3.8-Flash-Next-NVFP4

日期：2026-10-09  
基线：SGLang `v0.5.21`，commit `e00930c5489053f26d86b179cee0d087f846acbb`  
硬件：RTX 5090 32GB、约180GB RAM；RTX 4080 SUPER用于小型GPU测试  
模型：`nvidia/Qwen3.8-Flash-Next-NVFP4`  
目标：纯文本、单请求、FP8 KV、最高约220K上下文

## 1. 已确认事实

Checkpoint约123.53GiB：

| 部分 | 大小 |
|---|---:|
| Routed experts | 63.28GiB |
| PLE/N-gram表 | 47.75GiB |
| Dense/shared等 | 9.16GiB |
| MTP | 2.51GiB |
| Vision | 0.84GiB，第一阶段不加载 |

模型有48层，每层512个routed experts，top-k=10，hidden=2560，MoE intermediate=640。单expert平均约2.64MiB。

标准SGLang会把routed experts常驻GPU，32GB 5090无法直接运行。解决范围只限routed experts；QSA、GDN、PLE、MTP、API和调度继续复用SGLang。

## 2. 运行架构

```text
GPU常驻：dense/shared + MTP + FP8 KV + QSA/GDN状态
RAM：PLE + 63.28GiB runtime expert pack
Decode：每层GPU热专家cache
Prefill：单层512-expert arena，逐层复用
```

第一阶段加 `--language-model-only`，并将官方checkpoint实际架构 `Qwen4ExpForConditionalGeneration` 加入v0.5.21的 `LANGUAGE_MODEL_ONLY_ARCHITECTURES`。`qwen4_exp.py`已有language-model-only视觉权重跳过逻辑；专用loader仍应在调用 `get_tensor()` 前跳过 `model.visual.*`。

## 3. 源码改动

新增：

```text
python/sglang/srt/model_loader/qwen38_nvfp4_cache_loader.py
python/sglang/srt/model_loader/qwen38_nvfp4_pack/{build,format,validate,inspect}.py
python/sglang/srt/layers/moe/qwen38_expert_cache.py
python/sglang/srt/layers/quantization/qwen38_nvfp4_cache.py
python/sglang/kernels/ops/moe/qwen38_expert_cache.py
python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu
```

修改：

```text
python/sglang/srt/configs/load_config.py
python/sglang/srt/model_loader/loader.py
python/sglang/srt/server_args.py
```

新增load format：

```text
qwen38_nvfp4_cache
```

尽量不修改 `models/qwen3_5.py`、attention、MTP和scheduler代码。

自定义cache/ID映射kernel优先使用v0.5.21仓库内的 `sglang.kernels.jit`/`load_jit` 机制，随挂载源码在容器内按内容缓存编译。只有JIT无法满足时才修改 `python/sglang/kernels/aot/` 并在镜像内重建AOT扩展；不能假设仓库存在独立的 `sgl-kernel/csrc` 目录。

## 4. Runtime expert pack

离线按层构建。每层创建真实形状的 `FusedMoE`，使用v0.5.21原始weight loader加载512专家，调用 `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading()`，再写出FlashInfer CUTLASS直接使用的runtime布局。

Pack保存：

```text
w13_weight / w2_weight
w13_blockscale_swizzled / w2_blockscale_swizzled
g1_alphas / g2_alphas
每层input scale与SwiGLU参数
```

Manifest绑定checkpoint revision、SGLang commit、FlashInfer backend、SM120和shape。相关版本变化时重建pack。

运行时expert pack为普通只读mmap，只保留256～512MiB pinned staging ring，不创建完整pinned副本。

## 5. 专用loader和quant method

Loader使用单线程 `safe_open`，在 `get_tensor()` 前跳过：

```text
主模型 .mlp.experts.*
model.visual.*
```

保留shared expert、MTP、dense、attention和PLE。

`Qwen38CachedModelOptConfig`继承 `ModelOptMixedPrecisionConfig`，仅为主模型routed `FusedMoE`返回新的 `Qwen38CachedNvFp4MoeMethod`。后者不创建完整512-expert GPU参数，只持有cache引用和固定地址buffer。

## 6. Decode cache

初始配置：

```text
56 slots/层 × 48 = 2688 experts
约7.0～7.1GiB（按实际runtime payload复核）
```

稳定后测试64 slots/层，约8.0～8.2GiB。每层建议分为profile-pinned tier和adaptive tier；adaptive使用衰减LFU、LRU tie-break。

Decode流程：

```text
global topk IDs
→ cache lookup/H2D miss
→ remap到本层slot IDs
→ FlashInfer CUTLASS NVFP4 runner
```

最终路径必须用C++/CUDA custom op处理ID、slot和event，不能每层执行Python `topk_ids.cpu()`。

Slot必须保存key、generation、ready event和last-use event。kernel前等待ready；last-use完成前不能覆盖。

## 7. Prefill arena

长prompt会触达接近全部专家，不能走56/64-slot decode cache。

使用一个固定地址、容纳当前层512专家的arena，约1.32GiB：

```text
上传layer N全部experts
→ 使用原始0..511 IDs运行MoE
→ arena复用于layer N+1
```

初始chunk=8192；显存验证后测试16384。第一版单buffer，后续才评估双buffer预取。

## 8. 推荐运行参数

```text
纯文本：--language-model-only
单请求：--max-running-requests 1
上下文：--context-length 225280
主KV：--kv-cache-dtype fp8_e4m3
draft KV：--speculative-draft-kv-cache-dtype fp8_e4m3
QSA indexer：bfloat16
Mamba state：bfloat16
page size：64
prefill chunk：8192
MoE backend：flashinfer_cutlass
第一阶段：关闭MTP和CUDA Graph
```

稳定后开启NEXTN：3 steps、4 draft tokens。MTP verify最多形成40个expert路由项/层，使用active-slot保护。

## 9. 测试策略

不运行SGLang完整测试集。项目测试位于：

```text
tools/qwen38_cache_tests/
```

### CPU自包含

```bash
tools/qwen38_dev/bin/test-cpu
```

使用 `unittest`，文件名为 `test_cpu_*.py`；覆盖pack、manifest、cache policy、slot lifecycle、ID remap、loader filter和trace replay。

### 4080 SUPER小测

```bash
tools/qwen38_dev/bin/test-gpu-small
```

预算8GiB，单项不超过10GiB；空闲不足则skip。不启动server、不加载完整模型。覆盖cache/H2D/event、控制路径，以及使用checkpoint-format fixture的Marlin NVFP4数值测试。SM89不能消费SM120 FlashInfer CUTLASS runtime pack，因此pack最终布局与CUTLASS parity只在5090验证。

### 5090里程碑

仅用于SM120 kernel、完整模型、prefill/decode、MTP、128K/220K和性能。一次模型加载运行整批测试。

## 10. Docker开发

全部测试在用户制作的v0.5.21镜像中运行。镜像提供 `HF_HOME`；脚本不覆盖。

挂载：

```text
repo → /workspace/sglang
results → /workspace/results
```

容器设置：

```text
PYTHONPATH=/workspace/sglang/python
--ipc=host
--ulimit memlock=-1
```

AI首次进入项目完整阅读：

```text
tools/qwen38_dev/AI_DEVELOPMENT.md
```

统一工具：

```text
tools/qwen38_dev/bin/test-cpu
tools/qwen38_dev/bin/test-gpu-small
tools/qwen38_dev/bin/check-job
tools/qwen38_dev/bin/docker-job
tools/qwen38_dev/jobs/current.json
```

## 11. 5090 Fire-and-Forget

纯Python supervisor从普通终端预先启动，独立于27B后端。Codex更新工作集合和 `jobs/current.json` 后只执行：

```bash
tools/gpu_switch/switch-test dispatch --config <config>
```

投递成功后立即结束turn，不停服、不轮询、不等待。

Supervisor随后：

```text
延迟数秒
→ 停27B
→ 等5090释放
→ docker-job启动测试容器并跑整批测试
→ 删除测试容器
→ 恢复27B并等待健康
→ 写artifacts/latest-test.json
→ 可选执行Codex session wake命令
```

## 12. 抗Compact工作集合

仓库内维护：

```text
.agent-workset/
├── RESUME.md      # compact后唯一入口
├── COMPACT.md     # compact前临界状态
├── STATE.md       # 当前事实
├── DECISIONS.md   # 稳定决策
├── NEXT.md        # 3～7项短期动作
├── FILES.md       # 源码地图
├── TESTS.md       # 测试状态
├── ENV.md         # 环境
└── artifacts/latest-test.json
```

Compact后读取顺序：

```text
RESUME → COMPACT → STATE → DECISIONS → NEXT → FILES → TESTS → latest-test
```

随后运行 `git status --short`、`git rev-parse HEAD`、`git diff --stat`。聊天摘要与工作集合冲突时，以实际工作树和工作集合为准。

## 13. 里程碑

```text
M0 固定版本和纯文本模式
M1 runtime expert pack
M2 专用loader
M3 同步decode cache
M4 prefill arena
M5 C++ fast path和异步H2D
M6 FP8 KV与128K/220K
M7 MTP
M8 64 slots、16K chunk、CUDA Graph优化
M9 可选4080 vision encoder
```

## 14. 实现注意事项

- FlashInfer decode runner使用56/64 experts，并接受remapped IDs；如元数据绑定512，创建独立decode runner。
- Runtime pack是后端布局，不是通用checkpoint格式。
- Prefill每个chunk会读取完整63.3GiB experts，chunk大小决定总H2D量。
- 运行时只保留一个完整expert pack副本。
- CUDA Graph先实现all-hit路径，miss路径保持eager。
