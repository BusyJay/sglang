# GLM-5.3-Flash + DFlash2 在线运行

2026-11-02 在 BlazesMacStudio（M5 Ultra 256 GiB，applegpu_g17d）验证。
复用现有 Python 环境，MLX 0.32.4 / mlx-lm 0.32.2。
目标为 mlx-community GLM-5.3-Flash 混合 4bit checkpoint，revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`；草稿为 incoai DFlash2 原始 BF16，
revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`。
DFlash2 共享目标 embedding/head，加载时省去 native MTP 权重，匹配加载阶段节省 4.2642 GiB。

## 启动

在远端执行：

```bash
cd /Users/jay/workspace/sglang
bash llm-online/start.sh
```

完整命令（与最终 HTTP 验证配置一致）：

```bash
cd /Users/jay/workspace/sglang
PYTHONPATH=/Users/jay/workspace/sglang/python \
SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 \
/Users/jay/workspace/prod/common/sglang-metal/bin/python -u -m sglang.launch_server \
  --model-path /Users/jay/workspace/prod/huggingface/hub/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e \
  --host 127.0.0.1 --port 30007 \
  --context-length 8192 --max-total-tokens 17024 \
  --chunked-prefill-size 256 --max-running-requests 2 \
  --mlx-enable-sampling --random-seed 1234 \
  --enable-deterministic-inference --disable-cuda-graph \
  --disable-overlap-schedule \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /Users/jay/workspace/prod/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --speculative-dflash-block-size 4
```

默认 block-size=4，即 1 个 anchor + 3 个草稿。5 草稿使用：

```bash
DFLASH_BLOCK_SIZE=6 bash /Users/jay/workspace/sglang/llm-online/start.sh
```

7 草稿用 `DFLASH_BLOCK_SIZE=8`。脚本支持 REPO_DIR、PYTHON_BIN、TARGET_MODEL、
DRAFT_MODEL、HOST、PORT 覆盖。默认监听 127.0.0.1:30007。不要添加 native MTP 参数。
前台运行，日志输出到终端；按 Ctrl-C 停止。

```bash
curl -f http://127.0.0.1:30007/health
curl -N http://127.0.0.1:30007/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain why the sky is blue.","sampling_params":{"temperature":0,"max_new_tokens":128},"stream":true}'
```

## 真实 decode 计时

之前 HTTP 脚本的 `decode_tok_s` 实际是首末 SSE 事件之间的 token 到达速率，命名不准确。
它受服务调度、序列化和事件发送影响，不能用作模型 decode 耗时。最终 HTTP D3/D5
的 67.511/66.592 tok/s 是客户端流速；并发数字也仅表示服务端到端吞吐。

下面采用已保存的完整模型 decoder 循环原始数据：每次 `decode_batch` 开始计时，
调用 `mx.eval` 等待输出和缓存完成，再 `.item()` 读回 token 后停止。
计入草稿生成、目标验证、缓存提交、主机建图和必要同步，排除加载、prefill 与 HTTP。
这是实际 decoder 循环墙钟耗时，尚无独立 GPU kernel 时间线。
投机解码一次验证会产出多个可消费 token，平均耗时必须按最终输出 token 分摊。

| 输入 token | 模式 | decode tok/s | 平均 ms/token | 去掉前 16 步 tok/s |
|---|---|---:|---:|---:|
| 32 | ar | 37.335 | 26.784 | 37.451 |
| 32 | dflash3 | 71.114 | 14.062 | 71.656 |
| 32 | dflash5 | 78.601 | 12.722 | 79.062 |
| 32 | dflash7 | 76.956 | 12.995 | 78.711 |
| 2050 | ar | 34.255 | 29.193 | 34.251 |
| 2050 | dflash3 | 61.791 | 16.184 | 61.196 |
| 2050 | dflash5 | 60.704 | 16.474 | 60.186 |
| 2050 | dflash7 | 61.290 | 16.316 | 59.502 |

每个 context/模式合计 381 个实际 decode token（三条提示，各 128 输出，prefill 已产生
第一个 token，因此每条计入 127 步）。聚合方法是总 decode token / 总墙钟耗时，
稳态列分别去掉每条提示的前 16 步。模式按提示轮转顺序，AR 与 DFlash 共用同一常驻目标；
一个 repeat，属于有限工作负载测量。HTTP 与这里的提示不同，差值不能全部归因于 HTTP。
32/2050 是初始输入长度，decode 中上下文继续增长。

这些数据对应最终 KDA 输入/输出 bank 和向量读回；随后辅助池 async snapshot 改动
不参与这个独立循环，最终服务语义另行通过 HTTP 验证。5 草稿在短上下文最快，长上下文
3 草稿略快；启动目录采用已验证的 3 草稿配置，不能宣称 5 草稿普遍最优。

原始数据、最终源码哈希和优化说明见 `benchmark/mlx/dflash2_20261102/`。
复现当前 decoder 循环（服务停止后，在远端单独运行）：

```bash
cd /Users/jay/workspace/sglang
PYTHONPATH=/Users/jay/workspace/sglang/python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 \
/Users/jay/workspace/prod/common/sglang-metal/bin/python \
  benchmark/mlx/dflash2_20261102/verify_host_ab.py \
  --tokens 128 --repeats 1 --output llm-online/decode-results.json
```

## 验证与未解决问题

最终生产源码 21 文件哈希与远端一致；远端 auxiliary 6 + DFlash serving 8 + GLM 63
设备测试通过，相关 dense/affine/mHC 测试在对应优化轮次通过。
最终 HTTP 两种草稿配置均与 AR 的 17 项及两并发请求语义签名一致，覆盖 token、logprob、
采样、stop、长 prefix reuse、abort 和后续请求。格式/lint/isort/registered admission 检查通过。

全部改成一次 async 提交会让验证更慢；原生 async_eval 的调度等待原因尚未定位到
任务数节流或分配压力，需要 GPU 时间线/原生 instrumentation。保留已测更快的分组提交。
最终 HTTP 期间共享机器 swap used 从 1229.81 增至 1624.06 MiB，归因未确认；
不宣称服务内存成本完全解决。首次请求 TTFT 约 1.5 秒，后续约 0.112–0.145 秒，
启动约 55 秒。没有修改系统 wired-memory 配置。

## Attention 执行路径

本脚本使用 MLX/Metal。DFlash2 草稿直接调用 `mx.fast.scaled_dot_product_attention`，
目标模型使用原生 MLX KDA/MLA kernel。通用 `--attention-backend fa4` 不选择这些
调用，因此命令不填写该参数。这份性能结果只对应上述 MLX 路径。


## Vision：本机 encoder 与 EPD 分离

GLM-5.3-Flash 默认加载原生 MLX BF16 vision tower；语言模型继续使用原 checkpoint
4-bit 权重和 DFlash2。DFlash 启用时不加载 native MTP。Vision encoder 同分辨率的帧
批量进行非因果 attention，不同分辨率保持独立 attention 域。像素由原 HF processor
处理，图片绝对 token span 跨 chunk 注入 target embedding，并同时进入 DFlash target captures。

本机 encoder（默认模式）：

```bash
cd /Users/jay/workspace/sglang
bash llm-online/start.sh
```

EPD：先启动 encoder，再在另一个终端启动语言服务。

```bash
cd /Users/jay/workspace/sglang
bash llm-online/start-encoder.sh
```

```bash
cd /Users/jay/workspace/sglang
bash llm-online/start-language.sh
```

`start-language.sh` 使用 `--language-only`，不会加载本机视觉权重；从 encoder 接收
CPU BF16 embeddings，转换为 MLX 进行同样的 span 注入。传输后端是现有
`zmq_to_tokenizer`，encoder HTTP 地址由 `ENCODER_URL` 覆盖。默认均监听 localhost，
跨机器部署需设置 encoder 的 `HOST`、语言端 `ENCODER_URL`，并确保双方可达传输端口；
当前实测是同一 M5 Ultra 上的独立进程，并未验证跨主机网络。

这个转换 checkpoint 的 `chat_template.jinja` 是纯文本 GLM-5.3 模板。视觉服务现在
自动使用仓库 GLM-5.3-Flash Python renderer，生成 image/video 标记；用户显式
指定的 chat template 仍优先。图像预算使用 `--mm-process-config
'{"image":{"max_image_tokens":256}}'`，新版 GLM processor 不接收 `max_pixels`。

复现材料位于 `benchmark/mlx/vision_20261102/`。完整验证报告和性能记录在
`llm-online/vision-validation.md`。HTTP 响应耗时包含 encoder、prefill、decode、
序列化和调度；真实 decode 数字来自单独等待输出与 cache 完成的 decoder 循环。

长图文前缀复用已验证。需要在 OpenAI usage 中显示命中 token 数时，启动参数加
`--enable-cache-report`；默认未开启时 usage 的 cached_tokens 为 0，不能据此判断
是否重算。最终生命周期验证以 scheduler 命中日志、设备状态回归和实际耗时为证据。
