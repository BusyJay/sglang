# GLM-5.3-Flash MLX vision 验证记录

测试日期：2026-11-02。目标机 BlazesMacStudio，M5 Ultra、256 GiB；
MLX 0.32.4 / mlx-lm 0.32.2 / Torch stable 2.13。
目标 checkpoint：fa04cbc3622b16aa7475404060b05e506d81bd4e，
DFlash2：bf582e4eacc1810f76656d1811693ff6c6737d2a。
语言权重维持原 4-bit，视觉权重维持原 BF16，没有重新量化或改 checkpoint。

## 实现

本机 encoder 默认加载。EPD encoder-only 仅加载视觉 shard；language-only 不加载
视觉权重，沿用 SGLang CPU BF16 + zmq_to_tokenizer 协议。
支持图像 patch projection、24 层 RMSNorm/QK norm/2D RoPE/noncausal attention/
clipped SwiGLU、空间下采样、GELU merger。相同 patch 数的帧批量 attention；不同
尺寸的帧保持隔离。原 HF processor 负责图像尺寸、patch 顺序和 prompt span。

图片绝对 span 跨 chunk 注入 target embedding，DFlash 同时捕获视觉 target hidden；
native MTP prefill 同样使用实际 image embedding。宽图片哈希在 scheduler 保留
int64，进入词表前仅覆盖视觉 span 的位置被替换。请求完成、撤回、清空释放视觉状态。

服务修复包括转换 checkpoint 的纯文本模板选择、encoder TP1 预处理共识路径、
macOS 共享内存整页补齐导致复制和 reshape 长度错误，以及 worker 图片 padding 入口。
MLX worker 在 extend 前消费 scheduler 的 deferred auxiliary COW/clear，恢复命中
前缀的原生 KV/KDA 状态及 DFlash captures；保留 source slot 与请求 slot 所有权。

## 功能证据

`http_vision.json`：本机/EPD 各八个请求，红图 Red、蓝图 Blue、双图顺序 red/blue、
OCR HELLO 123。两种模式逐项文本一致；usage 图像 token 分别 16/24/40/256。
chunked-prefill-size=32，覆盖横跨图片 token span 的分块。

`lifecycle_local.json` / `lifecycle_epd.json`：相同尺寸红→蓝→红、长图文 prompt、
两并发颜色请求、图像 SSE 中止及后续蓝图、文本 2+2 回归。

`reference_quality.json`：同一常驻完整语言模型分别消费 MLX 与 Transformers BF16
视觉特征，四项最终答案一致；红图、双图、OCR token 一致，蓝图 reasoning 长度不同。
这属于有限功能覆盖，不是完整视觉质量基准或逐位数值一致证明。

## 数值定位

`encoder_parity.json`：checkpoint BF16 权重在对照臂临时 widen FP32，独立 Torch
conv3d/conv2d/SDPA oracle 对照 MLX FP32 方程，相对 L2 约 1.2e-5～2.3e-5。
生产仍用 BF16。TF32 关闭，对照不修改 checkpoint 文件。

`hf_vision_parity.json` 与逐层诊断：BF16 输出相对 L2 约 6%～9%，cosine 约
0.996～0.998，不宣称 BF16 逐位一致。修复 QK epsilon 配置与 RMSNorm 舍入边界后，
逐算子 SiLU/GELU 采用 FP32 内部计算再 cast BF16，匹配参考 activation 精度。
相同 Q/K/V 下 MLX/Torch CPU BF16 SDPA 相对 L2 0.00131；双 FP32 SDPA
1.27e-7，最大绝对误差 7.15e-7。Torch CPU 自身 BF16/FP32 SDPA 相对 L2
约 0.0017。误差在 attention 输出出现，经残差与 24 层累积；不能据此称为 Metal
硬件不支持。没有用放宽测试容差掩盖数值差异。

`hf_fp32_rope_parity.json` 诊断另测恢复参考 inv_freq FP32 buffer，仍存在 BF16
误差，排除了仅由 Torch 模型整体 cast 导致的频率 buffer 差异。参考特征生成对照
已确认本次有限样例的最终答案保持一致，复杂图像/视频质量仍需独立数据集评估。

## 性能

`decode_vision.json`：一个常驻目标、AR/D3/D5 轮转顺序，三条提示，每臂 128 输出。
逐 decode_batch 建图、草稿、验证、缓存提交、mx.eval 输出/cache、主机 token
读回计时，排除加载、encoder、prefill、HTTP。固定长度忽略 EOS，所以会生成后续
伪对话，属于受控性能样例，不能作为正常请求自然输出质量证据。

| 图片 / prompt token | AR tok/s | D3 tok/s | D5 tok/s |
|---|---:|---:|---:|
| 112×112 / 42 | 36.982 | 53.038 | 45.988 |
| 168×112 / 50 | 37.286 | 49.702 | 42.665 |
| 448×448 OCR / 283 | 36.762 | 52.135 | 48.262 |

三臂各案例输出 token 完全相同。D3 草稿命中分别 75/153、70/166、74/158，
D5 为 78/239、72/265、80/229；5 草稿增加验证长度但接受数增长有限，因此本次
视觉负载采用 3 草稿。没有将原文字负载 78.6 tok/s 当成 vision 吞吐目标。

Encoder 112×112 首次 1.742 s，后续 9.69～12.88 ms；168×112 后续约 9.12 ms；
448×448 首次 28.17 ms，后续 20.64～20.89 ms。第一次包含 Metal 编译等冷启动成本。
测量同一进程同一 checkpoint，有完整 raw_seconds；不外推其他机器吞吐。
该 decoder 实验全局 swap used 1571.12→1571.44 MiB，未修改系统 wired-memory，
不能归因共享机器全部 swap 行为。

HTTP 最终本机首个红图 1.162 s / 重复 0.191 s。
OCR 本机 1.400/0.258 s。EPD 最终结果见下方验收补充。运行顺序本机后 EPD，编译缓存已热，
所以不能据首次数值断言 EPD 更快。encoder-only 启动约 4 s，语言约 65～70 s。

## 范围和复现

EPD 实测同机两个进程；跨机传输、网络容量、TP>1 MLX vision 尚未验证。
视频帧 attention 隔离有设备属性测试，尚无完整视频 HTTP 接受测试。
本机与 EPD 共用完整启动命令，见 README 和 start*.sh。
实验文件排除权重、虚拟环境及大型 capture。

## 最终验收补充

最终目标机测试 152 passed、3 skipped、123 subtests passed。3 项为平台条件
跳过，完整日志 `final_tests.log`。新源码 Ruff 检查通过；大型既有文件避免无关
格式改动，最终语法、diff 与 registered 测试入口另行检查。对全部修改文件单独
执行 Ruff import-sort 报告 72 项，包含既有 import 分组规则差异；没有将此项标为
通过，未扩大到无关代码的全文件格式整改。新 vision 源码及测试 Ruff 完整检查通过。

最终本机 1,242-token 图文长前缀首次 5.651 s、重复 0.213 s；修复前重复约
5.5 s。调度器命中 1,216 token 后仅计算剩余部分，不再完整重算。配置未开启
`--enable-cache-report`，OpenAI usage cached_tokens 会显示 0；该字段不是此次
复用判据。报告不以耗时单独证明数值正确，配套恢复状态设备测试及答案校验。

EPD 最终八项与本机逐项文本一致；1,242-token 长前缀 5.660/0.225 s，OCR
1.417/0.267 s。两种模式均通过并发、abort 后续恢复和纯文本回归。encoder-only
启动 4.084 s，语言本机/EPD 分别 57.379/58.346 s；按顺序运行，非交替启动对照。

最终代码已提交到远端 metal：`be7800e078` 共享内存修复，`3a510ba2b1` 前缀
状态恢复，`5a02d52492` vision/EPD 接入。23 文件 SHA256 清单为 source_hashes.json，
最终测试对应此清单。最终格式清理仅移除无关的格式 diff，模型计算 AST 不变。
