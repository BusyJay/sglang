# M5 Ultra：NA、FP8 KV 和 MLA 投影融合实测

2026-11-06。基准 `46b98b4030`，生产候选 `00f5b6c890`。本机实测，不外推其他设备。

## 已采用的机制

MLA 的 latent-query 投影在写回时完成原有的缩放，并按 `[token,head,channel]`
存储；逻辑接口仍为 `[1,head,token,channel]`。保留“投影舍入 → scale 转为输出
dtype → 乘法再次舍入”，没有改变点积归约、checkpoint 字节或量化。
覆盖的 H64/D512 路径默认使用；通用 SDPA 保留自己的缩放边界。

匹配的 96-token 生成 inventory 中，341 次调用各消除了一个 multiply 和一个
transpose copy，总 dispatch 从 113303 降至 112621，encoder 数仍为 3539。
这是实际 dispatch 减少，不代表 shader 时间同比减少。

最终源码 ABBA，三条 192-token 请求，每臂六次热态测量：

| 路径 | 合并 decode tok/s | 峰值分配 GiB |
| --- | ---: | ---: |
| 基准 | 70.7217 | 172.609 |
| 投影写回融合 | 71.0539 | 172.608 |

约 +0.47%，收益小且接近运行波动，不能宣称显著提升，更不能宣称达到 90。
另一次等价原型 ABBA 为 71.3842 → 71.7223；它与最终源码测量分开保留。
各次 token、草稿接受统计和 cache offsets 一致。逐层路由 ID 没有额外采集；
不把生成一致冒充独立路由审计。新增设备测试逐位检查了投影的两次舍入与布局。

本轮之前的隐式 MLA 索引已单独提交为 `0af43c8343`；它在固定 1K 历史测得
59.389 → 60.417 tok/s，而短请求基本持平，不能与上表相加。

## NA 实际覆盖与没有收益的原因

实际模型 attention query 是 **BF16**，不是 target head 的 FP32。
BF16 KV 与 packed FP8 KV 都执行了 `ragged_scores_nax` / `ragged_values_nax`。
原生 dense NAX GEMM 也出现在图中。“FP8 没走 NA”不是这里的回退原因。
`pipeline_inventory.json` 记录真实 pipeline 名称和次数。

KDA 的 MXFP8 小批量投影、FP32 target head、原始 NVFP4 MoE 仍有 SIMD 路径。
不把这些现有实现选择称为 Metal 平台限制。

| 候选 | 本机证据 | 决定及未解决部分 |
| --- | --- | --- |
| MXFP8 四行补到 16 行，强制原生 NA | 原始 KDA bank `[24576,4096]`，约 112 → 136 µs；pipeline 确认由 QMV 变为 `mxfp8_qmm_t_nax`，内部 BM32 | 不采用 padding 方案；额外行与准备开销抵消收益。没有降低精度 |
| 直接 cooperative-register NA 小批量 kernel | 合法的 BM16/BN32/BK32 约 416 µs，明显慢于 QMV；split-K 未修好 | 当前原型缺少原生 QMM 的装载/流水机制，K32 循环和无效行仍多；不是硬件上限。BM8 被本机 MPP cooperative API 的静态检查拒绝，仅是该 API 约束 |
| MXFP8 SIMD groups / 每线程输出行调整 | KDA 单行部分布局改善，四行多有回退；target head 个别布局约 2–4% 局部改善 | 没有证明完整模型净收益，不全局替换几何。未降低 dot 精度 |
| 把 query 缩放塞进 QK kernel | 完整模型 ABBA 71.5860 → 71.2476 tok/s，token/接受一致 | 不采用这个位置；缩放会按 key tile 重复。改进为投影写回融合，已采用 |

不能通过单纯增加 NA kernel 数量保证吞吐。当前 batch 小、原始格式解码和布局准备
都需要付成本。FP32 路径如何高效使用 NA 并保留原始方程，仍是实现工作；本轮没有
证明它不可行。未采集逐 shader 的带宽/指令瓶颈计数，不把全模型断言为纯带宽受限。

## FP8 KV：两次改进后仍没有稳定净收益

现有 packed KV 使用 E4M3 + 每 32 值 UE8M0 scale，indexer 保持 BF16。
当前 QK 会重复解码，并写每个 query 的 BF16 临时 KV，供 PV 使用。
第一次改为共享 prefix/tail 解码，第二次把两个解码合并为一个 dispatch；两者相对
原 FP8 路径均保持生成 token 和接受统计一致。

| 同一实验内的对照 | 原路径 tok/s | 共享解码 tok/s |
| --- | ---: | ---: |
| 三条短请求，分别解码两个 segment | 70.7917 | 70.6544 |
| 固定 1K history，分别解码 | 57.0716 | 57.1859 |
| 固定 selected 2K history，分别解码 | 64.5319 | 63.4682 |
| 三条短请求，合并两个 segment 解码 | 70.3263 | 70.1770 |

随机输入的 isolated attention 四行微基准却有明显改善，例如 2K 长度约
468 → 202 µs。它不能替代上表全模型结果。trace 确认第一版每次 attention 多出
两个 unpack dispatch；第二版已减少为一个。共享解码还会遍历 segment 容量，
selected 路径可能解码未选中的位置。原路径则重复解码/写入 query 临时缓冲。
这些额外工作由源码与 inventory 确认，**各因素抵消收益的时间份额尚未独立分离**。
没有把“微基准快”直接写成“完整模型快”，也没有把接受率当作同精度候选的回退原因。

在同一历史实验内，BF16 为 1K 58.3953、selected 2K 63.4407 tok/s；FP8 对 BF16
的排序也随请求变化，且 token/接受轨迹因量化不同。没有做 GPQA 等质量评测。
最终保留原有 FP8 支持，服务默认 BF16；没有新增默认关闭的实验开关。
两个共享解码原型未进入生产。后续需要减少解码/临时状态本身，并用真实 selected
地址轨迹复验；不能仅把容量解码换成另一份全量拷贝。

## 并行性与剩余机会

现有 D3 已批量投影并执行私有下一轮 prefetch，多请求也共享无状态投影工作。
本轮没有新增 stream 或把串行 fallback 包装成并行支持。encoder timestamp 不提供
单个 dispatch 的执行区间，不能据此宣称所有 NA 单元或 GPU 已充分利用。

包含 mixed SwiGLU 的 encoder 占据很多时间，但同一 encoder 内还有其他操作；
不能将其总时长直接全部归给 MoE。更值得继续测的是实际 MoE bank 的解码、权重
读取、寄存器/装载和调度，而不是重复优化只占很小部分的 head 后处理。
GPU 接受状态机、固定地址执行和完整分页仍未完成；本轮没有把它们宣称为已优化。

## 环境、原始数据与验证

- M5 Ultra 256 GiB；MLX 0.32.4；目标 revision
  `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`，草稿
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`。
- TF32=0、DFlash D3、BF16 KV/A16、eval groups 2/4/2、MLX cache 1 GiB、
  max MB/ops per buffer 1024/200、进程 memory/wired budget 为 recommended 的 .78。
  没有安装依赖或调整系统 wired-memory 参数，没有同时加载两份完整模型。
- `measurements.json` 保留逐次顺序、逐 token 原始秒数、输出 hash/最终完整 token、
  接受统计、offset 和内存。吞吐是 192 输出中的 191 个 decode 区间，排除
  prefill/首 token；合并时用总 token 数除总时间，不平均 tok/s。
- `pipeline_inventory.json` 是计数摘要；完整 stage trace 保留在本机 `resident/`，
  未提交大 capture。encoder 内可含多个 dispatch，持续时间不能逐 kernel 归因。
  本机支持 encoder-stage counter sampling；collector 使用有界 sample buffers。
  resolved timestamp 已是纳秒，未再次除以 raw clock frequency。
- 一次诊断误用了 qk_nope_head_dim=128，实际 checkpoint 为 256；全部 `producer_*`
  结果排除。修正后的 `producer2_*` 与最终源码数据通过 token/接受比较。
  一次 scalar 参数索引编译错误也没有计入性能结果。
- 最终 82 个 registered MLX 设备测试通过，包含 GEMM、ragged attention、GLM
  runtime/cache 测试。三条 96-token FP8 全模型新旧路径输出与接受统计一致。
- 最终 HTTP：17 个顺序 case、2 个并发请求、原 SSE 的 terminal abort 和后续复用
  通过；覆盖 32/2563-token prompt、chunked prefill、prefix hit、logprob、固定种子
  sampling、stop 和请求生命周期。HTTP 96-token decode 约 61–69 tok/s，口径与
  engine 上表不同。HTTP 源码 hash 与最终生产文件一致。
- 三个生产文件 Ruff format 通过；修改文件的 Ruff 诊断与基准一致，没有新增。
  GLM 文件原有 13 条、dense 测试原有 8 条 lint 诊断；没有混入无关清理。
  测量后 GEMM 文件仅换行格式变化，AST 一致，见 `format_equivalence.json`。

服务脚本为 `../decode_gain_20261105/launch_default.sh`，已提交监听
`0.0.0.0:39007`。本机 `127.0.0.1`、`10.173.1.94`、`100.103.62.74` 健康检查均为
200；没有替外部设备测试防火墙/NAT 可达性。

## 复现

服务正在运行。以下全模型 A/B 脚本会检查端口并拒绝同时加载第二份模型；需要另行
安排服务停机后运行。使用现有环境和相同权重，无需安装工具：

```sh
PYTHONPATH=python MLX_ENABLE_TF32=0 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200 \
 /Users/jay/workspace/prod/common/sglang-metal/bin/python \
 benchmark/mlx/engine_optimization_20261106/reproduce_final.py \
 --output benchmark/mlx/engine_optimization_20261106/replay --tokens 192 --rounds 1
```

脚本从 Git `46b98b4030` 提取基准方法和算子，候选使用当前 checkout；同一份
resident 权重交替 A/B/B/A，每臂预热。`--history dense1k` 或 `selected2k` 可复用
固定 token history。这份整理后的复现入口只做静态检查，历史实测使用
`resident.py` / `job_12.py`；其本机原始文件 hash 在 `local_artifacts_manifest.json`。

```sh
PYTHONPATH=python MLX_ENABLE_TF32=0 /Users/jay/workspace/prod/common/sglang-metal/bin/python \
 -m unittest test.registered.kernels.ops.gemm.test_mlx_dense_rows \
 test.registered.kernels.ops.attention.test_mlx_ragged \
 test.registered.unit.hardware_backend.mlx.test_glm5_next_mlx

PYTHONPATH=python /Users/jay/workspace/prod/common/sglang-metal/bin/python \
 benchmark/mlx/engine_optimization_20261106/http_validate.py --mode dflash --count 3 \
 --tokens 96 --output benchmark/mlx/engine_optimization_20261106/http_replay.json
```

首次请求完整冷启动、长达 8K 的完整质量/吞吐、跨机器结果仍未独立验证。没有以
AST/import 检查代替设备测试，也没有把此次小幅改善当作最终吞吐上限。
