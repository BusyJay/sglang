# combined-patch：M5 稠密共享与性能诊断

基准提交 `58a6c24e3ff9aefbc2e9e0eeb76da422e40fb62c`，当前分支 `metal`。
原包 `combined-patch.tar.gz` 的 patch SHA256 为
`0c3a13416282b11367345153c7fedff75a28143f565955a852f334133fe1ecb2`。
应用前核验了基准、patch 哈希、全部 11 个源码文件哈希与 `git apply --check`；
应用后再次比对文件。原始清单见 [source_manifest.json](source_manifest.json)，输入 patch 保存在
[source_candidate.patch.gz](source_candidate.patch.gz)，解压后与原 patch 哈希一致。
它是原始审查输入；最终源码以本轮
提交和 validation.json 指纹为准，不将旧输入 patch 当作最终发布包。

## 采用决定与默认行为

保留并合入稠密共享 kernel、KDA 验证投影接入和诊断工具。
兼容形状下的运行时优化随既有 `ffn` 验证模式默认启用，不需要新的环境变量。
原有 FFN 批量融合与 MLA 投影缓存保持启用；未修改权重、量化配方或模型方程。
正常服务不导入或启动 profiler，诊断工具通过独立 CLI 显式运行。

本轮短请求中，推荐 **`--mlx-mtp-num-steps 2`**。
MTP head 的加载仍由该已有参数显式控制，CLI 默认 0 没有改变。
动态模式仍可用（-1），本轮速度略低于固定两步。该选择针对下述 checkpoint、
上下文和请求面板，不是所有模型/上下文的通用最优值。

长期 patch 采用、诊断、默认启用和提交规则见根目录 [AGENTS.md](../../../AGENTS.md)。

## 环境与隔离

- Apple M5 Ultra，256 GiB unified memory；MLX 0.32.4、mlx-lm 0.32.2。
- 使用已有 `sglang-metal` Python 3.12 环境，没有安装依赖或下载权重。
- 同一 `mlx-community/GLM-5.3-Flash-4bit` snapshot：
  `fa04cbc3622b16aa7475404060b05e506d81bd4e`。
  实际专家权重混合 4/5/6 bit，affine metadata 为 FP16；KDA 大投影、KV-B、
  vocabulary head 仍为 BF16。本轮收益来自计算/权重复用，没有改成另一种“4bit”配方。
- 同时只运行一个完整模型进程，沿用 serving 的进程级 recommended working set
  （240 GiB），未更改 OS wired-memory 设置。
- HTTP 旧版对照使用独立的原基准 checkout 和显式 PYTHONPATH，核验实际导入路径。
  对照完成后移除了该干净 checkout；用户已有未提交文件保持原状。

## 实际机制

`gemm.mlx_dense_rows` 在一个 SIMD 行组中让多个 token 的累加器共享矩阵读取，
保留 MLX 0.32.4 原生 singleton GEMV 的 K 切片、浮点累加和 shuffle reduction。
它支持当前对齐几何的 FP16/BF16 同 dtype 权重/输入，输出维度有尾部 mask。
较长链按最多四个向量分块，是控制寄存器占用的实现选择，不是 Metal 行数上限。
不在该归约几何内时继续使用已有 MLX 运算；没有强制转换 checkpoint dtype。

输出头现在可共享实际 BF16 权重，解决了上轮只有 affine head 能用行 kernel 的缺口。
KDA 在每层先计算已知行的六路大输入投影，再把对应行的 `projected` 元组显式传给
原有逐行卷积、门控、recurrence 更新。没有临时修改模型模块标志，也没有用末尾
state 冒充各个中间位置。f_b/g_b 小投影仍逐行执行。

这没有接入跨请求的逐层合批：普通 GLM decode 仍逐请求执行完整模型。
单请求普通 decode 没有多行可共享，除了上轮已有的 MLA 缓存，不会获得本轮
MTP 多行共享收益。后续可把这些 kernel 用于独立请求的无状态投影和 FFN，
但仍须管理各请求独立的 cache、长度和采样状态。

## 完整模型 A/B

[dense_ab_initial.json](m5/dense_ab_initial.json) 在同一模型实例中切换四种候选。
所有候选保留旧版 FFN/cache 优化；reference 使用旧版逐行 attention 循环与原生
稠密 GEMV。每个强制链四次 warm 测量，顺序交替；128-token 生成正反顺序各一次。
控制仅存在于 benchmark 进程中。

| 候选 | 固定 step 1 模型内 decode tok/s，两轮 |
| --- | ---: |
| 原 FFN/cache 基线 | 35.24–35.46 |
| 仅共享 dense head | 36.85–36.92 |
| 仅共享 KDA 大投影 | 43.32–43.64 |
| 二者一起 | **45.51–45.69** |

组合相对配对基线均值提升约 **29.0%**。主要收益来自 KDA，共享 head 也有独立收益。
所有八个生成面板的 token hash 一致，均为 65 次验证、65 个草稿、61 个正确草稿。

| 验证行数 | 原路径中位数 ms | 仅 KDA ms | 组合 ms |
| --- | ---: | ---: | ---: |
| 2 | 50.88 | 40.40 | 39.37 |
| 3 | 64.38 | 48.52 | 46.63 |
| 5 | 101.47 | 74.44 | 72.63 |
| 9 | 223.07 | 127.99 | 119.51 |

16 个强制链面板的完整 logits、hidden、每个位置的每层 cache 均与 reference
逐位一致，cache offset 也一致。没有放宽数值容差。独立 head 的短链计时有噪声，
不能由某一行的微基准单独推断整段生成收益。
完整混合候选面板峰值 MLX allocation 为 **193.79 GiB**，是该次完整测量的峰值，
不是跨运行推算的内存下降；新机制没有额外保存一份整模型稠密权重。

## HTTP 与 step

[http_candidate.json](m5/http_candidate.json)：context 4096、chunk size 256、
最多两个运行请求；固定 seed 1234。英文、中文、代码各生成 128 个 greedy token，
每种请求两轮，第二轮反向排序，每个计时请求先 flush 请求缓存。
吞吐是排除 TTFT 后的 decode 速度。

| 草稿步数 | 三类请求平均 HTTP tok/s |
| --- | ---: |
| 0 | 28.72 |
| 1 | 41.90 |
| 2 | **44.97** |
| 3 | 43.76 |
| 4 | 33.74 |
| 动态 -1 | 43.08 |

独立原提交对照 [http_baseline_after.json](m5/http_baseline_after.json)：
普通 decode 28.69，固定一步 **33.96 tok/s**。
同一步数的候选版提升 **23.4%**；采用本轮最佳两步后比旧版最佳一步提升 **32.4%**。
普通 decode 的差异约 0.1%，不属于本轮多行共享收益。
反向进程顺序复测 [http_candidate_after.json](m5/http_candidate_after.json) 为
step 1 **43.10**、step 2 **45.52 tok/s**，再次确认最佳两步。
[http_comparison.json](m5/http_comparison.json) 核验十组配置的 **60 组计时输出、
140 项契约输出**，全部 hash 一致。

所有候选配置通过 14 项 HTTP 契约：chunked prefill、prefix exact/suffix、seeded
sampling、logit bias、regex grammar、token/top logprobs、stop token、短 token
预算、并发请求、断开后的清理和继续使用。计时与契约的 hash 均跨配置和原提交比较，
不是只比较候选自身的不同参数。

步数变化有实际原因：多行投影共享使三行验证的成本从约 64 降至 47 ms，
两步草稿更容易摊薄成本；更深草稿仍有额外 head/验证工作和下降的条件命中率。
动态模式还支付探索较大窗口的成本。接受统计以 HTTP 整个配置累计为口径，包含
契约请求，不能把末尾 adaptive score 当作正常计时请求的稳态吞吐。

## Profiler 验证与修正

工具说明见 [README.profile.md](../README.profile.md)。提供：

- 不插桩 baseline、多轮正反顺序 host 插桩和独立 sync 归因。
- 模块/融合入口的 shape、dtype、None 返回和权重清单；有界 DOT 图和本地 viewer。
- Xcode Metal capture 与 Metal System Trace 导出；设备区间按 PID 过滤和区间并集统计。

新增工具不在生产热路径启用。本轮补上了完整模型的进程级资源设置，与正常 serving
一致；否则巨量权重可能落入 pageable GPU 工作，不能拿停顿结果评估 kernel。
同时补记 mlx-lm 版本与生成 token 预算，compare 明确报告不同预算的配置差异。
新增 XML 引用解析、PID 过滤、重叠区间和 hook 异常恢复等正常 registered 测试。

本机小模型 [profiler_tiny/REPORT.md](m5/profiler_tiny/REPORT.md) 的三行验证，
baseline/host/sync 中位数约 2.15/2.97/10.69 ms，全部 hash 相同；图导出和所有 hook
恢复正常。host/sync 耗时是仪器干扰，不能作为正常吞吐或逐 shader 时长。

完整 checkpoint 的 [profiler_full/REPORT.md](m5/profiler_full/REPORT.md)
也经过本机运行：三行验证 baseline/host/sync 中位数为 **45.13/46.98/184.57 ms**，
三种模式全部 hash 相同且没有丢失事件。host 插桩约增加 4.1% 时间，强制同步约为
baseline 的 4.09 倍；后者只能用于定位，不能当作正常吞吐。该过程 swap 前后均为
719.19 MiB，没有增长。原始 host/sync trace、shape 表、环境和权重描述符保留在本机，
便携报告不包含权重值或完整 GPU buffer capture。

完整模型两个 host trial 记录了 68 次 KDA 稠密共享入口（34 层 × 2）和两次
输出头入口，全部返回非 None。实际形状分别为 BF16 `[24896,4096]` 和
`[154880,4096]` 权重；这证明当前形状采用了入口，不是逐 shader 时长或 DRAM 流量测量。
压缩的 `profiler_full/host.trace.json.gz`、`sync.trace.json.gz` 保留完整 host 事件，
可用 `gzip -dk` 解压后查看；权重清单只含元数据，可能含共享 storage 的描述。

本机 Xcode 27.2 的五秒上限小模型录制/导出也已实际通过。
[device-summary.json](m5/device/device-summary.json) 筛选 PID 83865，得到 254 个设备/
encoder 区间，观测跨度 192.51 ms，区间并集 40.07 ms。包含启动、预热和仪器干扰，
不能当作正常 decode GPU 利用率或逐 shader 时长；host/device 时间原点不同。
原始 34 MiB Instruments trace 和 XML 留在本机，不纳入提交；筛选后事件在
[device.trace.json](m5/device/device.trace.json)。

随包的 M4 Pro 历史结果保存在 `source_profiler_*`、`source_device_summary.json`，
只证明此前工具验证；没有拿它们换算 M5 吞吐。原始包保持不变。
GPU capture 可能包含权重，不作为便携证据附件；metadata/host span 本身不证明
实际 DRAM 字节数、occupancy 或 kernel 带宽饱和。

## 复现

复用已有环境，令 `$MODEL` 指向上述已有 snapshot：

```bash
$PY benchmark/mlx/bench_glm5_next_dense_verify_ab.py \
  --model-dir "$MODEL" --rows 2 3 5 9 --repeats 4 --output-length 128 \
  --output benchmark/mlx/results/dense_ab.json
$PY benchmark/mlx/bench_glm5_next_mtp_http.py \
  --model-dir "$MODEL" --steps 0 1 2 3 4 -1 --repeats 2 \
  --output-length 128 --check-contracts \
  --output benchmark/mlx/results/dense_http.json
$PY benchmark/mlx/profile_glm_mlx.py run \
  --model-dir "$MODEL" --revision fa04cbc3622b16aa7475404060b05e506d81bd4e \
  --workload verify --rows 3 --rounds 2 --warmup 1 --sync \
  --output benchmark/mlx/results/dense_profile
```

完整模型比较是短上下文、小请求面板，不替代长上下文、质量 benchmark 或高并发
性能验证。MLA 输入投影、跨请求调度、跨行专家权重显式共享和更完整设备 counter
分析仍是待改进的实现机制，不能称为 Metal 平台限制。

## 最终验证

正常 registered suite：**152 tests、196 subtests 通过**。其中包含完整 PyTorch
小模型 oracle、native dense/affine 数值对照、稠密长归约和尾部/strided 输入、
KDA 每位置状态/权重更新失效、MTP 接受/拒绝及 profiler 的五项 CPU 测试。
沿用既有环境中两项 AMX import probe 的排除；其中 common.py 探针对缺失 Triton 的
已有失败未修改。本轮未安装依赖来掩盖它。格式、isort、致命 lint 和 whitespace 检查通过。

[validation.json](m5/validation.json) 保存最终源码/证据指纹、原包来源、测试与
对照摘要。测试日志仅去除了行尾空白，未改变结果；原始大 trace、权重和虚拟环境
不进入便携提交。

机制提交：`d7ee7c9406`（稠密 head/kernel）、`e818e75989`（KDA 投影共享）、
`a54f0797af`（profiler）。本报告与对照脚本作为独立证据提交。
