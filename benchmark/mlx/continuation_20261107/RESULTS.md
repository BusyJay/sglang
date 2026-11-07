# GLM MLX 后续优化：5、6、7（2026-11-07）

本轮以单请求为主，没有双请求吞吐测试。代码默认启用已采用的机制；未推送远端。
此前 KDA 投影/链路、延迟状态及 mixed singleton 测试已分别提交为
`2dc2535e0a`、`700d710ab5`、`45a6746df1`，证据见
[上一轮报告](../kda_gates_20261107/RESULTS.md)。

## 采用的机制与边界

- **5：元数据输入，`0b94d2b1df`。** lengths、capacity、KV split 独立传入
  attention kernel，消除元数据拼接造成的三个 uint32 copy dispatch。
  长度、索引表宽度、缓存 split 继续使用运行时参数。trace 的单次 attention 从
  6 个 dispatch 减至 3 个。没有改变注意力计算；独立全模型吞吐收益未测定。
  本轮没有全面重做计算图或提交调度，进一步的资源声明/提交边界优化仍有空间。
- **7：共享选中历史，`34ee726b04`。** 低精度 MPP QK/PV 中，64 个 query heads
  共享选中历史 tile、索引与解码；QK 复用 24 KiB scratch 的生命周期。
  保持原有 K 分区、归约树、舍入与分段 KV 地址。FP8 使用已有 decoded buffer，
  BF16 不新增全局 gather。覆盖 H=64、D=512 的 FP16/BF16 MPP 路径；FP32
  继续原有实现。本轮共享 heads，没有新增跨 verification 位置共享。
- **6：请求私有草稿词表，`b6a8d1ea08`。** 每四次 readout 刷新一次完整词表，
  保留各草稿位置全局前 16384 行及最近 512 个已消费 token。其余轮次直接读取
  原始 MXFP8 权重的候选行，不复制选中权重、不修改 checkpoint 或量化。
  默认 B=1/D=3 时池容量 65536，加近期 token 后投影 66048 列（完整词表
  154880）。重复 ID 由 top-k kernel 合并，无须刷新后的两次排序/去重。
  保留原有 multiplier、softcap、codebook reranker；目标验证仍评分完整词表。
  仅用于现有 MXFP8 A16 无 bias head、selector top-k≤16；其他 head 继续其原有
  完整投影。候选元数据按请求持有，prefix 恢复重新刷新，不存入规范 KV 快照，
  只观察真正消费的上下文 token。已有 8-gram context-copy 机制继续工作，本轮未新增。

小池/慢刷新（4096/16）在 Python 上略快，但其他文本接受率回退，综合
73.364→69.254 tok/s；已淘汰该参数。复用分块 top16 作为池的方案也因正确草稿
89→82、验证轮数 38→45 而淘汰。最终扩大池并缩短刷新周期，三类短文本恢复原有
正确草稿数与验证轮数。候选池会影响 proposal 分布，不能承诺所有请求接受率不变。

## 单请求全模型结果

硬件为本机 **256 GiB M5 Ultra**，MLX **0.32.4**，`MLX_ENABLE_TF32=0`。
目标 checkpoint：`nvidia/GLM-5.3-Flash-FP4-Local`，revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`；草稿：
`incoai/GLM-5.3-Flash-DFlash2`，revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a`。
D3、BF16 KV/prefill，eval groups decode/prefill/verification=2/4/2，
确定性 greedy、每请求生成 128 token。一个常驻完整 target+drafter 内交替，
每臂先一轮预热再测量，顺序 ABBA。吞吐为实际返回 token 数除以逐步调用总时间，
包含拒绝、prefetch 和草稿计算；不把草稿 token 当作输出 token。

| 对照 | A tok/s | B tok/s | 变化 |
|---|---:|---:|---:|
| 约 2K 选中历史，仅比较 7（metadata→shared，完整 draft head） | 64.574 | 67.122 | +3.94% |
| 短英文，5+6+7 组合 | 69.545 | 70.266 | +1.04% |
| Python，5+6+7 组合 | 74.094 | 74.941 | +1.14% |
| 中文提问，5+6+7 组合 | 75.684 | 75.923 | +0.32% |
| 三类短文本按总 token/总耗时汇总 | 73.013 | 73.626 | +0.84% |

组合 A 是 `f473b42085` 的 attention 与完整 draft head；B 为最终三个机制。
这些是小样本本机数据，组合收益较小；没有证明第 6 项独立的吞吐收益，也没有将
微基准换算为整模提升。三类 A/B 的正确草稿/验证轮数分别为 86/41、89/38、90/37。
生成 token 完全相同，目标 consumed cache 逐位相同；第一条历史 baseline 作为缓存
参考，后续三条历史臂与其比较。所选短文本中 candidate reranker 的细分草稿路径可不同。
不声称路由数组或广泛文本分布完全一致，未另行采集全层 MoE route 数组。

每臂单独的原始 tok/s、逐 token 秒数、prefill 时间、token IDs、接受统计和缓存检查
保存在 [evidence.json](evidence.json)；聚合见 [summary.json](summary.json)。
加载约 27.1 秒为本轮一次常驻加载观测，没有冷启动 A/B 结论。
短文本 measured prefill：A 0.170–0.223 秒、B 0.171–0.224 秒。
组合峰值 MLX 分配约 A 172.623 GiB、B 172.625 GiB；它不是进程 RSS 或系统总占用。
使用正常 runner 的进程级 memory/wired 默认比例 0.78、cache 1 GiB；没有修改系统
wired-memory 参数。常驻 worker 前后系统已有 swap 为 1675.31→1861.38 MiB，
不能声称零 swap 增长或归因于本改动；worker 已退出后才恢复服务。

## 验证与未解决项

最终源码的 **36 项**定向 MLX 设备/运行时测试通过，涵盖：
共享历史与原生单行 attention 的 dtype/归约、稀疏/空/超 4096 长度、分段缓存、
运行时长度图复用；indexed MXFP8 与完整投影逐位相同、混合合法/非法 ID；
跨 tile 重复 ID、NaN/inf/tie top-k；DFlash 拒绝/消费、chunked prefill、prefix
restore、新请求所有权与私有 prefetch。未新增大规模吞吐矩阵。

按仓库 hook 的 Ruff `F401,F821,F823,UP037` 和 format 检查通过，diff whitespace
检查通过。完整 pre-commit 未运行（本机未安装），未安装依赖；既有未修改代码的
isort/完整 Ruff 历史诊断未混入本轮修正。
全模型测量后只整理了导入空行/别名格式、参数类型 guard 和测试；计算 kernel、
最终 16384/4 参数均与测量一致。最终 HTTP smoke 重新加载提交源码。
最终文件哈希见 [final_sources.json](final_sources.json)。

本轮 HTTP smoke 的四个冷/热 prefix 请求和两次单请求 streaming 均通过：32-token
prefix 的 output/logprob/top-logprob 与旧参考逐位相同；512-token prefix 分别对照
其原有 cold/warm 参考，streaming output IDs 对照旧参考。
**既有 512 整 chunk prefix 的 cold/warm logprob 差异仍未解决**，旧版同样复现，
不能声称该场景 cold/warm 数值已一致。HTTP 证据见 [http_smoke.json](http_smoke.json)。

未覆盖更广文本的接受率、长上下文组合方案、多请求吞吐或所有模型/head。
第 5 项进一步提交调度和第 7 项跨验证位置 tile 共享仍待优化。

## 复现

复用已有虚拟环境与完整权重，先停止正常服务；不要同时加载两个整模进程。
脚本不安装第三方包，所有输出留在工作区。创建新的输出目录并复制队列：

```sh
mkdir -p benchmark/mlx/continuation_20261107/reproduce
cp benchmark/mlx/continuation_20261107/reproduce_jobs.jsonl benchmark/mlx/continuation_20261107/reproduce/jobs.jsonl
PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 SGLANG_MLX_CACHE_LIMIT_GB=1 MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=200 /Users/jay/workspace/prod/common/sglang-metal/bin/python benchmark/mlx/continuation_20261107/resident.py --output benchmark/mlx/continuation_20261107/reproduce
```

脚本读取固定 token 历史，私有加载固定 commit 的 attention 做 A/B，最后队列 quit
退出释放模型；之后恢复服务。`http_smoke.py` 使用 `http_references.json` 中保存的
旧参考，只向本地 39007 发有限请求。原始 micro trace 结果保存在
`attention_probe.json`、`shared_probe.json`，属于 operator 证据而非整模吞吐。

定向测试命令（最终源码）：

```sh
PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 /Users/jay/workspace/prod/common/sglang-metal/bin/python -m unittest test.registered.kernels.ops.attention.test_mlx_ragged test.registered.kernels.ops.gemm.test_mlx_mxfp8_tiles test.registered.kernels.ops.sampling.test_mlx_topk test.registered.unit.hardware_backend.mlx.test_dflash2_mlx test.registered.unit.hardware_backend.mlx.test_dflash_serving
```

测试原始输出：[final_tests.log](final_tests.log)。HTTP 原始输出：[http_smoke.log](http_smoke.log)。
