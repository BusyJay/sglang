# 2026-11-05 远端 decode 优化与精度决策

远端 metal 源码提交：
- 9f5b0979a7 [Perf] Fuse GLM MLX routed and shared verification work
- b6cb133d20 [Feat] Retain optional MLX FP8 KV and W4A8 prefill paths

开发、实测和提交都在 /Users/jay/workspace/sglang，未推送。
用户要求：降精度未超过原精度时恢复默认原精度，保留适配用于后续实验。

## 默认与可选

本目录 launch_default.sh 恢复 BF16 MLA KV、BF16 routed prefill、
W4A16 routed decode/verify、DFlash D3。auto KV 不自动切到 FP8，测试覆盖。
checkpoint 原有 NVFP4/MXFP8 权重及共享专家 dynamic A8 保留。
mHC learned weights/router/KDA recurrence FP32 未新增降精度；
indexer/draft cache 保持原精度。

显式实验参数：
- --kv-cache-dtype mxfp8
- --mlx-prefill-activation-dtype fp8

../draft_verify_optimization_20261105/launch_fp8_kv.sh 和
launch_fp8_prefill.sh 保留。FP8 E4M3+UE8M0/32 将512元素 MLA pool
行1024字节降到528字节，节省48.44%；不是所有缓存节省或吞吐证明。
新增 BF16 norm 输出舍入没有合入生产代码。

## 完整生成趋势

M5 Ultra256GiB，MLX0.32.4，TF32=0，normal serving process memory/wired
budget .78，cache1GiB，eval groups2/4/2，prefill chunk256。
目标 revision76348ddb2be2e35e39ecfc00e75dc07703f4f3d5，
DFlash2 revisionbf582e4eacc1810f76656d1811693ff6c6737d2a。
3 chat prompts：天空、Fibonacci Python、中文过拟合。
300token/D3，每prompt先BF16后FP8，无重复ABBA：

| 配置 | 天空 tok/s | Python tok/s | 中文 tok/s | 合并 tok/s |
|---|---:|---:|---:|---:|
| 原精度 BF16 KV/A16 | 64.964 | 72.303 | 65.327 | 67.368 |
| FP8 KV/A16 | 66.148 | 64.692 | 62.841 | 64.532 |

FP8合并吞吐低4.21%。历史与接受率变化，这是生成趋势而非固定历史
attention kernel速度对照，不宣称广泛质量不变。
逐token耗时、接受量、验证次数、tokens、环境见precision_final_trend.json；
全部本轮历史screen.json。不存在本轮稳定新增降精度decode收益。

结合凌晨高收益机制的候选：
- BF16 norm输出，保留FP32router/compact mixed：
  68.011/70.477/69.608→69.220/65.651/66.524；
  token分歧56/41/91，未合入。
- 投影inline FP8量化输出逐位一致，但重复scale计算导致微基准更慢：
  gate1.090→1.291ms，down.535→1.038ms，未合入。
- mHC/RMS生产者量化一次，显式codes/scales：
  输出、量化字节、完整tokens一致；
  67.811/70.602/68.726→67.850/70.104/69.715；
  弱趋势且冷路径含编译成本，未采用默认。
  hc_quant.py、branch_quant.py、moe_norm_quant.py、job_13.py留在远端实验目录。
- RPS/bitdecode/graphobservation/KDA512/evalgroup/crossverify等弱候选
  与失败编译记录保留实验目录，未混入生产提交。

凌晨有效compact router、mixed gate/up/down、mHC边界融合、
private prefetch、verified context copy保留。
此前D3 53.269→67.997属于此前成果，不记作本轮新增收益。
fixed-address scratch/ring、更多graph/MLA边界融合尚未完成。

## 验证与源码边界

独立Git index/review_checkout与b6cb133d20最终tree一致；
排除继承startup warmup、draft末层FFN裁剪及packaging改动。
首次隔离遗漏encoder依赖及warmup移除语法问题已修复后重跑。

- 13 registered文件，116设备/运行时测试通过。
- 第一条pipeline提交单独通过mHC/mixed/router/context-copy/GLM/DFlash；
  FP8pair encoder缺依赖暴露后补入并device重跑通过。
- fatal Ruff、新文件import checks、gitdiffcheck、registered validator通过。
  现有A[...] doc注解F722误报忽略；旧文件import样式未全量重写。
  Black/Triton不可用，未安装依赖。
- 最终提交tree15完整模型runs；BF16/FP8各3prompts
  96token AR/D3一致，cacheoffset和最终sourcehash通过。
- BF16默认另3条192token：70.279/73.730/68.868 tok/s，
  合并70.901；这是最终原精度结果，不是新增量化收益。
- 主工作区30007 BF16默认HTTP17cases+concurrency+abort全部完成；
  长2563prefix、复用、seed、logprob、stop、abort后复用通过。
  最后3条96token SSE decode61.175/67.276/68.970 tok/s。
- HTTP包含继承warmup/draft末层FFN裁剪，和提交tree性能分开。
  final_sources.json分别记录committed/worktree SHA256；
  model_runner/glm5_next差异为未提交warmup。
- 首个遗漏process wired budget的worker为无效实验，排除其耗时。
  正式实验采用serving process预算，没有改系统wired memory；
  swap已存在，HTTP原始before/after记录于http_default.json。
- 同精度AR/D3一致不代表跨精度生成质量一致；广泛质量验证仍未完成。

## 复现与结束状态

默认：sh benchmark/mlx/decode_gain_20261105/launch_default.sh
服务PID2329，127.0.0.1:30007，结束时health200。
worker886已停止，oracle2112完整结束，没有fullmodel实验残留。

证据：precision_final_trend.json、screen.json、
commit_fullmodel/screen.json、commit_oracle.py/log、
commit_validation/checks.json、pipeline_validation/、
pipeline_dependency_final.log、http_default.py/json/log、
final_summary.json、final_sources.json。
commit_oracle.py复现保留review_checkout提交tree；HTTP使用主工作区，
不可混称源码一致。原始实验产物未纳入Git提交。
