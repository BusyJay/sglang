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

此前独立Git index/review_checkout与b6cb133d20 tree一致；
当时排除继承startup warmup、draft末层FFN裁剪及packaging改动。
首次隔离遗漏encoder依赖及warmup移除语法问题已修复后重跑。
现已另行提交warmup（3b9c6993d3）和draft末层FFN裁剪（40d5b1ed3a），
运行中的生产源码全部进入metal提交；packaging按用户要求保持未提交。

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
- HTTP包含warmup/draft末层FFN裁剪，其性能与此前隔离tree数据分开。
  这两项源码保持HTTP验证时的实现，本次提交没有重新做吞吐实验。
  原final_sources.json保留历史committed/worktree SHA256；
  remaining_runtime_commit/final_state.json记录当前提交与源码哈希。
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

## 剩余运行时代码提交（2026-11-05）

- 3b9c6993d3：启动时实际求值小批量NVFP4 FFN及decode/verification图，
  使用私有缓存，不注册请求、不发布prefix、不更改RNG。
  仅预热配置形状，后续未见形状仍可能编译，不宣称新增稳态吞吐。
- 40d5b1ed3a：DFlash保留完整8行attention KV/queries，仅裁剪末层FFN
  不被短草稿使用的尾部，最低6行维持原生矩阵归约。
  1/3/5/7草稿的hidden与tokens前缀逐位一致，不修改checkpoint精度。
- 本次最终源码通过4个registered文件、84项测试：
  startup warmup 1、DFlash2 6、GLM 65、DFlash serving 12。
  包含实际Metal执行、独立BF16方程/缓存oracle与请求生命周期检查。
  Ruff fatal/import/format、registered validator、git diff检查均通过。
  完整日志和提交前后源码哈希位于remaining_runtime_commit/。
- 已有完整模型/HTTP结果沿用上述原始记录；本次未追加性能实验，
  不将两项提交合并后的独立吞吐增益作为已测结论。
- pyproject.toml及pyproject_other.toml的已有工作区状态与哈希保持不变；
  benchmark原始日志、快照和诊断脚本保持远端实验记录，未推送。
