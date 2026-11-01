# GLM MLX 优化实现与验收

2026-11-02。remote `/Users/jay/workspace/sglang`，branch `metal`。基准 `6310a056f6e6bae388aa328cb9aa6168841be049`。代码提交 `07ef2e5b6759dfd4b445d5204b2b54d5e95cb957`（E）与 `3e678036ed`（显式验证可见范围）。随后合并 upstream/main `e2982985110fb432fd8ddba5f0147a5004fd03ab`，合并提交 `124fbf132baf538916417e81a7cfd939490246c0`。没有推送、权重传输或安装。继承 pyproject 修改及删除按原始 SHA 精确恢复，未提交。

设备为 M5 Ultra 256GB，MLX 0.32.4 / mlx-lm 0.32.2。checkpoint `mlx-community/GLM-5.3-Flash-4bit` revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`，保留原始混合4/5/6bit和BF16，TF32=0。独立进程顺序加载，不改系统wired设置。

## 采用与移除

| 工作包 | 决定与证据 |
|---|---|
| A | 完成自然R2/R3/R5与2050AR1路由、真实操作数及对称控制。2889记录；R3重复picks约33.6%。有encoder证据，无shader/DRAM/occupancy/spill归因。 |
| B | 多轮均衡/register/shared调度调整后，最佳候选仍慢2.43% elapsed；六个自然R3窗口均回退。移除生产候选，保留原ordered down。只复用当前tile/Kchunk的packed读取，不是完整decoded缓存。 |
| C | 独立shared/routed gate dispatch五组全模型回退：短/长AR -0.795/-0.786%，MTP2 -1.642/-1.011%。数值/token/acceptance一致；移除具体失败机制，不归为Metal限制。 |
| D | 正确BF16 deferred mHC AR慢8–11%、MTP慢5–6%；多轮依赖/geometry/projection改进仍慢。移除packet/deferred候选，保留原方程与BF16边界。 |
| E | 默认采用同输入multi-QMV，独立weight/scales/biases及promotion，不拼接完整权重。load/live graph observation准备stride；clone传显式数组和layout key。整组资格含dense时保持原独立投影，不融合子集。移除未采用static masks实验。 |
| F | F4长MTP2五组吞吐 -0.284%，五组都负；compact/stream迭代亦回退，移除融合kernel及选择接口。保留运行时split、每row可见范围、selected length和内部合法索引合同；原AR不变。 |

E-only归因对照：短MTP2 +0.729%（五组均正），长MTP2 +0.028%近中性；AR组间方向混合。其两侧共同准备，不能代表相对原HEAD的总收益。

## 原HEAD与最终源码对照

五组交替ABBA/BAAB，20个独立模型进程；每个覆盖32/2050输入、AR/MTP2、两次重复、96个committed调用；cold5与steady91分列。所有token hash与acceptance-by-depth/budget一致。下表为稳态committed tok/s；不将算子局部收益相加。

| Prompt | 模式 | HEAD tok/s | 最终 tok/s | 变化 | 五组变化% |
|---|---|---:|---:|---:|---|
| 32 | AR | 37.0872 | 36.8019 | -0.769% | -0.479, +0.280, -0.450, +0.212, -1.397 |
| 32 | MTP2 | 50.8729 | 51.0461 | +0.341% | +0.128, -0.056, +0.090, -0.041, -1.658 |
| 2050 | AR | 33.8968 | 33.7847 | -0.331% | -0.736, +0.730, -1.292, -0.308, +0.044 |
| 2050 | MTP2 | 51.2237 | 51.0013 | -0.434% | -0.436, +0.792, -2.980, -0.946, -2.360 |

原HEAD整体对照没有稳定吞吐提升：短MTP2汇总 +0.341%，短AR/长AR/长MTP2汇总 -0.769/-0.331/-0.434%。长MTP2四组为负，不能用E-only短MTP2增益掩盖此结果。E本身独立对照长MTP2近中性、短MTP2五组均正；最终还包含显式验证可见范围合同。整体差值的E/F归因及环境波动未完全分离，保留这一性能限制，不宣称整体性能优化完成。

完整对照固定在合并前immutable源码freeze；合并后GLM与kernel SHA未改变，服务/调度另测，不能把upstream变化混作E加速。

全进程启动ABBA一组：HEAD 16.8424s，最终 16.7339s，elapsed变化 -0.645%。这是新Python进程，未flush OS文件缓存；不是独立shader cold或E-only准备成本。

## 正确性与服务

合并后设备/模型81项通过，无跳过；MLX scheduler/runtime16项通过。E-only中间版本71项数值/模型检查通过，不冒充完整模型性能。final scoped lint无新增诊断，继承baseline诊断保留。注册检查和diff --check通过。

合并前完整HTTP四配置AR/MTP2 × slots1/2，以及合并后MTP2双槽HTTP均通过：32/320/2050 prompts、chunked prefill256、cold/prefix精确输出/logprob/top-logprob、stop、terminal abort SSE、after-abort prefix/fresh槽位复用、并发streams及跨配置一致性。32-token prompt低于辅助快照步长可cached0；320明确cached256。context配置32768/max-total65536不等于已验证8K/32K模型容量。

上游机制审查见 `UPSTREAM_REUSE_REVIEW.md`：KDA去重conv snapshot布局、mHC小batch分割输出、混合长度indexer的tile均衡值得后续验证；MoE路由缩放/greedy直接argmax已有对应机制。owner-list与abort修改是服务生命周期集成，不记作吞吐收益。本轮不再启动新的候选实验。

## 解包缓存实验

真实选定layer3/4 Q5 down，完整mask12缓存12MiB/expert，保持原12-term qdot顺序；R3 down相对原生产慢9–10%，R1混合。FP32完整down缓存32MiB/expert，命中路径比packed慢6–29%，不逐位一致的差异保留为数学顺序诊断。没有稳定hit收益，未合入LFU权重池。CPU有限路由LFU模拟不代表异步GPU或完整模型稳态；Q8/async/v2未执行的helper不作为验证。

## 复现与证据

使用已有 `/Users/jay/workspace/prod/common/sglang-metal/bin/python`，`MLX_ENABLE_TF32=0 PYTHONPATH=python`。同目录 `results/` 保留完整原始路由、各候选五组记录、freeze源身份、HTTP raw/server log、memory/swap、数值诊断；这些大型实验目录未整体入git。`accepted_evidence/`保存紧凑原始五组A/B样本、最终manifest与验收报告，不含权重/venv/capture。

全模型复现（从本任务目录执行）：`compare_final.py --baseline results/original_head_final --candidate results/closeout_final_freeze --groups 5 --harness isolated_model_v2.py --output results/reproduced_original_compare`。一次只运行一个设备实验。freeze由git blob精确基准构建，manifest必须先核对。

最终模型/算子测试入口为已注册测试：`test_mlx_affine_rows.py`、`test_mlx_affine_multi_prepared.py`、`test_mlx_ragged.py`、`test_mlx_glm.py`、`test_mla_prepared_strides.py`、`test_glm5_next_mlx.py`。HTTP：`final_http.py --model-dir <existing checkpoint> --source-manifest results/merged_manifest.json --steps 2 --max-running 2 --output results/reproduced_merged_http --run`。历史移除候选的脚本依赖其历史freeze，不能当作当前生产API。

未验证：其他设备/量化/checkpoint、8K/32K完整模型capacity/吞吐及完整LFU异步池。没有无依据放宽数值阈值，未静默更改模型方程。
