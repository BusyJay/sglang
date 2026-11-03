# M4 FP8 算子链根因与改进验证 — 2026-11-03

用户指出此前候选无收益后没有追到根因。本轮在远端不可用期间，使用本机 M4 Pro 实测一次完整的“回退复现 → 成本归因 → 消融 → 机制改进 → 链级验证”。
没有访问远端，没有安装依赖或下载权重，没有提交/推送。这里不是 M5 全模型吞吐报告。

## 硬件、工作负载与边界

M4 Pro 48 GiB、applegpu_g16s、MLX0.32.4，现有 `.venv/mlx-dtype-audit/bin/python`；Xcode/MetalToolchain与 gpudebug 已安装。
进程内存上限3GiB、allocator cache256MiB；实测峰值约193–195MiB；系统 swap 全程检查为0。
使用合成 packed E4M3 权重、原 /128x128 FP32 block scales和/32动态激活量化。
八层不同权重的 shared FFN：gate/up各[2048,4096]、down[4096,2048]，每层有clamp/SwiGLU及残差；对应归档目标模型的 shared FFN形状。
每个ABBA样本运行连续300步、每步八层依赖链；分开1/4/8输入行。每组两层mx.compile。原始逐步walltime保存JSON。
权重工作集仅约192MiB，没有复现171GiB模型的带宽/缓存行为，也没有真实路由、attention、draft、prefix cache或HTTP。
因此M4百分比不能换算M5 tok/s；它能证明该FP8实现机制在本机的链级效果，并筛除具体错误设计。

## 最终源码实测

当前源码哈希见 final_summary.json / final_source/。每行两次baseline、两次candidate，均为300步；数值记录包括连续300步最终输出精确相等。

| 输入 | baseline ms/八层 | candidate ms/八层 | 延迟下降 |
| --- | ---: | ---: | ---: |
| 1行FP32、每2层compile | 2.1794 | 1.8074 | 17.07% |
| 4行FP32、每2层compile | 6.0127 | 3.4972 | 41.84% |
| 4行BF16、每2层compile | 5.9706 | 3.4173 | 42.76% |

原始：ffn_macro_g2_c1_fp32.json、ffn_macro_g2_c4_fp32.json、ffn_macro_g2_c4_bf16.json。
前期未compile的8行链11.3446→6.4433ms；这组只确认一次八层输出精确，没有记录300步最终比较，且生产文件后来仅格式调整，故不作为最终源码主表。
其他旧JSON同样保留其原始源码哈希，不冒充最终文件。个别早期输出文件被后续实验覆盖，ffn_initial_c4.json名称也不代表首次全模式实验；首次全模式结果仅有ffn_macro_c4.log，不能称其有完整逐步JSON。

## 回退原因被怎样确认

1. 直接FP8编码替换二分搜索：1018边界样例、BF16/FP32随机输入及倍率精确，孤立eval约.11ms且无稳定收益。
2. 只共享量化：八层链baseline约6.41ms、candidate约6.36ms，没有可靠收益。
3. 成对投影但仍每组1输出行：同轮6.1761→4.8203ms，约22%延迟下降。
4. 成对投影＋每组2输出行＋等价half中间解码：最终compile链改善约42%。
5. 回退复现：投影tile内重复最大值/尺度/编码，八层4行链baseline约6.416→17.363ms。两个candidate样本17.374/17.351。
6. 进一步独立消融，真实形状[2048,4096]4行单投影：预量化.4140ms；只重复编码.7682ms；重复编码＋尺度1.0163ms。[4096,2048]分别.2982/.4598/.6228ms。
   所以编码重复与尺度重复都单独造成回退，不能用少了dispatch掩盖增加工作。原始 repeated_ablation.json。

新pair同时保留gate/up两个独立输出，未融合激活、更没有偷偷移除FP8激活量化；下一步可继续追down/producer，但本轮收益无需改模型方程。
投影几何独立消融覆盖RPS1/2/4、branch/half decode、不同N/K/count；projection_ablation.json及final.log保留9轮AB/BA。
[2048,4096]count4：separate约.7470ms，branch pair1 .4796、pair2 .3695、half pair2 .3436；输出几何收益大于解码替换。
这不是所有形状的全局最佳geometry；较大投影有RPS4更快，本轮正式API选RPS2覆盖主要sharedgate/up形状，其他形状仍待全链调优。

## GPU证据及限制

干净2层capture使用独立新stream，确保所有2层dispatch在capture内：

| 模式 | Commands | GPU replay time | gate/up shader spilled bytes |
| --- | ---: | ---: | ---: |
| baseline | 30 | 2.10ms | 0 |
| 改进pair | 22 | 1.18ms | 0 |
| 重复量化 | 36 | 6.14ms | 0 |

baseline projection shaders cost约98.65%，quantization约.84%；说明先前只优化编码不够。
baseline gate/up shader临时寄存器34，改进pair46，重复量化42，均零spill；不能把回退说成已证实寄存器溢出。
baseline shader341指令，重复量化538指令；这是编译shader静态信息，不是动态指令总量或GPU利用率。
较多scalar placeholder copy仍存在；改进没有解决所有dispatch生命周期问题。
GPU profile在replay默认performance-state/overlapping执行，独立于walltime ABBA，不使用二者相加；对照采集顺序不是GPU ABBA，不将replay数字作为稳定收益百分比主证据。
初始八层capture捕获了部分command buffer，因此gpu_metrics/gpu_pair_metrics等早期数字只用于诊断；最终结论使用*_final_metrics.log和repeated_gpu_metrics.log，已按每个shader调用次数核对。
.gputrace约数GiB，仅本机保留，不放便携包；gpudebug摘要日志保留。

## 源码、正确性与接入

新增生产候选：
- python/sglang/kernels/ops/gemm/mlx_block_fp8.py：paired block-FP8 projector，gate/up共用quantized输入，保持原32lane reduction、分别round到输出dtype；K/N/count运行时参数，count最多4行一tile，任意正count多tile。
- python/sglang/kernels/ops/quantization/mlx_fp8_dynamic.py：等价位编码器；单项无吞吐收益，作为明确输入量化producer用于pair链。保留原scale公式及underflow行为，没有静默换量化。
- 对应逻辑组lazy KernelSpec；registered numerical test test_mlx_block_fp8.py。

测试在现有MLX-only环境运行真实kernel及独立oracle：2个registered方法通过。test_utils不可完整导入（torch/msgspec等缺失），run_device_tests.py只抽取CustomTestCase资源清理方法并绕开CI环境依赖；不是普通完整registered runner通过。
覆盖codebook中点/邻值/奇偶tie/正负零、blockscale/tailrow、count1/3/4/5/8/17、BF16/FP32、NaN各bank隔离；独立单矩阵shader保留原lane/reduction。
编码消融256codes比对有限值bit模式与NaN位置；float极端幅值额外诊断保持原scale underflow行为。
实际归档MLP/FP8Linear经AST isolation运行，original/patched count1/4/8、BF16/FP32各连续300步精确，额外bias检查通过；mlp_integration_check.json。
实际补丁保留原binary quantizer，compiled单MLP ABBA300：1行0.3903→0.3301ms（15.41%下降），4行0.9181→0.5574ms（39.29%下降），最终300步精确；actual_mlp_abba.json。因此接入收益无需替换量化编码器。
没有完整GLM路由/接受率/缓存生命周期验收；也没有目标机M5 fullmodel性能与质量验收。
CI registration lint、no-registered-tests-in-package及git diff--check通过。Ruff/Black/precommit均未安装，未安装新工具；AST检查只标为语法检查。

本地checkout缺少远端FP4模型实现；没有复制整个远端模型覆盖本地。远端MLP接入补丁 remote_shared_fp8_integration.patch 对应HEAD303af39b966c48e18d2268bdd610405e1316dc74、glm5_next.py SHA2562c2f0c3afb0fd7e53988aaaed82922e2d54bdadc9427680aeb4f148a1ee35691。
补丁在隔离源副本apply-check及实际MLP检查通过，含bias语义；尚未应用远端。原baseline算子不变。
这属于已完成M4机制改进、待M5集成验证，不能说生产吞吐目标已达成，也不以默认关闭结束。
NVFP4专家worklist、完整MoE fusion和M5矩阵分派仍未完成，本轮没有把FP8链结论外推到它们。

## 复现

工作目录 `/Users/rustyli/workspace/sglang`，使用已有 `.venv/mlx-dtype-audit/bin/python`。
先运行本目录 `quant_check.py`、`projection_ablation.py`、`repeated_ablation.py`、`run_device_tests.py`。
主要链基准命令：

```sh
.venv/mlx-dtype-audit/bin/python benchmark/mlx/m4_fp8_rootcause_20261103/ffn_macro.py --count 4 --steps 300 --modes pair --compiled-group 2
```

BF16增加 `--dtype bf16`；1行使用 `--count 1`。实际MLP检查需要先把archive复制到 `integration_check` 再apply上述patch，运行 `mlp_integration_check.py`。
原始结果按原文件保留，脚本重跑会覆盖对应JSON，复现时先另建结果目录/修改OUT以保留旧证据。
