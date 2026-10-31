# v3 第三轮独立验收（2026-10-31）

**结论：通过本轮明确界定的M4验收范围。**
包应用、已测算子数值、缓存/MTP提交语义通过；此前pool BF16可重复残余回退在v3
未稳定复现。公平一次gather对照下，独立attention应保守描述为持平至小收益；
mixed/dense MLA单层收益明确复现。没有放宽数值阈值或删除不利计时样本。

这不是M5生产通过，也不是任意形状/上下文/工作负载的性能保证。完整checkpoint
质量、自然MTP接受率、HTTP、长期缓存生命周期、目标机内存及tok/s仍待M5验收。
不从本轮M4算子/随机模型结果换算目标机吞吐。

## 包、源码与应用

包：`glm_mlx_revision_20261031_v3.tar.gz`

SHA256：`c562ca71bdc8f5f39bb9cbcb5d4fc1ad8298a9f8fd4e580b75eabfcc49c9543c`。

- 包哈希与用户提供值、package_v3_sha256.txt一致。
- 解压包中的14个源码文件与manifest及当前工作区逐一匹配；47个证据文件哈希匹配。
- 累计changes.patch、首次accepted输入的revision.patch、v1_to_v3.patch、
  v2_to_v3.patch四条路径分别在独立git目录应用检查并校验最终文件hash，全部通过。
- 相对v2只改mlx_ragged.py；相对v1另包含mixed-KV测试变化。
- 当前工作区portable/仍是旧版，所以本轮以**v3归档解压出的manifest/证据**为准，
  没有把旧portable目录当作v3。复跑脚本来自实际工作区，最终源文件都与v3匹配。
- 生产源码和用户测试未修改；没有提交/推送。14个文件AST及git diff --check通过。

见package_check.json、upgrade_check.json。check_package.py终端摘要沿用旧固定
“15 evidence”文字，实际循环与JSON验证47项；不影响检查结果。

## 改动审查

QK固定slot0/slot1移出K循环，符合当前32列tile里每个lane两次加载的映射，尾部
index读取仍由合法长度保护。Q/P、KV的TG暂存改回原T/V dtype，进入SIMD fragment
时转FP32；未改变输入数值、FP32 accumulator及投影/输出cast边界。
regular-tail-first、splitK-tail-last顺序保持。既有低精度值先升FP32再降回相同
低精度会得到原值，因此此变更可降低TG存储成本而不要求改变量化或数值容差。
本轮设备测试也未发现数值变化。

## 环境与检查范围

Apple M4 Pro/48GiB；MLX0.32.4、mlx-lm0.32.2、TF32=0，现有持久venv。
所有GPU测试顺序执行，作者已暂停其GPU实验。没有安装依赖、下载权重或改系统限制。
使用随机算子/单层权重与合成token，不是完整GLM原权重。

- 26项设备回归通过。
- 独立44组ragged扩展全部exact：FP16、BF16、FP32、FP32-query/BF16-KV，strided
  query/KV，毒化尾部、空行、长度8..4096，包含511/512/513、527/528/529等边界。
- MTP扩展通过：4层随机模型（含H64/D512 mixed MLA、KDA、MTP），steps1/2/4/8/16，
  双请求共80次decode，41次显式草稿消费、17次显式拒绝、remaining=1零步裁剪及
  释放，logits和每层cache/offset逐位一致。这些是人为轨迹，不是自然接受率。
- prefix snapshot/chunk handoff、zero-step lazy由26项回归覆盖。

runner通过AST绕过Torch/CI初始化执行真实生产代码和测试断言；未运行Torch oracle、
完整scheduler/allocator/HTTP，不把这些结果冒充完整CI或服务验收。

## 一次gather公平性能

v3包内random/pool基线已修正，每行构造一次selected KV，再同时作为K/V。
主脚本300次预热，20轮正反配对，每块32次调用；每路径共640次计时调用。
本轮还重用v2独立公平脚本，避免只复跑作者面板。

### 独立确认与稳态四面板

| 面板 | 原生一次gather ms | v3 ms | 解读 |
|---|---:|---:|---|
| pool BF16独立确认，20配对块 | .497355 | .490223 | 小收益，4/20块较慢 |
| random FP32独立确认，20配对块 | .549965 | .546103 | 近持平，7/20块较慢 |
| random FP32四面板轮 | .556604 | .538673 | 小收益 |
| pool BF16四面板轮，较热 | .719656 | .721826 | 约+0.30%，持平 |
| pool FP32四面板轮 | .535375 | .522424 | 小收益 |
| random BF16四面板轮 | .699975 | .690789 | 小收益 |

四面板轮每路径300预热、12块*32调用，正反交替。独立确认同样预热、20块*32调用。
输出全部exact。源自pool_bf16_confirm.json、random_fp32_confirm.json、
fair_attention_steady.json。独立脚本沿用的JSON key `v2`实际表示**当前v3导入函数**，
不是在该面板运行旧v2；该说明记录在label_notes.json，未改写原始样本。

### 包内主脚本复跑（不利面板也保留）

| 面板 | 原生ms | v3 ms |
|---|---:|---:|
| random BF16 | .629173 | .590937 |
| random FP32 | .655608 | .687362 |
| pool BF16 | .505033 | .485518 |
| pool FP32 | .541378 | .514105 |

该轮random FP32约+4.8%，短扩展面板也曾偏慢；因此额外做了上述充分预热独立
确认，得到近持平，另外四面板轮得到小收益。与v2 pool BF16多轮单向劣势不同，
本轮未得到稳定一致的残余回退证据。**通过不意味着每个样本或每个运行都更快**，
也不能将这些低幅变化宣称固定百分比提升。全部原始样本保留。

## 整层与routed复测

mixed/dense层为3次预热、4轮正反顺序、每路径8样本。所有输出exact，mixed
每位置cache也exact：

| 面板 | 旧路径ms | v3 ms |
|---|---:|---:|
| mixed MLA prefix16 | 2.051959 | 1.504374 |
| mixed MLA prefix2050 | 3.525208 | 2.135666 |
| dense BF16 prefix16 | 3.162146 | 1.623625 |
| dense BF16 prefix2050 | 3.767646 | 2.170000 |

routed主基线为旧生产mlx_affine_rows + compiled activation，300预热、20对*32：
4bit .683092→.655914ms，5bit .981493→.965728ms，全部exact。
4bit局部小收益，5bit继续保守视为近持平；native oracle单列，未用作生产提速倍数。

峰值MLX allocation：mixed+routed组合脚本11.251GB、dense5.890GB、扩展算子
.902GB、MTP扩展.778GB；包含随机权重生成临时量，不是生产常驻/全模型内存。

## 采用与剩余工作

通过已测M4范围，可保留当前候选默认覆盖路径并进入目标机阶段；本轮没有发现
需要继续阻塞该范围的功能、数值或稳定性能回退。维持原有未覆盖形状说明，不把
H64/D512、8..4096之外称为已优化。

仍未实现或验证：跨token compact专家权重复用、跨请求逐层权重复用、完整MTP
overlap、draft head主要矩阵成本优化；真实checkpoint的路由/自然接受率与质量、
长上下文和完整HTTP生命周期、目标机峰值内存与吞吐。不要把本报告用于无条件
生产通过，也不要承诺达到公开竞品tok/s。

M5下一阶段应固定原checkpoint revision/量化/TF32与请求，交替原基线/候选顺序，
保存原始latency、内存、logits/KL/top1、路由及自然接受率，并覆盖普通decode、
MTP拒绝/消费、chunked prefill、prefix exact/suffix、sampling/grammar/logprobs、
并发、断开与释放。上述是目标机待办，不属于本轮M4已通过范围。

## 交付状态

设备实验全部结束，无后台GPU任务，可以恢复作者实验。
报告、原始JSON/log和复现脚本保存在本目录。最终源码、包和证据指纹见
acceptance_manifest.json；v3压缩包及解压manifest是本轮权威输入。
