# Revision v3 evidence — 2026-10-31

已通过第三轮明确界定的M4独立验收范围。M4 Pro48GiB、MLX0.32.4、mlx-lm0.32.2、TF32=0。
均为随机算子/真实形状单层，未执行完整GLM checkpoint、HTTP或M5吞吐验收。

v2独立验收定位主基线重复gather，v3已纠正random/pool脚本为一次gather，K/V
共用同一array；不再将v2双gather对照的收益作为生产成本证据。旧包与独立报告保留。
主算子脚本每路径300次预热、20组正反交替配对、每块32次调用，保存全部样本。
另保留v3 steady四面板及pool BF16独立确认，不删较热状态与不利结果。

| 一次gather主对照 | 旧中位ms | v3中位ms | 输出 |
|---|---:|---:|---|
| random BF16 | .510648 | .488438 | exact |
| random FP32 | .561088 | .542490 | exact |
| pool BF16，较热状态 | .718454 | .717523 | exact，持平 |
| pool FP32 | .534251 | .518255 | exact |
| pool BF16独立确认 | .491447 | .485492 | exact，小收益 |
| routed4bit旧生产affine+compiled activation | .678079 | .654964 | exact |
| routed5bit旧生产主基线 | .974883 | .958413 | exact，保守视为持平 |

另一组steady四面板为random FP32 .547712→.524296、pool BF16 .564539→.549074、
pool FP32 .534044→.511813、random BF16 .627488→.586529ms。存在跨块/进程状态
波动，不能承诺固定收益。routed native oracle单独列出，不作为旧生产主基线。

v3保留v2 vec4读取和PV按step staging，增加两项机制：QK固定slot读取移出K循环；
Q/P与KV的threadgroup暂存保留输入T/V，再在SIMD矩阵加载时转FP32，减少低精度
暂存带宽与容量。FP32 accumulator、dtype边界、regular-tail-first/splitK-tail-last
归约顺序不变，不排序索引，不改量化与误差阈值。阶段正反配对中位v2→v3：
QK .296061→.284723、softmax .136734→.136304（代码不变）、PV .316190→.305999ms。
非配对阶段trial结果也保留，不以其漂移结果独立归因。

最终真实形状整层（3预热、4轮正反、每路径8样本）：mixed MLA prefix16
2.103334→1.545687、prefix2050 3.443354→2.093750ms；dense BF16 prefix16
3.136208→1.582104、prefix2050 3.985333→2.244292ms。输出与mixed每位置cache exact。
bench_revision峰值11.251GB（十进制），dense5.890GB；随机权重，非完整模型内存。

26项真实设备回归、44组strided/毒化尾部/空行/FP16/BF16/FP32/mixed边界均exact。
MTP扩展steps1/2/4/8/16、双请求80次decode、41次强制草稿消费选择、17次拒绝选择、
remaining零步裁剪与释放均通过；人为轨迹不代表自然接受率。prefix/chunk helper
由26项覆盖。AST绕过Torch包/CI初始化，仅执行生产模块和注册断言，不代表HTTP。

默认保留covered shared投影、KDA所有快照、routed gate/up融合、ragged机制。ragged
覆盖H64/D512合法长度8..4096与空行，其他几何使用通用批量attention。未完成：
跨token expert compact权重复用、跨请求逐层权重复用/MTP overlap、完整draft head
带宽优化；不将已有top1 epilogue包装成实质吞吐收益。M5同checkpoint logits/KL/
top1/路由/自然接受率、长期缓存释放与HTTP及tok/s仍需生产机器验收。

累计/首次验收输入/v1/v2增量应用及逐文件hash检查见包；基准commit与源码/证据
fingerprints见manifest。git diff --check、AST通过；未安装格式器，ruff不可用。
不含权重、环境与庞大capture，未修改系统资源参数，未提交/推送。

最终交付包：生产源码与已验收v3完全一致。两份基准脚本仅删除未使用的v1目录加载，见delivery_script_changes.patch；独立验收报告及原始证据在acceptance/。该报告验收的原v3包哈希与最终交付归档哈希不同，源码哈希一致。M5仍待目标机验收。
