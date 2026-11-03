# GLM Metal 优化账本：做过什么、为什么排除、还缺什么

2026-11-04 整理。覆盖当前 GLM-5.3-Flash FP4 checkpoint 的 AR/DFlash 性能探索：
此前 M5 短生成、远端不可用时的 M4 机制实验、最后的 M5 完整模型回访。
早期 64/128-token 结果单独标注；最后两轮主要使用 300-token 生成或 300 步宏基准。
历史报告保留当时的决定；最终采用状态以本账本与 [README.md](README.md) 为准。

## 结论的边界

**没有达到 AR40/DF90，不等于没有优化空间。** 已证明的是一些具体实现无稳定收益或
回退，并非 Metal 带宽、ALU 或 neural accelerator 的上限。trace 的 95.45% compute
调度 duty 不能当硬件利用率。当前 checkpoint 内部有对照；上一 checkpoint 与当前
checkpoint 同机、同请求、相同量化与完整服务配置的因果对照仍不完整，尚不能将新权重
全方位较慢归结为单一原因。

“不采用”只针对测过的实现与范围。“未运行”不计为失败；小范围精确不代表全模型
质量/缓存语义已证明。rounding 差异是实现定位问题，不是 Metal 不支持批量的证据。
M4 百分比不能换算 M5 tok/s，不同机制的局部收益不能相乘。下面 DF 指 DFlash2，
配置三条 proposals；CLI block size 4 包含 anchor。

## 早期 M5：基础优化与第一轮短生成

环境、commit 与最后一轮不同，不能拿跨表差值作收益。完整历史报告：
[AR](../ar_optimization_20261103/README.md)、
[FP4 DFlash](../fp4_dflash_20261103/README.md)、
[300-token 第一轮](../throughput_20261103/README.md)。表中相对文件名分别位于这些目录。

| 机制/假设 | 实测范围与结果 | 决定、排除依据及仍未证明的部分 | 原始证据/脚本 |
| --- | --- | --- | --- |
| MXFP8 去掉四行零 padding，真实行共享解码权重 | M5，128-token 六对 AR，27.9108→29.5404 tok/s；40 个真实投影精确、ID 一致 | 已采用基础机制；保留 KL16 lane/reduction 和原权重。没有消除全部首形状成本 | AR `final_ab.json`、`final_ab.py` |
| block-FP8 将 scale 读取移至 K128 block 外 | 36 个真实投影精确，64-token 六对 ID 一致；结合上一机制 128-token 对照 28.0405→30.6915 | 已采用；组合收益不能拆分相加。远端历史代码提交 `415e14a561`、`4ac0c74624` | AR `block_hoist_ab.json`、`combined_ab.json`、`combined_ab.py` |
| 固定 calibrated NVFP4 global scale 缓存 | 30.707/30.696 vs30.718/30.700 tok/s | 无有意义收益，不增加可变缓存所有权；不能证明所有 setup 都无成本 | AR `scale_cache_ab.json`、`scale_cache.py` |
| padding16/32 行触发 native matrix/NAX selector | AR 约24.4 tok/s；零行也需计算 | 实现回退；selector 阈值属于 MLX 实现，不是 Metal 无法做窄矩阵。不采用 TF32/dtype 改动 | AR 报告、`nax_ab.py` |
| 自定义窄 NAX/MPP、split-K、每形状 RPS | 真实 dtype 微基准不优于精确 wide；RPS 全 AR 28.04→27.1 | 未采用，仍缺高效 MMA/dequant 几何与真实集成。不同 reduction 需单独验证 | AR `mxfp8_nax_test.json`、`mxfp8_nax_test.py`、`operator_tune.py` |
| 窄 native-like NVFP4 投影 | AR29.55→28.77；DF ID 在 token23/95/78 改变，接受计数改变；FP32 K4096 max3.05e-5/RMS6.59e-6 | 性能与输出回退；reduction/calibration 定位未完成。不是仅以 bitwise 差异否定批量 | AR `nv_final.json`；DF `narrow_ab.json`、`narrow_numeric.json`、`narrow_ab.py` |
| E4M3 位编码替代 binary search | 边界/随机编码精确；全模型差值在基准波动范围 | 单项成本不足以带来已测吞吐收益；producer 的后续作用另测 | AR `quant_full.json`、`quant_direct.py` |
| ICB + argument buffer 提前组织提交 | 128 dependent synthetic dispatch：CPU submit .0528→.0147ms，FP8 completion2.4668→2.4131ms；vector GPU 更慢 | SDK能力已试；MLX buffer生命周期、stream依赖与全模型集成未完成，不能称已排除ICB | AR `icb_probe.mm` 与报告 |
| DF proposal width1–5 | 128-token 六对：AR30.5324；DF1/2/3/4/5 为41.8362/49.7967/51.1928/50.7700/51.2572 | 部署3；5有时利于code、损害解释，不是通用最优值；没有证明能达到90 | DF `widths.json`、`widths.py` |
| verify eval group、只共享FP8量化、expert排序 | group8回退，1/2/4小漂移；量化共享50.4509→50.6475；排序50.4366→50.5210，ID/计数一致 | 保留group2；单项无可靠全模型收益。没有覆盖全部负载 | DF `groups.json`、`ffn_ab.json`、`sort_ab.json` 及同名脚本 |
| KDA MXFP8精确half-value缓存 | 300-token24次：AR30.4297→30.5096、DF49.8746→50.2217；+2.125GiB | <1%且无稳定收益，不采用；不据此否定所有权重预组织 | 第一轮 `cache_ab.json`、`cache_trial.py` |
| dynamic quantization融合进输出projection tile | 300-token24次：AR30.4148→25.8788、DF50.2224→35.9245，生成精确 | tile重复尺度/编码，少dispatch却增加工作；后续M4独立消融确认 | 第一轮 `quant_ab.json`、`quant_trial.py`、`fp8_quant_fused.py` |
| 旧同专家多token复用、动态accumulator | 300-token12次DF50.1979→45.1682，ID/接受改变；count4 micro约.26→.33ms | 旧实现回退；后续定位rounding/spill并重写，不能永久排除专家复用 | 第一轮 `shared_ab.json`、`nv_shared.py`；下节 |
| getter预规划图观测检查 | 300-token24次AR30.5297→30.4752、DF50.2734→49.6867，ID/计数一致 | 无收益不采用；最后一轮进一步复验observer bypass | 第一轮 `observation_ab.json`、`observation_trial.py` |
| 用fixed MLA layer编译单token MLA | 45层输出/cache/offset精确；300-token12次AR30.6154→30.0156 | 这个封装回退约2%；未证明attention已最优或编译永远无益 | 第一轮 `mla_ab.json`、`mla_numeric.json`、`mla_trial.py` |
| 早期NVFP4 gate/up合并 | 真FP32 max约4.8e-7，guard在完整ABBA前终止 | 只有诊断，无完整性能结果；后来继续定位，没有放宽tolerance | 第一轮 `pair_trial.json`、`pair_trial.py` |

## M4：机制改进、数值根因与预热

M4 Pro48GiB、`applegpu_g16s`、MLX0.32.4，复用 `.venv/mlx-dtype-audit` 与现有
Xcode/MetalToolchain。实际宽度的合成packed权重，工作集远小于远端171GiB。
局部设备测试有隔离导入glue，不能称完整serving runner通过；详情在各历史README。

| 机制 | 结果与已定位原因 | 最终状态与边界 | 原始证据 |
| --- | --- | --- | --- |
| shared FP8成对投影、RPS2、等价half中间解码 | 八层group2 compiled ABBA300：FP32一行2.1794→1.8074ms，四行6.0127→3.4972，BF16四行5.9706→3.4173；300步精确 | M4改进成立；实际MLP一/四行.3903→.3301、.9181→.5574。后续M5只有modest DF收益 | [FP8报告](../m4_fp8_rootcause_20261103/README.md)、`final_summary.json`、`ffn_macro_g2_*.json`、`actual_mlp_abba.json` |
| 编码器/共享量化/重复量化消融 | 只共享量化约6.41→6.36ms；重复量化6.416→17.363。单投影预量化.4140、重复编码.7682、编码+尺度1.0163ms | 编码与尺度重复各自导致回退；不能笼统归因spill。正式pair可继续使用原quantizer | FP8 `repeated_ablation.json`、`projection_ablation.json`、同名脚本 |
| 旧NVFP4增加GPU worklist/四token static accumulators | worklist-only四层7.597→9.013ms；旧动态/四static均336 spilled bytes；后者静态指令356→650 | worklist单独不足；强展开更差。随后改为两pick有界复用 | [NV根因报告](../m4_nvfp4_rootcause_20261103/README.md)、`profile_shaders.log`、`moe_macro_real_c4_fp32_4layers_pair,list.json` |
| NVFP4 cross-block rounding修复 | K512精确，K1024/4096不同；只cross-block scale*accum禁contraction恢复精确，全局禁contraction反而错误 | 一处数值根因已确认；scoped pragma替代volatile产品后zero spill。去guard mutant变1321值，正式变0 | NV `parity_probe.json`、`regression_proof.json`、`profile_final_shaders.log` |
| 两pick/static accumulators/RPS4 | 四层四行FP32 7.4482→5.4117、BF16 7.5128→5.3909ms；实际类2.0560→1.4870，300步精确 | 曾进入远端默认候选patch，M5完整模型回退后移除。不能用M4约27%承诺DF90 | NV `final_summary.json`、final macro JSON、`integration_check.json` |
| 八输出singleton tile/无重复routing | 八输出无重叠链7.584→8.002ms；保留四输出无重叠显式/inline约+1.16%/+0.78% | 八输出排除；仍有无复用时控制开销，不能宣称全部route加速 | NV报告、`iterations/production_before_single8.py`、`single8_trial.py` |
| AR gate/up+clamped SwiGLU融合，保留dtype rounding | 八层median4.1567→4.0062ms；实际类median.6261→.6162，但含慢样本的mean.638247→.636507，仅.27% | M4 AR收益小且不稳定，M5 fullmodel未受益。direct/branchless/half/vector替代只是诊断 | [AR pair报告](../m4_nvfp4_pair_20261103/README.md)、`final_summary.json`、`integration_check.json` |
| eager kernel+同一compile owner启动eval | FP32未热first10.015/5.175→warm2.176/2.210ms；warmup4.940/5.108，BF16类似，steady无收益 | 一次性预热有效；构图不eval mutant失败。未清OS cache；未覆盖所有verify/并发/layout/增长 | [warmup报告](../m4_nvfp4_warmup_20261103/README.md)、`chain_*.json`、`chain_summary.json`、`device_mutant.log` |

M4干净两层FP8 capture command30→22，replay2.10→1.18ms；重复量化36/6.14ms，均
zero spill。capture不是GPU ABBA，只作机制证据。早期capture不完整，不作主性能结论。
历史源各有哈希，不冒充最后M5格式/scale修复后的源码；M4表中“采用”已由M5更新。

## 最后M5：从micro到全bank、完整MoE、完整生成

统一环境/checkpoint见README。所有macro ms是各自脚本walltime，不可相加估算模型。
表中结果路径均在 `results/`；raw_seconds、顺序、source_sha256、内存/接受字段原样保存。

| 实验/机制 | 同实验内原始结果 | 排除依据、最终决定与未解决部分 | 脚本 → JSON |
| --- | --- | --- | --- |
| 四模式同驻留全模型 | 48次300-token：baseline AR/DF30.6122/50.9610；NV30.4364/50.3191；FP8 30.4410/52.8741；combined30.2393/49.8727 | FP8 DF约+3.75%、AR约-.56%，未建置信区间；NV/组合回退，ID/接受精确。回退不是接受率降低 | `fullmodel_ab.py` → `fullmodel_ab.json` |
| M4同一个合成四层macro到M5 | 1.611→1.518ms约5.8%，M4约27.34% | 设备差异已实测；M5合成链也不能预测fullmodel | `m5_synthetic_macro.py` → `m5_synthetic_macro.json` |
| 真route/expert overlap | 全42层picks配对均值36.9048%，旧M4 pattern68.75%；26/35/41高、早层低 | 复用机会缩水，但不能独自解释live graph反转；仅一个采集边界，不是全请求分布 | `rootcause_replay.py`/`tile_replay.py` → 同名结果 |
| full-bank data_size触发提交假设 | threshold1024/8192/32768MiB对AR/候选无实质改善 | MLX输入accounting可能按整bank约1.125GiB，只调threshold未解决，未证明主因 | `buffer_ab.py` → `buffer_1024.json`等 |
| compact/full实际bank四层300步 | full native/candidate1.784/1.826；compact1.886/1.820ms | compact单独没恢复M4收益；归一化recurrence只是诊断，不是完整模型工作集 | `working_set_ab.py` → 同名结果 |
| closure route→runtime ids、热层geometry | dynamic ids层26 native/reuse2 .5095/.4720；35层.5061/.4675ms；closure结果不同 | 固定route图影响测量，未证明constant folding是主因 | `ffn_geometry.py`/`ffn_dynamic_routes.py` → 同名结果 |
| 全42 FFN bank轮转 | native14.857/RPS4 18.273/AR-fused15.756ms | hot层收益不能覆盖跨bank；随后继续geometry并全模型复验 | `full_ffn_sweep.py` → 同名结果 |
| 全bank RPS1/2/4/8、inline | native14.858；RPS1 14.271，RPS2 14.429，RPS4 18.045，RPS8 16.838，inline18.317ms | RPS1修好此macro，不能当完整生成成果 | `full_ffn_geometry.py` → 同名结果 |
| RPS2/AR-fusion、RPS1全模型复验 | skyABBA FP8ref30.518/51.312；combined30.443/48.739；reuse2 30.403/48.315；fused30.375/49.813。后续FP8ref30.551/51.046；RPS1 30.451/49.095 | 几何改进后完整模型仍回退，最终移除NV模型接入 | `fullmodel_mechanism.py`/`fullmodel_reuse1.py` → 同名结果 |
| graph扩展：闭包weight/显式weight/完整MoE | closed14.744→14.218；explicit14.773→14.286；MoE20.483→20.895ms | 显式weight binding单独不是收益消失原因；完整MoE边界反转 | `macro_envelope.py` → 同名结果 |
| combine/shared/live router分段与重排 | combine15.361→14.620；captured route+shared18.956→17.768；live20.429→20.891；reorder20.451→20.869ms | 定位到live-router完整MoE图，重排未解；底层调度/locality原因未确认 | **`macro_composition_v2.py`** → `macro_composition.json` |
| captured FP32 router覆盖、route重算 | fused router未覆盖FP32，改用原general select；41/42层ID精确，第6层一expert281 vs89，配对率不变 | 是实现覆盖缺口；不能把该route变化称只是排序。两candidate臂用同router；capture/live不是全模型等价证明 | `route_compare.json`、composition v2源码 |
| explicit contiguous route | normal native/reuse20.557/20.968；contiguous23.012/21.014ms | candidate没改善；额外materialization不是修复 | `macro_contiguous.py` → 同名结果 |
| stride-aware ids消除该copy | native/reuse/strided20.587/21.071/21.352ms，finite/exact | 直接读strides仍回退，copies不足以解释主因；不采用。fullmodel strided未运行 | `macro_strided.py` → 同名结果 |
| gate-only/down-only hybrid | native20.516；allreuse20.940；gate-only23.316；down-only24.045ms | 此分阶段方案回退；没有fullmodel hybrid结果 | `macro_hybrid.py` → 同名结果 |
| observer bypass、eval group4/8 | normal30.531/51.321；bypass30.603/51.330；g4 30.095/50.666；g8 29.473/49.294 | observer不是所测critical path重大成本；bypass只诊断，不忽略production invalidation | `observation_ab.py` → 同名结果 |
| FP8 lossless half展开 | +1.3125GiB、.0269s预展开，局部约4–5%改善；全模型separate30.613/48.799；packed30.526/51.403；half30.449/50.818 | half慢于packed，当前布局不采用；没否定其他checkpoint组织 | `predecode_ab.py`/`fullmodel_half_v2.py` → 同名结果 |
| 固定K/N specialization | 热层仍有gain；全模型FP8ref30.531/50.936；current30.425/48.708；static30.380/48.793 | 固定维度没解全模型回退；不能隐藏增长维度编译 | `ffn_static_width.py`/`fullmodel_static.py` → 同名结果 |
| 精确FP4 register table/IEEE bit decode | 层3 count4 original RPS1 .2700→table.3072ms；bit variant.2708→.3373 | 此算术替代更慢，不采用；不代表所有解码改写被排除 | `decode_ab.py`/`decode_bits_ab.py` → 同名结果 |
| gate/up+SwiGLU再融合reuse | 层3count4 native.5786/fused1 .5499；fusion1 .6578/fusion2 .5725/fusion4 .5649ms | 不稳定优于简单fused；fullmodel fused仍慢于FP8ref | `fused_reuse_ab.py` → 同名结果 |
| compact/mapped/persistent worklist、uniform pair/single双kernel+where | 层26 native/reuse4/compact32/16/8 .5141/.4740/.4905/.5129/.5417ms；uniform.5187 vsreuse.4774 | hot层不胜简单reuse；旧capture worklist<.2%不证明所有producer成本无关 | `compact_work_ab.py`/`uniform_pairs_ab.py` → 同名结果 |
| gate/up/down独立projection chain | 层3gate.2881→reuse1 .2793；up.2813→.2716；down.2852→reuse2 .2722ms | 单投影小收益存在，轮转/完整集成未复现，不作为adopted全模型加速 | `projection_chain.py` → 同名结果 |

## GPU/CPU证据与最终HTTP资格

最后M5 separate serial capture：native/candidate均zero spill，临时registers62 vs64，
静态instructions739 vs1363/1328。native的consistent-state为no，采集不是ABBA，
只能提出调度/寄存器/解码假设，不能证明利用率或因果。旧M4 spill已修复，不能沿用
为最终M5回退原因。CPU profiler将C扩展等待折到Python self，归属不是CPU独占时间。
同步阶段verify约58ms/draft4.3ms只用于选调查对象；同步破坏重叠，子项不能重复相加。

最终HTTP：AR/DF均值29.5463/51.1319tok/s，DF每提示48.83–55.57。六对ID一致；
prefix32/514、chunk256、logprobs、seeded sampling、stop、双并发、abort后复用通过。
旧HTTP baseline DF46.975/53.413/46.644是顺序服务实验，未形成平衡最终HTTP A/B。
采用FP8的因果依据主要来自同驻留模型A/B，不能把跨轮HTTP漂移宣称为收益。

启动至health AR42.35/DF41.35s；首自然提示TTFT1.286/1.323s，later约.203s。
预热仅覆盖部分已知形状；first setup/JIT、context/cache容量增长、未见layout待测。
真实Metal最终测试：FP8 2方法、NVFP4 3、warmup1方法/4子例、GLM65、DF serving9、
DFlash2数值检查、pool9通过。namespace117通过1失败：未改common.py导入缺失Triton。
最终九候选文件Ruff check/format通过；registry无新增lint错误。原始日志留存。
未改checkpoint数值、量化或放宽断言；功能qualification不等于广泛生成质量评测。

## 执行版本、patch与复现前置条件

1. composition v1在fused FP32 router返回None时、计时前失败；有效版本是
   `macro_composition_v2.py`。half v1的cloned module id lookup不可靠、未运行；有效
   版本是`fullmodel_half_v2.py`的immutable module属性。fullmodel strided/hybrid/
   reordered/contiguous没有完整结果，不能计作验证。失败/未跑版本不作为复现入口。
2. 任务基准是remote HEAD `303af39b966c48e18d2268bdd610405e1316dc74`加已有FP4集成，
   不是该HEAD的干净checkout。model/runner基准SHA分别
   `2c2f0c3afb0fd7e53988aaaed82922e2d54bdadc9427680aeb4f148a1ee35691`、
   `bc4966b0aa9fc2822a1531e8e6e683ad7e55d19b3100d3b5bf990ef6e9b0f55d`。
   本次`baseline/`按哈希恢复两个文件及未改`glm5_fp4.py`。
3. 最终candidate patch SHA
   `7f8e5dd14423cb828b06393f3f9adf8e5f0a733bc522d37f497dec498a63825d`；registry patch SHA
   `958e35f0c3567b24bf19534fc36cd6e8eb4065b1683b93fb478d6c6d9033c341`。
   远端隔离Git根应用14文件精确，live未改。最终model SHA
   `7bde9e42538edb61545889f691c842b959189c46b09275f4116e110293d2f2f1`。
   本地取得的source副本不包含三个远端registry init全文；远端已有独立应用记录。
4. 初始fullmodel四臂使用尚含NV接入的历史模型。任务基准上先应用M4 warmup cumulative
   patch，再shared FP8 integration patch，然后`fix_scale_layout.sh`修复serialized
   scale接受方式。结果记录的历史source SHA优先。最终patch用于选定部署，不能替代
   历史candidate patch直接重跑四臂，否则NV和baseline可能同路而产生虚假结果。
5. 脚本按运行字节归档，含远端绝对路径及历史模块依赖。全模型需同target/draft revision、
   现有serving环境和硬件容量；不在M4加载完整权重。`actual_ffn_operands.npz`仅远端
   存在，`rootcause_replay.py`可再采集，但新route要作为新输入记录。脚本默认覆盖输出，
   重跑前保护原始结果。部分旧诊断未记录全部源SHA，本次清单不补造历史身份。
6. 本次在目标机仓库归档提交，不重新运行GPU/HTTP。`verify_archive.py`检查SHA、文档链接、JSON、
   脚本语法、最终candidate patch隔离应用；这是档案一致性检查，不是设备测试。
   未安装formatter，历史实验脚本未重新格式化。没有push、权重或大capture。

## 仍有依据继续做的工作

| 待查问题 | 现有边界证据 | 有判别力的下一实验与采用条件 |
| --- | --- | --- |
| live router完整MoE为什么抵消reuse | captured-route+shared快，live慢；RPS、重排、contiguous/strides无解 | 同图/全42bank依赖下定位router producer、submission/resource scheduling/locality；采用需全模型ABBA无回退且ID/接受/缓存合格 |
| 更多设置/编译能否前移 | eager/真实compiled-owner预热有效，但首TTFT仍高 | 记录first与增长边界graph/pipeline miss和setup分段；同时验证startup/first/steady，避免逐token specialization |
| 更合适的checkpoint组织、窄matrix与MLX replay | half单独失败，ICB仅synthetic，selector不利少行 | 保持码值/scales/方程，减少重复读取与无效MMA/dispatch；真实dtype/fullbank/live-MoE后全模型，不凭micro胜出采用 |
| 新旧checkpoint整体速度差原因 | 当前revision内部对照完整，旧revision同口径不完整 | 同机/同请求/精度/配置核对算子、route、工作集、验证循环与服务开销；未完成前不归咎硬件或权重数值 |

这些是未完成的根因和机制问题；当前不能保证它们填满40/90的全部差距。
