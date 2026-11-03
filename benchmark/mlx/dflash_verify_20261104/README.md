# M5 DFlash 长候选 verify 成本调查

2026-11-04，M5 Ultra256GiB、MLX0.32.4。同驻留 target revision
76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 和 draft
bf582e4eacc1810f76656d1811693ff6c6737d2a。草稿训练 block8，最多7 proposals。
实际单请求 serving 从 verify_mtp_requests 转到 verify_mtp_rows，已有固定KDAgroup/MLAlayer。

## 已完成证据

原本36×300-token矩阵开得过大，包含明显诊断异常，延迟了反馈。用户已要求
探索改用快速短实验；后续使用单层24次、最多12步/遇到>.5秒即停的诊断。
只有发现明确候选收益后才做少量300-token最终验证。没有将本机M4结果外推。

| proposals | 对照热生成 tok/s | FP8调用合并 tok/s | 平均产出/verify | correct/proposed |
|---|---:|---:|---:|---:|
|3|52.9804|53.1699|3.2066|.7429|
|5|51.1432|51.1311|4.0556|.6200|
|7|48.5700|48.5750|4.5600|.5164|

3proposals使用第二轮3prompts对照：第一轮前两个请求分别1.431/2.287tok/s，
多次verify约2.3秒，持续到第二请求约230token后消失；第三请求及后续正常。
不能删除这些样本再声称优化带来巨大收益。完整raw保留在fullmodel.json。
所有36次token序列精确，深度接受和计数一致；79个真实FP8投影批处理输出逐位一致。
batch_fp8只是诊断在_project_rows内以projection(x)替代逐行调用，没有生产采用，
热吞吐无可靠收益。系统swap1557.38→1560.88MiB，没有进程级归属数据。
峰值约172.55GiB；没有并行加载完整模型、安装依赖或修改系统wired参数。

## 随候选增长的成本

同步诊断破坏重叠，不能与真实吞吐相加。相同短prefix重复token的verify，
取replay1..3均值（仍有第2次形状稳定开销），3/5/7proposals：
verify88.86/112.72/128.37ms，KDAgroup54.48/72.77/83.85ms，
MLAlayer27.91/31.31/35.28ms。KDAgroup包含attention+mHC+FFN，并非纯KDArecurrence。

单个真实layer26，控制输入由归档FFN操作数repeat构造，不是实际serving残差；
单层weightcache加载，峰值4.096GiB。24次、交替phase顺序的mean ms：

| phase | 4rows | 8rows |
|---|---:|---:|
|whole layer|1.7163|2.1968|
|attention|.4706|.6811|
|FFN+mHC|1.1976|2.0646|
|inputprojection|.3619|.5026|
|outputprojection|.2021|.2439|
|recurrence+snapshots|.1838|.1921|
|recurrence无snapshots诊断|.1715|.1828|

独立phase时调度不同，不能将子项相加作为完整layer成本；snapshot消融最后state和
output精确。snapshot额外写约.01ms，并非此输入长chain主要增长项；无snapshot只是
定位，不能用于真实reject/消费cache。

## 继续追因的短实验

sort_boundary.py：8rows×8picks触发mlx-lm SwitchGLU >=64排序。直接执行保留
pick/output/reduction顺序，各phase 9检查精确、24次交替：8rowswhole2.0755→2.0753，
FFN1.9039→1.9205、MoE.9602→.9540；没有解释大幅增长，未采用。

hc_batch.py：mHC逐token residual matmul改成原生batched matmul，9检查精确。
4rowswhole1.584→1.500，6rows1.836→1.826，8rows2.061→2.050；
FFN8rows1.8988→1.8919、MoE.9528→.9475。局部scope、样本少，尚非生产候选结论。
该实验使用诊断函数绑定，在编译前分别热两arm；需要独立owner复验数值、真实残差、
完整模型短比较才能采用，不把<1%漂移当稳定收益。

recompile.py：同一单层_FixedLayerGroup 12次同shape，不断改变输入：前2次observation
rebuild，之后False、graphs1，约1.89–3.52ms。没有复现原先每verify2秒。
完整模型短 cProfile/行追踪定位到 verify_mtp_rows 的 async_eval；这些工具影响 CPU 提交节奏，不能作为普通执行耗时证据。

## 失败与验证边界

初版操作数capture取到编译追踪占位符，eval报错；改成fixedgroups关闭的实际操作数
采集，随后恢复编译进行吞吐。AST不能替代这79项真实device结果。
此轮尚无采用的生产优化，不宣称达到90tok/s。服务恢复与health检查见下方收尾记录。
raw、源码SHA、脚本及用于复现路由的少量真实residual NPZ随远端证据归档；
不包含权重、虚拟环境或庞大GPU capture。

## 短实验继续追因（本轮后续）

所有新增完整模型吞吐 A/B 都没有 profiler、行追踪或阶段同步。
直接调用同一个 DFlash serving engine，每次读取结果 token；这是模型/engine
吞吐实验，不是完整 HTTP 吞吐。四臂顺序 reference/candidate/candidate/reference。
原脚本漏了普通 runner 的进程级 residency 设置；后续完整模型统一在 target
加载后调用 mx.set_wired_limit(max_recommended_working_set_size)，与 runner 相同。
这不改变系统 wired-memory 参数。补齐后没有再出现秒级停顿；用户要求先放下
该停顿。原生 sample 显示 async_eval 内 condition wait、Metal 驱动提交线程等待；
尚未证明唯一触发条件。关闭 residency 后仍快的反例保留，不声称因果已完全解决。

独立 mHC 方法 globals/编译 owner 的全模型检查：1399 个 hidden、features、token
及全部 immutable cache arrays 精确。64-token ordinary engine A/B：
reference38.539/42.677，batched41.202/41.442 tok/s；去掉前16步仍随顺序漂移，
没有稳定收益。没有生产采用。

NVFP4 gate/up 原生拼接 bank：全模型方程不变，但单层输出出现约0.23–0.29
绝对差异，且8行whole2.0718→2.2038ms、MoE .9635→1.0614ms。
已有 gate/up 融合 shader 数值精确，whole2.1413ms、MoE1.0653ms，同样慢。
没有只因不逐位一致便弃用 bank；其吞吐已回退，数值触发的具体 native geometry
仍未完全定位，候选保留为诊断，没有采用。

新 down projection + routing weighted sum + shared combine 单次 dispatch：
保留原 packed codes、/16 scales、每 expert global、每vector qdot及ordered topk sum。
24单层检查精确；RPS1/2/4与TILE1/2、RPS8/16都没有 MoE 收益。
8行nativeMoE .9553ms，候选1.032–1.044ms；只把shared相加留在外面仍慢，
gate/up提交顺序恢复native仍慢。独立预先eval activation的down：
4行native .4340→.2724–.2872ms，8行 .3823→.3694–.3971ms。
因此不能以4行算子收益宣称8行/完整MoE收益；当前不同launch/producer图仍有成本，
不称Metal限制。初版错误地把grid当threadgroup数量，未完整写输出，出现伪收益及
数值失败；修正grid后重新测量，down_combine.json等对应修正后的完整输出。
首次未修正的控制台证据保留在tmux日志，不能作为正确性能结果。

新 NVFP4 多pick weight reuse 在构造8行输入中：R4/RPS2 whole2.0934→1.9893ms、
MoE .9699→.8647ms，24检查精确。全模型1399数组精确，但128-token吞吐
reference41.961/42.979、candidate41.833/41.634 tok/s，没有改善。
继续采集真实layer26 verify residual（正常draft、不是重复token）：
构造输入其实是4行重复两遍，21个专家、全部64pick成对；真实8行33个专家，
23个专家只出现1次，26个pick不能配对。R4工作组填充：构造10组4+12组2，
真实5组4+2组3+7组2+24组1。由此定位到构造输入过度复用的测量偏差。
真实 residual 单层原候选whole2.2291→2.2661、MoE1.0039→1.0603ms。

改进为显式 singleton qdot 快路径，减少一次访问专家的冗余多vector处理；
真实输入whole2.2026→2.1373ms、MoE1.0047→.9500ms，数值精确。
在正常生成只对8行使用该候选，小形状沿用原生既有计算（shape覆盖诊断，不是
最终默认关闭方案）；128-token reference41.780/42.943、candidate42.792/42.633，
前16步以后reference41.515/41.776、candidate41.626/41.430，仍无稳定增益。

继续采集全部42个真实MoE residual，在layer3/7/15/31/43复验R4/RPS2：
whole分别2.9509→2.9673、2.8702→2.8846、2.8700→2.8566、
2.8309→2.7976、2.8450→2.8227ms；第26层的3%收益不代表所有层。
这些MLA单层用1-token初始化cache，未重建完整目标prefix，所以只归因输入/FFN，
不是完整attention稳态复现。三个phase输出及cache checks精确。
R2/RPS2、R2/RPS4、R3/RPS2、R4/RPS4、R8/RPS2继续测真实layer26，前3项
whole约改善1%，R4/RPS4无益，R8寄存器/工作增多明显慢；没有以大reuse盲目扩大。
最终R2/RPS4的128-token结果：reference41.743/42.477、candidate42.402/42.298；
前16步以后reference41.479/41.293、candidate41.293/41.108，仍没有稳定收益。
日志沿用了reuse4标签，源码实际为probe_reuse=2、probe_rps=4，以源码为准。

所有完整模型短A/B保留token、raw逐步seconds、verify标记、accept计数和峰值。
仅在明确收益后才做300-token最终验证；当前没有这样的候选，不追加长矩阵。
候选未生产采用，现有量化、服务/缓存机制不变。剩余实现缺口是实际图中跨专家
稀疏reuse、singleton效率和activation producer连接的收益，不能据此声称没有空间。

## 收尾与普通服务检查

最终R4+singleton全模型ABBA聚合42.353→42.712 tok/s（+0.85%），
R2/RPS4为42.107→42.350（+0.58%），均小于首末对照漂移，不能宣称稳定优化。
全部1399数组、128-token生成序列及接受计数精确。没有采用生产改动。

原服务30007按原launch_dflash.sh恢复，health200。首次非stream32token请求
和其后的health读曾超时；随后8-token SSE请求在2.253秒完成，completion_tokens8、
finish_reason length、[DONE]，raw在restored_http_stream.log。此处只是服务恢复检查，
不是完整HTTP性能/质量/缓存资格验证，不把短请求吞吐作为稳态数据。
提交后检查确认服务PID78455仍运行，3秒时限的health读取返回200。
