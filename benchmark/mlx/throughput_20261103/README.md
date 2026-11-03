# 远端 GLM FP4 / DFlash2 短生成吞吐实验收尾

2026-11-03。按用户要求完成最后一组 MLA A/B 后停止；没有启动 worklist 或更长生成实验。
本轮没有采用生产优化，AR 40 tok/s、DFlash2 90 tok/s 的目标均未达到。
原 DFlash 服务已恢复，PID 4709，2026-11-03 11:07:10 +0800 的 `/health` 返回 200。
恢复只检查了启动和健康状态，没有追加 300 token HTTP 性能/功能验收。

## 实验口径与源码身份

- 远端 `jay@10.173.1.94`，目录 `/Users/jay/workspace/sglang`，分支 `metal`，HEAD `303af39b966c48e18d2268bdd610405e1316dc74`。
- M5 Ultra、256 GiB、`applegpu_g17d`；MLX 0.32.4、mlx-lm 0.32.2。解释器 `/Users/jay/workspace/prod/common/sglang-metal/bin/python`。
- NVIDIA GLM-5.3-Flash-FP4-Local revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`；原 BF16 DFlash2 revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`。完整路径在 `profile.json`。
- 原生 MTP 关闭；DFlash 三个草稿，服务 CLI block size 4；共享目标 embedding/head。
- 三个短提示：天空解释、Python Fibonacci、中文过拟合；模板后分别 20、28、27 个输入 token。所有吞吐生成均为 300 个输出 token。
- `tok_s = 299 / sum(raw_seconds)`，首输出 token 来自 prefill，单独保存 prefill 时间；不包含 HTTP/tokenizer 开销。每个请求独立缓存，保留相同模型方程和量化。ABBA 顺序见各 trial 脚本。
- 模型和草稿 active memory 171.3391 GiB，加载约 24.10 s；实验使用进程 memory/wired limit 200 GiB、allocator cache 1 GiB，eval groups decode/prefill/verify 为 2/4/2；未调整系统 wired-memory 设置。
- 环境 `PYTHONPATH=python SGLANG_USE_MLX=1 MLX_ENABLE_TF32=0 MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=1024`。
- `baseline_manifest.json` 与 `final_source_manifest.json` 中五个生产模型/kernel 文件 SHA256 全部相同。远端已有 `python/pyproject.toml` 修改与 `python/pyproject_other.toml` 删除完整保留；本机生产 tracked 文件未修改。没有提交或推送。

## 基准与候选结果

基准三个提示 AR 为 30.3350 / 30.4799 / 30.4107 tok/s，DFlash 为 48.0652 / 54.5966 / 47.9931 tok/s；均值 30.4086 / 50.2183。
三个提示的 AR 与 DFlash 输出 token ID 完全一致。原始逐 token 时间、prefill、内存和草稿计数在 `profile.json`。
下表均为同一候选实验内各模式的算术均值，不是跨实验测速差异。

| 机制 | 300 token 运行数 | AR 基准→候选 tok/s | DFlash 基准→候选 tok/s | 正确性与决定 |
| --- | ---: | ---: | ---: | --- |
| 34 个 KDA MXFP8 输出权重的精确 half-value 缓存 | 24 | 30.4297→30.5096 | 49.8746→50.2217 | token/草稿计数一致；不足 1% 且无稳定收益，增加 2.125 GiB，不采用 |
| 动态量化并入输出投影 | 24 | 30.4148→25.8788 | 50.2224→35.9245 | 真输入 BF16/FP32 count 1/4 和生成一致；性能回退，不采用 |
| NVFP4 同专家多 token 权重复用 | 12 | 未测 | 50.1979→45.1682 | 三个提示 token 与草稿计数都改变；性能回退，不采用 |
| 预先规划 getter 的图观测检查 | 24 | 30.5297→30.4752 | 50.2734→49.6867 | token/草稿计数一致；无收益，不采用 |
| 通过已有 fixed MLA layer 编译单 token MLA | 12 | 30.6154→30.0156 | 未测 | 45 层输出、缓存数组及 offset 精确；三个完整生成一致；约 2% 回退，不采用 |
| NVFP4 gate/up 合并投影 | 没有完整 ABBA | 未测 | 未测 | 真实 M5 输入有最大约 4.8e-7 FP32 差异，guard 提前终止；仅保留诊断证据 |

完整数字见 `result_summary.json` 和对应 `*_ab.json`。未采用指没有写入生产路径；诊断候选保留在实验目录。
不能从小浮点差异推出 Metal 不支持批量，也没有无依据放宽测试。
同专家候选已检查完整生成影响，确认接受计数和输出变化，但尚未定位全部 native FMA/reduction 差异。

## 已确认的成本和机制问题

DFlash 三个提示总 decode 时间/verify 次数分别为 63.477 / 64.430 / 63.572 ms；每次循环平均输出 3.051 / 3.518 / 3.051 token（含 anchor）。
验证次数 98 / 85 / 98，正确草稿 201 / 214 / 201，拒绝 45 / 21 / 51。
这说明应继续缩短目标模型验证循环；这些数值包含 draft、验证、消费和主机开销，不是 GPU kernel 独占时间。
此前归档的阶段计时约目标验证 58 ms、draft 4.3 ms；本轮没有重新做该阶段拆分。

真实 MoE 投影输入为 FP32，不能使用只有 BF16 的合成微基准判断完整模型收益。
投影诊断关闭 compiled groups 并插入同步，包含上游依赖并破坏重叠；不可把各项相加作为 GPU 归因。
真实输入 replay 也包含孤立提交成本，仍没有完整 GPU 时间线。

动态量化投影融合在每个输出 tile 重复量化输入，减少 dispatch 却增加计算；全模型回退与这个实现机制一致。
同专家复用每个输出 tile 扫描 picks 并使用动态索引累加器，增加串行工作和潜在寄存器压力；真实 count4 微基准约 0.26→0.33 ms。
精确 half-value 缓存保持原 E4M3 数值、UE8M0 scales 和 reduction，但增大工作集；所测 KDA 投影约 0.211→0.244 ms。
这些是已测候选的实现问题，不是平台能力限制。

cProfile：AR 300 输出为 10.020 s、约 1080 万调用；图观测 `current` 6889 次、own 0.793 s / cumulative 1.116 s。
DFlash 300 输出为 6.337 s、约 448 万调用；verify own 4.097 s / cumulative 5.580 s，图观测 own 0.373 s / cumulative 0.513 s。
profile 会增加开销，MLX 等待会记在 Python frame，不能解释成 CPU 活跃比例或 GPU 利用率。
热身后 AR/DFlash 各 300 输出没有新增模型图缓存条目；这个短上下文范围内没有逐 token 模型图重建证据。
不能外推到更长上下文、MLX 内部 pipeline 编译或其他缓存增长。
预规划 getter 的 1000 次检查仍约 48 ms，全模型无收益。

基准前后 swap used 都为 1295.50 MiB；收尾为 1633.44 MiB，服务健康时 1632.88 MiB。
不能宣称整个实验期间 swap 完全不变；未获得进程归因，不将增长自动归因于某候选。
整个实验按顺序只运行一份完整模型，没有安装第三方库、下载权重或更改 checkpoint。

## 证据与复现边界

`tested_remote/` 保存从目标机取回的实际候选/runner 源码，后续复现应以它为准。
目录顶层 `nv_exact_pair.py`、`nv_shared.py`、`observation_fast.py` 与实际测试快照不同，后续编辑未在 M5 重测；不作为已验证实现。
`nv_worklist.py`、`worklist_trial.py`、`quant_hoisted.py` 等为待验证草案，本轮结束后未执行。
本机 M4 Pro 的合成数值检查不代表 M5 全模型验收。

本轮完成了目标机完整模型生成 A/B、真实输入 replay、MLA 层缓存比较及源码哈希检查。
没有新增 registered 测试或进行完整服务缓存生命周期验收，因此也不把这些结果当作正式集成验收。
`restored_service.json` 保存恢复服务的 PID、实际命令、健康响应和内存信息。

如后续获得继续实验指令，先通过 `sglang-remote` pane `%0` 核验 HEAD、服务 PID 和内存；停止已确认的任务服务，避免第二份完整模型驻留。
先复制要复现的 `tested_remote/*.py` 到单独工作区实验目录（脚本内 root 和 runner 路径需保持一致），再使用上述解释器与环境运行对应 trial；保留完整日志、JSON、源码哈希和 ABBA 顺序。
复现脚本依赖 `short_bottleneck.py`，各候选完整命令已保留在本目录 `*_run.sh`；不要直接把顶层后续草案当作已测代码。
本轮收尾不再执行这些命令。服务保持原生产配置，未启用实验 monkeypatch。
