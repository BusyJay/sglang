# GLM MLX revision v3 patch — 2026-10-31

基准 HEAD：`202d800b066829dc10ca5df40f83d40303c822ca`（metal）。
包含累计 1–4 优化与本轮验收修复，保留 checkpoint 权重、量化与 dtype promotion。
这是生产机器候选包；M5 同 checkpoint 完整模型质量/HTTP/吞吐尚未验证。

## 校验与应用

解压后进入包目录，用现有 Python 运行：

```sh
python3 apply.py --repo /path/to/sglang
python3 apply.py --repo /path/to/sglang --apply
```

第一条只检查。脚本校验 HEAD 与每个文件 SHA256，接受原基准、上一轮验收输入或
已经应用的最终源码；遇到其他文件改动拒绝覆盖。应用前将原文件备份到目标仓库
`benchmark/mlx/verification_revision_20261031/patch_backup`。不安装依赖，不修改
系统资源限制，也不提交或推送。全部文件指纹在 `manifest.json`。

也可对干净基准使用 `git apply --check changes.patch && git apply changes.patch`；
已应用上一轮包且指纹匹配时使用 `revision.patch`。累计与增量 patch 都已在隔离
目录应用并逐文件校验，见 `apply_checks.json`。

```sh
python3 apply.py --repo /path/to/sglang --rollback
```

回滚仅覆盖仍匹配本包最终哈希的文件，恢复备份的输入版本。

## 复测

将 `evidence/` 中脚本复制到目标仓库
`benchmark/mlx/verification_revision_20261031/`，在仓库根目录使用现有 MLX 环境：

```sh
/path/to/python benchmark/mlx/verification_revision_20261031/run_device_tests.py
/path/to/python benchmark/mlx/verification_revision_20261031/bench_revision.py
/path/to/python benchmark/mlx/verification_revision_20261031/bench_dense_mla.py
/path/to/python benchmark/mlx/verification_revision_20261031/bench_attention.py
```

runner 执行真实生产 MLX 代码及注册测试断言，AST 绕过依赖 Torch 的包/CI 初始化。
这不是完整 SGLang CI 或 HTTP 验证。正常 CI 应运行包中 registered 测试。

生产验收必须固定权重 revision/文件哈希、量化、请求/token IDs、MTP配置与资源
默认值；交替 baseline/candidate 顺序，保存逐样本耗时、峰值内存、logits/KL/
top1、路由、草稿接受率。覆盖 ordinary/0步/混合请求、接受/拒绝、chunked prefill、
prefix reuse与释放。baseline 的 MTP serial 对照可用现有 verification mode。
本包未新增关闭开关，covered kernel默认参与执行；不要将随机单层数值视为
checkpoint质量验收，也不要将本机时间比推算M5最终tok/s。

## 范围与未完成项

- 修复 affine 模块缺失 global_scale；mixed投影共享原packed权重与metadata。
- 两个 latent 变换保持 singleton GEMV/GEMVT，与实际promotion后的布局对齐。
- D512/H64 ragged直接读取合法索引，TG共享KV tile，保留原QK/softmax/PV rounding；
  覆盖长度8..4096和空集合。其他形状保留原通用批量attention，不声称已优化。
- KDA共享投影、递推所有状态快照及输出投影保留；dense/shared与routed gate/up
  激活融合默认参与覆盖路径。routed没有实现expert→token compact跨token权重复用。
- MTP仅合并验证执行边界；0步保持lazy。跨请求逐层权重复用和完整MTP overlap未完成。
- draft top1 epilogue保留，不宣称消除vocabulary矩阵读取或实质吞吐收益。
- v3纠正一次gather生产主基线，保留v2向量读取，新增原dtype TG暂存及QK slot循环外复用；完整正反分块样本和小收益/持平结论见REVIEW。已通过第三轮明确界定的M4独立验收范围。
- 无权重、虚拟环境、庞大capture；未安装第三方库。ruff/格式器不可用，AST与
  diff检查已执行，不冒充格式工具或完整模型验证。

从v1/v2升级：按`v1_manifest.json`或`v2_manifest.json`核验已有源码，然后分别使用
`v1_to_v3.patch`或`v2_to_v3.patch`，先运行`git apply --check`再应用。
`revision.patch`仍针对首次验收的11文件输入，apply.py遇到其他输入指纹会拒绝。

稳态微基准还需复制`steady_panel.py`；运行前设置`MLX_ENABLE_TF32=0`。
额外运行`fair_attention_v3_steady.py`、`pool_bf16_v3_confirm.py`、`extended.py`、
`mtp_extended.py`，覆盖公平性能、44组边界与扩展缓存轨迹。阶段对照需要`ragged_v2.py`
和`stages_v3_paired.py`。证据脚本在仓库根目录执行，结果保存在benchmark目录。

最终交付包：生产源码与已验收v3完全一致。两份基准脚本仅删除未使用的v1目录加载，见delivery_script_changes.patch；独立验收报告及原始证据在acceptance/。该报告验收的原v3包哈希与最终交付归档哈希不同，源码哈希一致。M5仍待目标机验收。
