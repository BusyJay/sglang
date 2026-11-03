# 归档清单与证据缺口

本次提交位于目标M5的 `/Users/jay/workspace/sglang`，内容是当前FP4 checkpoint性能探索记录，不改变生产路径。
本地伴随提交 `4367e938a7` 先于用户澄清产生；它不能替代本次目标机提交。
主结论在 [EXPERIMENTS.md](EXPERIMENTS.md)，最终交付状态在 [README.md](README.md)。
[archive_manifest.json](archive_manifest.json) 列出选入的原始文件、大小与SHA256。
源副本、历史patch和实验脚本保留原字节；其历史README中的“未提交”“尚未远端验证”
等描述是当时快照，采用/移除决定已在主账本更新。

范围包括AR基础机制、FP4+DFlash短生成、第一轮300-token对照、M4 FP8/NVFP4根因、
AR融合与预热、最后M5的局部/全bank/live-MoE/全模型/HTTP验证。已有其他checkpoint的
NA/head优化在前面的Git提交中；不混入不相关的vision、dtype研究和旧交付全仓库副本。
没有权重、环境、GPU capture、传输片段或后台登录信息。

可运行一致性检查：

```sh
python3 benchmark/mlx/remote_validation_20261103/verify_archive.py
```

它只解析历史Python脚本与JSON，检查文件SHA、三个入口文档链接、全模型/HTTP汇总，
并在本证据目录内的临时隔离Git根应用最终candidate patch，核对11文件最终SHA。
没有执行历史实验、触碰生产源码、安装依赖或访问远端。registry完整baseline未取回，
它的14文件应用结果仍以归档的远端 [patch_apply_check.json](results/patch_apply_check.json)
为证。本地检查不能冒充重新完成Metal数值测试、生成质量或HTTP验收。

以下材料没有完整取回，明确保留缺口：

- AR历史scale cache基准的逐步`scale_cache_ab.json`及`scale_cache.py`未在本地找到；
  该项数字来源于历史AR README，不能称已封存此项完整raw结果。
- ICB的主要源码`icb_probe.mm`已保留，详细历史raw probe结果未取回；账本数字来自
  AR README，仍只限synthetic范围。
- M4 `actual_mlp_abba.json`有原始耗时，但独立驱动脚本未找到；`mlp_integration_check.py`
  是正确性/类集成脚本，不能冒充那个计时驱动。
- M5实际activation/route文件`actual_ffn_operands.npz`留在远端，没有复制进此次提交。
  重采集脚本已归档；新采集不等同于历史同一输入。
- 远端三个registry init完整源未取回；最终patch、hash和远端应用检查已保存。
- 部分历史脚本依赖目录`archive`、AST fixture或已有生产源码；只保留已取得的final
  source/frozen inputs。没有足够输入时应明确重建前置条件，不把脚本可解析当作可运行。

完整历史源码哈希应取自相应结果/manifest；新清单只标明现在封存的内容，不能补造
过去没有记录的源码身份。复现完整模型须保留相同checkpoint/quantization/config、
恢复当时候选接入，并在每轮输出新的结果文件；不要覆盖本归档原始JSON。
本次提交前未安装formatter，原始脚本未重新格式化，避免让其SHA与运行结果脱节。

提交前一致性检查通过：147份Python解析、38份shell语法、360个归档文件SHA、11份最终
源码与candidate patch重建、5组全模型汇总、6对HTTP ID记录。初始历史fullmodel模型
也经两份历史patch在隔离根重建，SHA与原始fullmodel结果一致。
结构化文档/源码/JSON的`git diff --cached --check`通过。原始patch的空白context行、
GPU/测试日志和profiler文本有27处空白/EOF提示，已逐项归类保留原字节，不改写历史
哈希或patch内容。此限制不影响patch应用检查；不声称整个原始档案零whitespace提示。
