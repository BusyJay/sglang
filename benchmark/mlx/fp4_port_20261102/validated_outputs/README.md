# 已验证输出基线

当前 reference 由 CURRENT.json 指向，保存在 checkpoint revision / 基线 hash 子目录。基线不可覆盖；checkpoint、量化或计算语义改变后重新资格验证并生成新目录。性能优化只生成候选输出，与已经保存的 reference 比较，不再重新生成 AR。

覆盖：三种提示、32/2050 上下文、每组128个生成token；保存输入token、输出token/text/hash、模型revision/config/index hash、执行配置、数值验证证据、原始AR/MTP结果与源码快照。六组修复后原生MTP2与AR全部逐token一致。并不代表完整回答质量、HTTP接口、任意上下文/采样都已验证。

旧 warm cache 丢失 KV-B decoded_weight 的结果不能作为基线。当前版本恢复原始 decoded_weight；MTP冷加载与缓存加载所有参数逐位一致。对应回归测试在旧逻辑失败、修复后通过。

生成候选并直接比较冻结基线：

```sh
MLX_ENABLE_TF32=0 /Users/jay/workspace/prod/common/sglang-metal/bin/python \
  benchmark/mlx/fp4_port_20261102/compare_validated_outputs.py \
  --checkpoint /Users/jay/workspace/prod/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
  --output benchmark/mlx/fp4_port_20261102/candidate_comparison.json
```

如果候选输出已存在，增加 `--compare-result <json>` 即可离线比较，不加载权重。`--validate-only` 检查模型身份和输出hash。比较器在不匹配时返回非零。`first_difference` 是零起始token位置。

prepared weight cache默认开启：~/.cache/sglang/metal/nvidia--GLM-5.3-Flash-FP4-Local-b2b067c25a61997431c538e8。身份由不可变HF revision、config/index、shard header/size/mtime、layout版本生成，避免每次扫描165GiB。它不是对任意同mtime/size payload修改的完整内容hash。缓存保存CSF展开后的scale、expertbank、QKVbank，以及原始dtype的KV-B解码权重、embedding/norm/head。文件/manifest带generation防止撕裂读取；损坏条目回到正常加载再生成。vision还未纳入prepared cache。

M5 Ultra 256GiB、MLX0.32.4、相同checkpoint/config实测全语言模型+MTP46层及auxiliary缓存全部命中：24.173秒、173.1997GiB权重。仍有读文件/MLX数组建立成本。单轮交替顺序128token数据在raw结果里，不作为稳定吞吐承诺。
