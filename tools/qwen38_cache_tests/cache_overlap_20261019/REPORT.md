# Qwen3.8 cache overlap 目标 GPU 实验

基准：`d3eb24c40031999b3e62e4e9351c5b6118dfaa24`；输入包：`qwen38-cache-overlap-20261019.tar.gz`，包内 SHA256 校验全部通过。实验设备 NVIDIA RTX 5090 D（SM120），本地镜像 `swallow-sglang:nightly-dev-20261018-b234b5c3`。没有下载模型。

## 集成结论

三个 loader 开关已集成，均保持默认 `false`：`prefill_direct_cache`、`preserve_cache_on_resize`、`decode_early_cache`。可以在 threaded runtime 显式配置；prefill direct 还要求 `prefill_reuse_cache=true`。没有更改生产启动配置。

本次补充修正：

1. 原补丁用已初始化的有效专家 ID 作为零权重占位，导致同一 token 内出现重复专家 ID。GPU 上 prefill 数值错误可在 30 次热调用后稳定复现；同一组输入中有元素偏差 0.020263671875，超出原阈值。改为两种后端均支持的 `-1` 无效路由后，同样测试通过。CPU 断言覆盖无效路由哨兵、有效专家读取、未归一化权重和复用输出缓冲。
2. 原新增 resize GPU 测试的模拟 staged pool 缺少真实 VMM 的清理接口，导致断言完成后 cleanup 失败。测试清理时先移除模拟对象，再关闭 runtime。生产资源接口保持原样。
3. 数值断言补充 case/开关/graph 上下文，便于复现；代码通过 isort 和 ruff 格式检查。

## 数值与状态验证

- 348 项 CPU 回归通过，6 项新增 CPU 边界测试通过。
- GPU 合成 resize 测试通过，覆盖借用 arena 后恢复、层偏移变化、热点保留和淘汰。
- 修正后的 8 组 B12X A/B 全部通过：两项计算开关的四种组合 × eager/breakable replay；独立 BF16 参考校验 W4A16，resident CUTLASS 参考校验 W4A4。最大绝对误差 0.000244140625，保持包内 `rtol=0.02, atol=0.02`。
- Triton、W4A4/W4A16 prefill、共享分支与 breakable replay 的合成入口通过，见 `padding-parity.json`。
- 两层真实 FP4-Local 专家权重、真实 VMM backing，5 轮 shrink/grow 全部通过。预算 88 → 40 → 88 slots，VMM backing 从 247463936 降至 113246208 字节；确认热点仍驻留并逐字节检查保留的六种权重/scale 数组。每轮迁移、CPU 字节校验及扩容合计约 24–28 ms，不能把它当纯迁移耗时。该实验未调用完整 KV allocator。

日志：`final-regression.log`、`resize-fixed.log`、`vmm-resize.log`、`ab-b12x.json`。原始数值失败和修正前 A/B 留在 `parity.log`、`parity-repeat.log`、`ab-b12x-before-fix.json`。

## 初版性能结果（后续优化见文末）

下表来自固定合成权重和路由、单层、30 次热重复调用的 CUDA event 中位数；hidden=2560、intermediate=640、32 experts、16 cache slots、top-k=10。decode 使用 B12X，prefill 使用 CUTLASS W4A4。事件计时包含主机提交造成的空隙，不是纯 GPU kernel 时间。表中采用 eager 结果；graph 测例验证 replay 数值，但其重复计时仍调用 eager 入口。

| 场景 | 关闭 | 开启对应计算开关 | 变化 |
| --- | ---: | ---: | ---: |
| 单 token 热 decode | 0.336 ms | 0.628 ms | 约慢 87% |
| 3-token 热 prefill arena | 0.319 ms | 0.620 ms | 约慢 94% |

这组场景中，两次 dispatch、路由处理和合并开销超过了重叠收益；decode 提前部分还使用 baseline Triton。热重复计时没有持续制造新缺失，不能据此判断大批量或高 miss 场景的收益，也不能换算完整模型吞吐。两个计算开关不建议默认开启。热点搬迁验证通过，但是否值得启用仍取决于阶段切换频率与后续命中收益。

复现 A/B：使用本地镜像、挂载仓库至 `/workspace/sglang`，结果目录至 `/results`，设置 `PYTHONPATH=/workspace/sglang/python`、`SGLANG_QWEN38_CACHE_W4A16=1`、`SGLANG_QWEN38_CACHE_W4A16_BACKEND=b12x`，运行本目录 `ab.py`。

## 检查范围

`git diff --check`、修改 Python 文件的 isort/ruff 检查通过。全仓 CI 注册审计仍报告基准已有的 `test/registered/unit/models/test_qwen38_preallocated_gdn.py` 缺少注册；该文件未修改。新增 CPU 测试有注册、CustomTestCase 和 unittest.main 入口。

## 完整模型 canary

FP4-Local 本地 checkpoint、MTP=2、B12X、breakable graph、6388 基础 slots、KV stages 80000/160000/220032、mem_fraction_static=0.90。三个新开关全部显式开启，prefill chunk=2048。

完整模型 canary 通过：1024、8192、74000、170000 token 请求均完成，各生成 8 tokens；随后算术语义探针返回 `42`。日志确认实际 KV 归还阶段分别到达 160000/6643 experts 和 220032/6388 experts。请求耗时依次约 1.45、2.76、22.79、33.63 秒；这些连续请求复用 prefix cache，不是独立吞吐对照，也不证明通用模型准确率。报告和日志见 `model/report.json`、`model/server.log`，复现入口 `run_model.sh`。

MTP=2 的 target 主路径为 3-token verify，不触发单 token decode 提前分支；后者由前述合成 GPU 测试覆盖。本轮真实模型覆盖阶段缩容，扩容/尾部重借由前述合成和两层真实 VMM 实验覆盖。

8192-token chunk 的首次长上下文实验发生 CUTLASS workspace OOM，日志保存在 `model-chunk8192-oom/`；初始离线 checkpoint 路径问题的日志保存在 `model-cache-path-failed/`。关闭三个新开关的相同 8192 chunk 对照已通过 1K、8K、74K 请求及算术探针（`model-baseline/report.json`）。基础 slots、MTP、KV stages、mem_fraction_static 相同，两轮 expert stage 预算均为 [7006,6643,6388]。失败发生在 80K stage 切换之前，MTP target 也没有使用单 token 提前分支，因此这组证据将显存回退定位到 prefill direct 拆分执行。进一步 allocator history 已确定：cache/arena 不同大小的 CUTLASS workspace 与 HC/GDN 激活交错分配，导致大块被活跃张量切碎；不是简单的 workspace 泄漏。保持原来的 8192 chunk、6388 slots、KV 容量，使用 `PYTORCH_ALLOC_CONF=max_split_size_mb:512` 后，1K/8K/74K/170K 请求和语义探针通过，另一次 allocator 快照对照也通过。详见 [变慢与 OOM 排查](investigation/INVESTIGATION.md)。2048 chunk 的结果保留为此前的隔离验证，不构成对生产配置的修改。

因此，代码作为默认关闭的实验功能集成；两个计算开关仍不建议默认开启；若在本机同类 8192 chunk 配置中实验启用 prefill direct，应使用上述已验证的 allocator 设置。`decode_early_cache` 在本次单 token B12X 热缓存实验中也有明显性能回退。`preserve_cache_on_resize` 通过真实阶段缩容及 VMM 数据验证，但仍需根据实际阶段频率评估收益。

## 后续路由与峰值优化

最新改动、复现实验和性能限制见 [优化复验](optimization/REPORT.md)。该版本消除了 prefill 路由元数据 H2D，融合 decode 命中快照，并限制普通输入 CUTLASS MoE 调用的 token 数。外层 8192 chunk 的完整模型已在未设置 allocator 参数时通过 170K 请求；上述初版性能和要求使用 allocator 设置的结论属于历史结果。

## 动态路径选择

再一轮把“是否拆分/是否提前计算”交给策略决定：新增 `cache_overlap_policy`（`auto` 默认 / `always`）。decode 在 B12X 与 Triton 上都没有可靠的盈利区间，`auto` 直接绕过提前分支；prefill 按缺失专家数、批量与 arena 状态选路，只有大批量且确有主机搬运时才拆分。完整模型 canary 在三个开关同时显式开启、外层 chunk=8192、未设置 allocator 参数的情况下通过 1K/8K/74K/170K 请求，运行时计数确认拆分分支被真实模型选中。方法与逐档数据、限制见 [动态路径选择复验](adaptive/REPORT.md)。

## 当前默认值

前文的“三个开关均默认 false”是各轮实验当时的状态。基于以上实测，threaded runtime 下的默认值现在是：`prefill_direct_cache=true`（前提 `prefill_reuse_cache`，同为默认；同时带来 CUTLASS 2048-token workspace 上界）、`cache_overlap_policy="auto"`、`decode_early_cache=false`、`preserve_cache_on_resize=false`。理由：prefill 在 auto 选中的档位有可复现净收益且长请求依赖该开关；decode 没有盈利区间；resize 保留的代价是一次阶段切换停顿，收益取决于阶段频率，留给部署方决定。
