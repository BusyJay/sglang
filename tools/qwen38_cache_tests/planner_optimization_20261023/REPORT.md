# 分层 LFU planner 固定开销优化

2026-10-23。修改前为 `ff22937a50`，本轮保留既有 LFU 候选规则，优化同一份实现的执行开销。

**结果：固定 route 下，有 miss 的 planner kernel 延迟降低约 32%–41%；关闭 MTP 的短请求中位数 76.079→78.366 tok/s（+3.01%）。MTP 两轮短请求增益分别 +6.61% 和 +0.37%，不能只选择最好的一轮，也不能将全部收益归为 CPU 共算。**

## 实现

- 用 lookup 在同一趟循环中读取 resident slot 频次或 nonresident 历史，完成衰减和原始 route union 加 1，去掉单独的 resident seed 循环与一次 block barrier。
- 在并行更新热度时汇总 resident 数量和 eligible victim 最低热度，用四个 warp 的归约替代 thread 0 的逐 slot 扫描。
- 复用已经去重、按 expert ID 排序的 active list，避免在 CPU 候选选择中再次做 route 去重。
- 仅对冷 miss 计数；零 CPU、容量足够、候选超过 n 等分支直接结束，不再排序全部 miss。只有空槽会截断冷候选集合时，才按热度/ID 排序冷候选。

选择数量的等价关系：若 miss 数为 M、free slots 为 F、严格低于 victim 热度的 miss 数为 L，则原升序列表中的冷候选前缀长度为 `min(L, max(0, M-F))`。它超过 n 时仍全部走 GPU；不超过 n 时仍选同一组最低热度专家。空槽保留给最热的 miss，热度相同按原 expert ID 顺序决策。CPU 热度累积、晋升和 safe-mask 边界保持原语义。

改动位于 [planner](/home/user/workspace/sglang/python/sglang/kernels/jit/csrc/moe/qwen38_expert_cache.cu:618) 与 [选择函数](/home/user/workspace/sglang/python/sglang/kernels/jit/csrc/moe/qwen38_layered_policy.cuh:31)。没有添加配置项；原 `cpu_hybrid_mode=layered_lfu` 使用该优化。

## 状态与正确性验证

- 80 组单步固定场景，覆盖 slots=16/64/146/256、tokens=1/3、n=0/1、全 hit、CPU miss、热 miss、超过候选上限和部分空槽。修改前后的 defined planner 输出与状态 SHA256 全部相同。
- 12 组各 64 步连续随机 route 回放，覆盖 slots=16/146/256 和 n=0/1/3/30；每一步的状态都纳入累计摘要。热度、GPU keys/lookup/frequency/ages/generation、clock、counts、CPU mask、GPU weights、descriptor、统计和有效 plan 前缀全部逐字节一致。包含 CPU 升温、晋升、后续驻留与重复路由；每组累计 CPU 专家数也相同。
- 新增 CUDA 边界覆盖：512 专家中跨 warp 的 eligible 最低热度、129 槽的部分空位，以及空槽保留时相同热度候选不受 route 顺序影响。
- 相关 GPU/cache/loader/dispatch/resize/close/CPU build 回归：**157 passed，362 subtests passed**。完整用例列表见 test-files.json；原始 tests-final.log 留在本地。
- Python 文件语法检查、git diff --check 与本机离线缓存的 ruff 0.15.1 检查/格式化通过。

## 固定状态 kernel 测量

以下为 slots=146、tokens=3、n=1，每类 31 次 kernel 的 GPU 时间中位数。单独的 CUDA profiler 只采集小型 planner graph；graph 中包含状态 reset，表格仅提取 decode_plan_kernel，因此不把 reset 算入 planner 时间。graph 整体计时与全部样本见 before/after-planner.json。

| 场景 | 修改前 µs | 修改后 µs | 延迟变化 |
|---|---:|---:|---:|
| 全 hit | 12.672 | 12.864 | +1.52% |
| 1 个冷 miss，留 CPU | 39.424 | 26.720 | -32.22% |
| 热 miss，搬 GPU | 30.944 | 18.304 | -40.85% |
| 冷候选超过上限 | 34.592 | 20.896 | -39.59% |
| 部分空槽 | 34.464 | 20.864 | -39.46% |

全 hit 的 0.192 µs 差异也保留，不宣称每个分支都更快。kernel 时间的减少是固定输入、同状态下的独立证据；不能直接把 40% 换算为整个模型 40%。

## 真实模型前后对照

RTX 5090 D + Ryzen 9950X；离线相同 FP4 pack、相同 runtime 环境和 cache 配置、物理核 2–5。各服务串行，未锁 GPU/CPU 时钟。短组是预热后 3 个 1088 输入/512 输出请求取中位数；自然聊天另报。完整组额外执行 170064 输入/8 输出和长输入后的短请求。

| 对照组 | 修改前短 tok/s | 修改后短 tok/s | 变化 | 自然聊天变化 |
|---|---:|---:|---:|---:|
| LFU n=1，MTP 完整组 | 99.112 | 105.662 | +6.61% | +3.28% |
| LFU n=1，MTP 复测 | 101.452 | 101.831 | +0.37% | -2.95% |
| LFU n=0，无 CPU worker | 101.066 | 107.070 | +5.94% | +1.78% |
| 原分层 GPU-only | 103.700 | 102.984 | -0.69% | +0.88% |
| LFU n=1，关闭 MTP | 76.079 | 78.366 | +3.01% | +0.42% |

完整组：170k TTFT 30.077→29.994 s；长输入后短 decode 90.312→99.311 tok/s。均完成短→长→短的 cache/KV 生命周期。

MTP 组存在接受长度和生成文本变化，同一 revision 复跑也未完全复现文本（见 analysis.json 的 same_revision_text_equal），所以不能将 token/s 差异全部归为 planner。关闭 MTP 的对照去掉 NEXTN 所有参数，短组 +3.01%，自然聊天约 +0.42%；这是额外对照，不是与 MTP 组横向比较精度、显存或绝对速度。未执行新的全模型 accuracy 基准。

原分层 GPU-only 单轮短组 -0.69%，自然聊天为正，且伴随 MTP 波动；这不构成严格全场景无回退证明。全局 hybrid 的实现本轮没有改动，也没有重新执行其 E2E 性能验收，不能把这张分层表当作全局达标证据。

## 结论与后续限制

保留这项改动，因为固定 route 的 miss planner 有明确局部收益，连续状态完全一致，无 MTP 短组也获得约 3% 改善。它降低的是 LFU 的固定成本，n=0 同样受益；本轮没有证明 CPU 共算相对 n=0 的稳定独立净收益。

后续更大的机会仍是执行位置与长期驻留分离，以及 resident GPU 工作、miss 搬运和 CPU 计算的流水线。当前 C>n 全 GPU 的语义被保留；没有通过放宽该规则、降低精度或跳过专家来获得这轮数字。

复现与原始证据范围见 [EVIDENCE.md](EVIDENCE.md)。诊断服务仅使用 GPU 0；正式服务未由本轮启动或部署，GPU 1 服务不受管理。
