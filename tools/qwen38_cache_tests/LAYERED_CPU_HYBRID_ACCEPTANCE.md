# 分层 CPU 共算验收（2026-10-22）

## 分层 LFU 准入与 CPU 热度更新

按用户确认的规则新增 `layered_lfu`：低于本层可淘汰 GPU victim 热度的 miss 才是 CPU 候选，候选数≤n 时留 CPU，超过 n 则全 GPU。CPU/GPU 共同更新专家热度，热 CPU 专家可正常迁入 GPU并保留频次。133 项测试、340 个子用例通过；覆盖 CPU→GPU、部分空槽、B12X 保护和缩小 slot 容量后的迁入，真实 n=1 完成短→170k→短。各场景性能仍有波动，未签署稳定提速。最新实现、完整对照与复现见 [LFU 路由报告](layered_lfu_20261022/REPORT.md)。

## 全局分流算法借鉴后的复验

已新增 schema 4 工作量分桶，跨层/容量复用成本决策，保留分层 LFU。128 项测试、328 个子用例通过。热权重表将实际分流扩大到 4701 次 CPU 调用/12616 个专家，但独立 A/B 仍回退；冷权重下复验 33 个桶、99 个代表、4455 对样本，全部拒绝 CPU 分流。因此未把热表作为启用依据，不能签署分层端到端提速。实现、完整时序配对、搬运统计边界和复现见 [分桶复验报告](layered_bucket_20261022/REPORT.md)。

## 目标机应用验收

本次已在 RTX 5090 D / Ryzen 9950X 上应用并验证，107 项 CPU 测试、20 项 GPU 测试通过。真实服务完成短→170k 长输入→短请求，并实际执行校准后的 CPU 分流；新增未校准静态分支回退及混合分支搬运统计回归。分层端到端性能目标尚未通过，不能把局部分支计时收益等同于整体性能提升。最新配置、校准表、全部对照和限制见 [本次验证报告](patch_validation_20261022/REPORT.md)。

## 打包时历史复验

打包时结论（第二轮复验）：上次三个实现问题已修复，本地复验通过；GPU 集成与完整性能验收仍未完成。审核 HEAD `9321a3f4504534811676c4d1f39d62341089e610` 上当前未提交实现；不修改生产代码。按用户指示，sync/device/async 兼容性不作为阻塞项。下方第一轮问题保留为历史记录，历史状态以本节为准。

## 第二轮复验结果

### 已关闭的问题

1. 校准绑定：`CpuNvfp4.library_path` 保存加载库路径，`execution_metadata()` 加入 CPU 库内容哈希、fill_cta、cache_overlap_policy、resident_alphas 和 parallel_miss_plan；源码指纹补入 mailbox/overlap/threaded 编排和 B12X FP32 实现。shared-fill overlap 作为实际调用键，经 Python→CUDA 传递，schema=3，表宽47列。库或配置改变的拒绝检查通过，GPU查表执行尚未验证。
2. 整层 dispatch：layered service 调用 `run(..., batch_dispatch=true)`，预备专家任务集合后只调用一次 `workers.dispatch`，各线程依次执行专家并按输出行聚合；专家末尾新增 barrier，避免下一个专家复用 hidden/CSF scratch 时上一个仍在读取。普通和CSF四核连续服务测试通过。
3. close重试：组件在释放资源后才标记closed；超时保留page/CPU。外层runtime也在子组件关闭成功前保留mailbox和映射，并记录已drain状态，阻止新任务进入；两项超时/重试测试通过。

本轮在这些修复范围内未发现新的确定性阻塞缺陷。此结论不表示未执行的GPU和性能场景已通过。

### 本轮实际执行

- 仓库只读挂载到已有nightly容器：layered policy/close单测，**6 passed**。
- 当前源码以 `-O3 -std=c++17 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -fPIC -shared -pthread` 重编CPU库，输出 `/tmp/qwen38-layered-reacceptance/kernel.so`。SHA256：`94b60c543c668d14b2c0c62d0599b0d43ee0794dc7bb1b9ad12080039387b190`。
- `check_qwen38_layered_service.py --cores 0,1,2,3 --backend b12x`：普通scale、CSF各通过。每种format涵盖两种几何，每种几何30次连续请求，共120次正常请求；交替M=1/2/3、重复route、复用page，与同步入口结果逐位一致；另有4次非法route检查，不推进成功completion。
- 独立scalar参考检查：**30项通过**，max_abs_error=0.00024414062。
- native/CSF partition、fallback、routing和malformed-input检查：**299项通过**。
- `git diff --check`通过。

上述host-memory服务测试不覆盖GPU映射可见性、GPU merge及PCIe行为；同步入口与批量入口复用了底层expert_worker，因此另跑scalar参考作为补充。CPU小形状测试不是模型性能基准。

### 剩余验收

仍未找到分层模式的真实校准采集产物、目标GPU集成结果、新trace、短→长→短生命周期及A/B/C服务性能证据。现有GPU planner测试仍是forced prefix、空表、safe_mask=0，不能覆盖真实schema3查表和B12X mask/merge。

需要在目标机完成：独立LFU oracle；真实成本表GPU匹配与回退；B12X mask与CPU结果合并；实际CPU专家零权重传输；连续任务页复用与rebalance/KV resize；A原分层GPU/B新框架零分流/C真实分流的交错测量，包含MTP接受率、ms/sweep、tok/s、fill bytes及尾延迟。本轮未给出“已减少端到端等待”或“已超过原分层GPU”的结论。

## 第一轮历史记录

## 符合设计的部分

- CPU 子集从原本层 miss 中选择，不依赖全局 LFU rejected；支持零前缀和未知 key 回退 GPU。
- CPU routes 在调用原分层 LFU 前从 histogram/active set 过滤，因此不会被该次 LFU 插入 GPU cache，也不会出现在正常 GPU fill 列表中。
- 原始 route sampling 与 GPU LFU 过滤集合分开；保留逐层 cache 和 adaptive worker。
- 请求领取、计算调用、完成发布位于 native service，一层只发布一份聚合结果；正常执行不回到 Python 发布 completion。
- 首版 FP32 GPU + FP32 CPU merge 符合数值设计；未将旧逐输出轮询的融合 epilogue 接回默认路径。

以上是源码审查结论，不替代 GPU 执行、实际传输量和端到端验证。

## 待修复/补齐

### 1. [P1] 校准未绑定实际执行库和关键运行配置

位置：`python/sglang/srt/layers/moe/qwen38_layered_hybrid.py:59–80`。

source_fingerprint 包含源码，但显式指定 `SGLANG_QWEN38_CPU_LIBRARY` 时实际加载的动态库未参与指纹。两份具有相同导出符号、不同实现或编译参数的库会接受同一成本表。metadata 也不包含影响 full branch elapsed 的 fill CTA 策略、shared-fill overlap 配置；源码集合遗漏实际编排的 cache_mailbox 和 B12X FP32 变换模块。

影响：替换库或调整执行配置后，旧分流表仍通过严格相等检查，却可能把已不划算的 miss 交给 CPU，违背基于完整分支成本决策的前提。

要求：绑定实际 CPU library 内容哈希/构建身份、校准使用的关键执行配置及相关实现指纹。添加更换库、fill/overlap 设置后拒绝旧表或明确 GPU 回退的检查。

### 2. [P2] 仍按专家逐次唤醒和等待 CPU 团队

位置：`cpu_nvfp4/kernel.cpp:298–380`，`expert():200`，`Workers::dispatch():129–135`。

新 service 只是在外层循环调用原 run；run 对每个 expert 调用 expert，expert 每次调用 workers.dispatch，后者每次 notify_all 后等待 pending==0。因此 CPU 选择 k 个专家仍触发 k 次团队唤醒/完成等待，设计的“一层任务一次团队 dispatch”没有实现。

要求：将专家集合循环移入一次团队 dispatch，正确保留每专家计算阶段 barrier、临时结果复用和聚合；或明确作为未完成项，并先用目标机证据证明保留现状满足等待预算。不能将“一次 mailbox 请求”表述成“完成整层批量 dispatch”。

### 3. [P2] close 超时后不能重试清理

位置：`qwen38_layered_hybrid.py:177–195`。

close 在 join 成功前将 self.closed=True；如果线程未按时退出，抛出 TimeoutError 后，即使它稍后退出，第二次 close 也会直接 return，无法注销 mapped page 和关闭 CPU mapping。

已通过提取实际 close 方法、替换线程边界复现：首次 timeout 后 closed=True，模拟线程退出再 close，page.close/cpu.close 均未被调用。

要求：区分 stopping 与完全 closed，停止请求可幂等，只有所有资源释放完成后设置 closed；超时时保留资源和可重试清理状态。检查外层 runtime 的 closed/异常清理同样支持该约定。

### 4. 性能与GPU正确性验收材料不完整

当前新增 GPU 测试只有一个方法内四组 planner 状态对照，使用 safe_mask=0、空成本表及 forced prefix，未执行 B12X safe_mask=1、真实 CPU service 与 GPU merge 的集成链路。参考仍调用本次修改过的同一 CUDA planner，不构成独立 Python LFU oracle。

未找到分层路径专用校准采集程序/真实校准表、A/B/C 服务结果、新 trace 或分层模式的短→长→短生命周期证据。已有 bottleneck_review_20261022/IMPLEMENTATION.md 记录的是全局共算优化，不能作为分层模式验收。

要求：补齐独立 LFU oracle；目标 GPU 验证实际查表、B12X mask、CPU miss 零权重传输、结果合并、连续页复用、错误退出、rebalance/KV resize/图 warmup；交错比较 A原分层GPU、B新框架强制零分流、C真实校准分流。记录 CPU领取/计算/结果发布及暴露等待，报告 MTP接受率、ms/sweep、tok/s、fill bytes 和尾延迟。

## 本次实际运行

- 本地已有 nightly 容器、仓库只读挂载：`test_qwen38_layered_policy.py`，3 passed。
- 使用当前源码自动构建 CPU library，在 host-memory page 上调用真实 qwen38_cpu_layered_service；序号 1/3/5 对应 M=3/1/2，结果与同步 CPU B12X 入口逐位一致；非法 route 发布 failure，不推进成功 completion，线程退出。此项不覆盖 CUDA/system-memory 可见性、GPU merge 或四核调度性能。
- close timeout/retry 的控制流复现确认上述资源清理问题。
- `git diff --check` 通过。
- 当前主机无可用 CUDA 环境，未运行 GPU planner/数值测试、完整模型或性能 benchmark。没有作出已超过原分层 GPU 的结论。

## 交接建议

先修复校准绑定和 close；完成整层 dispatch 或提供保留现状的目标机测量。随后补齐 GPU 正确性与 A/B/C 性能证据，再判定验收通过。sync/device/async 按用户后续删除计划另行处理，本次不要求为它们增加兼容代码。
