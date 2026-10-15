# Mailbox direct mmap transport

单 token mailbox decode 的 miss 从注册的 expert-pack 内存读入 GPU cache slot，
删除 mailbox Python payload staging；hit 和 miss 均无需 CPU 确认。
单 token decode 的 adaptive sampling 同样使用 GPU 直接填充，route 在独立
缓冲区采集后批量交给 CPU。prefill、warm-up、多 token verify 保留原 transport，
与 mailbox 读取同一份源映射。NVFP4 专家计算、router 权重和累加顺序保持原样。

## 单份主机源权重

`Qwen38ExpertPack` 拥有唯一映射。首次准备 CUDA 源时等待预取结束，关闭旧的
只读映射，改为私有 `mmap.ACCESS_COPY` 映射。所有公开 byte view 都通过
`toreadonly()` 保持只读；普通 transport 和 mailbox 共同使用该映射。
RTX 5090 D 不支持只读 host registration，注册使用 `cudaHostRegisterMapped`，
会物化这份映射的 COW 页。注册成功后，对源文件执行 `POSIX_FADV_DONTNEED`，
释放已不需要的预取文件页缓存；没有全量第二份源权重常驻。注册过程中可能
暂时同时存在源文件缓存和正在物化的私有页。

私有映射不能修改磁盘文件，也不要求 expert pack 可写。GPU 注册源约 63.28 GiB
仍需锁页驻留，不能将删除重复文件缓存与减少同等大小的锁页预算混为一谈。
初版曾同时保留只读文件映射和独立私有映射，已由单份映射实现替代。

mailbox 关闭先排空 GPU reader，再 unregister 并释放 tensor view；映射由 pack
最终关闭。初始化异常不会关闭其他 transport 仍需使用的 pack 映射。
控制缓冲区显式使用 CPU 默认设备。若升级映射时仍有导出 view，明确报错并
保留旧映射，避免悬空引用。

## GPU 直接执行，取消 ready 往返（2026-10-16）

planner、alpha scatter、fill 和专家计算按同一 compute stream 执行。
fill 直接读取 GPU plan 与固定源地址，不再等待 CPU `ready`，也不再向
host 发布 miss plan。删除只做统计与确认的 mailbox worker、poller 初始化、
host heads/plans/ready 缓冲区和 epoch。普通 prefill/sampling transport 的
worker 保留。

每层 GPU 计数器累计 hits、all-hit calls、misses、miss calls；读取统计或
同步时等待各层完成，再将增量加入原有计数。多次同层 plan 可以连续排队，
不会因 CPU 未及时读取而漏计；重复 snapshot 不会重复累加。
cache slot 的前序 reader event、队列背压、迁移前排空、非法 plan 的 GPU trap、
映射 failure flag 和 completion watchdog 保留。GPU 不再有等待 host ready
的循环，超时仍通过 completion watchdog 检测。

这项修改仍注册完整源映射，因此不会消除约 128 MiB 的疑似 GPU 映射页表成本。
它也没有新增 GEMV/搬运并发；GEMV 与 fill 的重叠是另一个待分析的改动。

验证和交接见 [MAILBOX_NO_READY_20261016.md](MAILBOX_NO_READY_20261016.md)。

## Decode sampling 直接填充（2026-10-16）

sampling 不再要求回退到 CPU plan/staging；GPU planner 记录原始、排序去重后的
expert ID，权重继续由既有 fill kernel 搬进显存 cache。窗口完成后按层异步 D2H，
CPU 分析输入的 layer/cycle/index/active IDs 和缺口拒绝规则保持不变。
切回旧 transport 时交接部分窗口，保留 prefill 边界和 verify 的观察顺序。

双缓冲在 KV 预算之前预分配。48 层、512 步、top-k 10 时，GPU 和 pinned host
各增加约 2.06 MiB；分析队列的批量样本也受每层双 bank credit 约束。
这不保证 context 容量固定，实际分配与 allocator 预留以启动日志为准。

当前改动与验证见 [MAILBOX_DECODE_SAMPLING_20261016.md](MAILBOX_DECODE_SAMPLING_20261016.md)。

## 验证

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 tools/qwen38_cache_tests/run_cpu.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output /tmp/mailbox-gpu.json
```

此前单份映射版本的 235 项 CPU、48 项 GPU 测试通过。覆盖 required tensors、resident alpha、单/多
miss、重复 expert、跨 stream、缓存迁移、sampling、失败路径和对齐/非对齐尾块。
新增用例验证同一个源映射、公开视图只读、原文件不变、旧 view 冲突处理，以及
mailbox unregister 后 pack 仍可读取并最终关闭映射。

当前单份映射的测量与部署见
[MAILBOX_SINGLE_MAPPING_20261015.md](MAILBOX_SINGLE_MAPPING_20261015.md)。
初版四方案对照见 [MAILBOX_DEPLOYMENT_20261015.md](MAILBOX_DEPLOYMENT_20261015.md)。
历史四方案对照及单份映射 benchmark 在本次 sampling 修改前采集；当时前 512 个
decode token 绕过 mailbox。不能将那些测量视为本次 sampling 改动的吞吐结果。
