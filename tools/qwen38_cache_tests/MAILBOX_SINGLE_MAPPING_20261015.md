# Mailbox 单份源映射修复（2026-10-15）

已修复初版保留两份 expert 源权重的问题。普通 transport 和 mailbox 现在共用
`Qwen38ExpertPack` 拥有的唯一映射。核心代码提交：`241c5dbee2a0273d89c40c1becdf6ceb24c7bec9`。

## 实现

1. 首次准备 CUDA 源时等待预取完成，关闭原只读映射，再用同一个文件描述符创建
   `mmap.ACCESS_COPY` 私有映射，替换 pack 自身的 `_mapping`。
2. CPU 普通 transport 与 mailbox 都读取这一份映射。公开的 `tensor_bytes` 和
   `expert_tensor_bytes` 返回只读 memoryview，不改变原有只读接口约定。
3. CUDA host registration 物化并锁定私有 COW 页后，仅对源文件执行
   `POSIX_FADV_DONTNEED`，释放不再使用的预取文件页缓存；不执行全局 drop_caches。
4. mailbox 只负责注册/注销与释放 frombuffer view，pack 负责最终关闭映射。
   若旧映射还有导出 view，升级明确失败并保留旧映射，避免悬空引用。

原文件仍以只读方式打开，私有映射的修改不能写回磁盘。曾尝试共享可写文件
映射直接注册，当前环境返回 CUDA invalid argument；该实现未部署，失败日志
保留为 `gpu-shared-file-rejected.*`。最终方案继续使用已验证的私有映射注册，
但不再单独保留第二份源映射。

## 内存证据

已保存的线上进程 smaps 和 mincore 快照：

| 项目 | 初版 | 单份版本 |
| --- | --- | --- |
| expert 源映射数量 | 2 | 1 |
| 源映射 RSS 总计 | 约 126.56 GiB | 63.28 GiB |
| CUDA 注册的源页 | 约 63.28 GiB | 约 63.28 GiB |
| 源文件页缓存 | 约 63.28 GiB，与初版只读映射是同一批页 | 0 字节 |

单份源映射 RSS 为 `67948511232` 字节，均为私有匿名 COW 页；
源文件缓存为零。前后对照去掉的是重复常驻源数据，不是 GPU 专家 cache 槽位。
注册期间可能暂时同时存在预取文件页与正在物化的私有页，注册后会释放前者。

这 63.28 GiB 注册源仍必须锁页。旧文件缓存本来就被 Linux 视为可回收内存，
因此删除它后 MemAvailable 不会等额增加：快照中 MemFree 约 64.5 GiB，
MemAvailable 约 80.4 GiB。不能把 RSS 减少解释为多出同样大小的锁页预算。

## 离线验证及吞吐

CPU 235 项、GPU 48 项测试全部通过，无跳过。新增测试覆盖：同一源
映射、旧映射关闭、公开 view 只读、COW 文件保护、导出 view 冲突后的状态有效、
mailbox 注销后 pack 仍能读取并最终关闭。完整 GPU 回归覆盖字节传输、迁移、
sampling、跨 stream 和 NVFP4 层计算。

单独测试容器复用初版同一输入文件、启动脚本和配置，512/4096 token 输入，
每次固定输出 768 token；各一次预热、三轮正式测量。以下均为三轮中位数，
吞吐单位 tok/s；mailbox 段统计输出位置 513–768，并检查实际 mailbox 调用增量。

| 输入 token | 初版整段 decode | 单份整段 decode | 初版 mailbox 段 | 单份 mailbox 段 |
| --- | --- | --- | --- | --- |
| 512 | 38.55 | 39.63 | 56.10 | 57.27 |
| 4096 | 39.95 | 39.39 | 57.04 | 56.62 |

未发现吞吐回退。未重新运行四方案及无优化基线；初版测试只作为历史参照，
不把小幅波动解释为额外确定增益。配置仍为 MTP off、5140 总专家槽位、lazy PLE、
mem fraction 0.91、chunk 8192、单请求，模型及 NVFP4 计算方式不变。

## 启动代价和线上测试交接

测试容器在文件缓存已有数据时 ready 约 55.1 秒；实际重启前
文件缓存已经释放，本次从盘预取 63.28 GiB 用时 71.89 秒，总 ready 等待
122.05 秒。释放源文件缓存会使后续重启需要重新读盘，应区分
冷/热启动时间。原始启动日志已保存。

用户随后明确要求线上测试自行执行。此要求之后不再发送线上测试请求、不再
重启或调整服务；线上后续性能和业务验收交给用户。此前已产生的验收和内存
快照保留在 `production/`，不把它们表述成用户完成的业务验收。

原始记录：`/home/user/workspace/sglang/results/mailbox-single-mapping-20261015/`，
包括 `cpu.log`、`gpu.json`、`benchmark.py`、`inputs.json`、`single/report.json`、
`file-cache.json`、`production/memory.json` 和已产生的启动/请求日志。

修复前备份分支 `codex/mailbox-before-single-mapping-20261015` 指向初版双映射
部署 `ace38b3867`；更早的无本包优化基线仍为 `codex/mailbox-before-20261015`。
回滚或重启由用户决定执行。
