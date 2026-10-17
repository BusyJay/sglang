# 2026-10-18 专家源启动优化

原日志中专家源准备耗时 151.769 秒，主模型加载 154.01 秒，tokenizer_e2e
177.37 秒。专家源每次都重新读取 63.28 GiB 到匿名 THP 内存；保留磁盘页缓存
不能保证缓存不会被回收。当前 SSD 是 Samsung 990 PRO 4TB，端点和上游端口
均显示链路 2.5 GT/s ×4，端点支持 16 GT/s；没有修改 PCIe 配置。

4 GiB 冷读测试中，buffered/direct、1/4/8 线程均约 0.85–0.89 GiB/s。
当时没有发生直接回收或内存规整，因此没有引入无明显收益的并行读取。
独立完整匿名源准备 67.63 秒、CUDA 注册 0.61 秒；它与原服务的内存状态不同，
不能把两次差异全部归因于某一项代码修改。

新增可选 `shared_thp` 模式，把专家源填入持久于容器生命周期的主机 tmpfs。
CUDA 直接注册这一份共享映射，不再分配另一份匿名专家源。默认仍为
`anonymous_thp`，只有明确配置后才使用共享缓存。

| 完整 63.28 GiB 源 | 准备秒数 | CUDA 注册秒数 |
|---|---:|---:|
| anonymous_thp 独立测量 | 67.627 | 0.608 |
| shared_thp 首次冷填充 | 74.260 | 0.605 |
| shared_thp 新容器复用 | 0.0013 | 0.591 |

首次填充没有加速，主机重启后仍需填充。缓存按原文件的 device/inode/size/
mtime_ns/ctime_ns 区分；写入加锁，完成填充并复核文件身份后原子发布。中断写入不
会发布半份数据；同一文件的遗留临时文件会在重试时清理。原 expert pack 必须
保持不可变。读取过程中逐段请求释放冗余的磁盘文件页缓存。

实际模式两次容器重建结果（都命中同一个 tmpfs 文件 inode）：

| 指标 | 第一次 | 第二次 |
|---|---:|---:|
| expert source ready 秒数 | 0.001 | 0.001 |
| PLE allocation 秒数 | 3.955 | 3.950 |
| PLE cache read 秒数 | 29.161 | 25.168 |
| main load_weight 秒数 | 46.21 | 34.73 |
| tokenizer_e2e 秒数 | 68.44 | 61.42 |
| compose 重建至外部 health=200 秒数 | 89.52 | 85.26 |

PLE 仍需读入 pinned 表；剩余 25–29 秒的 PLE 读取也受当前 SSD 链路限制。
健康检查总时间还包含容器停启、入口脚本和启动 warmup，不能与 tokenizer_e2e
或 source preparation 混用。此处是两次实际启动观测，不是任意缓存冷热的保证。

保留用户当前配置：6340 cache slots、MTP=2、chunk=8192、context=220000、
mem_fraction_static=0.90。两次启动的 max_total_num_tokens 都是 229376；独立
prefill arena 为 0，借用 decode cache 1350 MiB，decode cache 16716.85 MiB。
此前 5340 slots 对应的 425088 token 容量不能与当前 6340 slots 直接比较。

内存核验：实际服务的源映射 RSS/PSS 为 67948511232 字节，Anonymous 为 0，
ShmemPmdMapped 为 67947724800 字节。原磁盘 experts.bin 驻留只有 49152 字节。
这是同一批 tmpfs 页被文件、CPU 映射和 CUDA 注册共同使用；smaps 中单个映射
的 Private_Dirty 名称不表示额外存在一份匿名副本，也不能把 tmpfs 占用与映射
RSS 再相加。共享缓存会在停止语言服务后继续占用约 63.28 GiB RAM；删除缓存
或重启主机才释放，不会保存到 SSD。

部署示例（需要主机 tmpfs 至少有 63.28 GiB 可用空间，普通容器 32 GiB shm 不够）：

```yaml
environment:
  SGLANG_QWEN38_SOURCE_MODE: shared_thp
  SGLANG_QWEN38_SOURCE_CACHE_DIR: /dev/shm/sglang-qwen38-experts
volumes:
  - /dev/shm/sglang-qwen38-experts:/dev/shm/sglang-qwen38-experts
```

生产 `models/fp4-local.yml` 已配置该主机目录绑定，所以 `down/up` 或强制重建
语言容器都能复用。ViT 与其他服务没有重建。内核需支持 tmpfs THP；当前机器
shmem_enabled=advise。磁盘目录会明确拒绝，避免误用之前无法注册的磁盘共享
映射。切换 checkpoint 会产生新的缓存键，不自动删除其他键；需要回收空间时
先停止使用缓存的服务，再删除该专用目录中的旧 `.bin`/`.tmp` 文件。

复测工具：`benchmark_shared_source.py`，分别在新容器执行 `--mode anonymous_thp`
和 `--mode shared_thp`；后者再次执行应命中缓存。完整尺寸每次校验首尾和固定
随机的 258 页并调用实际 mailbox 注册/注销。`read-probe.json` 和完整源 JSON
保留原始测量。GPU suite 在 shared_thp 下 62 项全部通过，无跳过。

最终回归：298 项 CPU 测试、shared_thp 下 62 项 GPU 测试通过。实际服务先以
23422-token prompt 持续生成 10240 tokens，在生成过程中排入 46821-token
prompt，随后完成 64-token 输出，两次 cached_tokens 均为 0。周期统计记录到
1 次 adaptive rebalance、6015 专家迁移，borrowed arena 恢复；无 OOM/回撤，
容器重启次数 0，最终 health=200。详情见 decode-then-prefill.json 和
live-validation.json。
