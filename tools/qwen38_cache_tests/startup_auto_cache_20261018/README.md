# 2026-10-18 PLE 复用、自动预算与目录清理

Qwen38 loader 默认 `SGLANG_QWEN38_STARTUP_CACHE=auto`。构造模型元数据后、
读取权重前，计算真实专家源大小与所有 pinned FP4 PLE 表的 packed 大小。
当前模型分别为 63.28 / 26.82 GiB，总计 90.10 GiB。

自动选择顺序固定为：专家权重 + PLE → 仅专家权重 → 仅 PLE → 两者均使用原有
非持久加载。最后一档仍是匿名专家源和原 pinned PLE 表，不会改成 lazy PLE，
也不会改变数值、GPU cache slots、prefill chunk 或调度策略。

预算取以下约束的最小值：

- 主机 MemAvailable + 当前可复用缓存已占的空间 − 预留。
- cgroup v2 容器可用内存 + 可复用缓存 − 预留（存在限额时）。
- tmpfs 剩余空间 + 可复用缓存。
- 可选 `SGLANG_QWEN38_STARTUP_CACHE_MAX_GIB` 上限。

默认预留为有效总内存的 20%，至少 8 GiB，可通过
`SGLANG_QWEN38_STARTUP_CACHE_RESERVE_GIB` 调整。已经占用的匹配缓存会加回预算，
避免第二次启动因缓存仍在内存中反而降级。这是持久缓存预算，不是完整模型的
内存准入检查；非缓存路径也需要其原有运行内存，HiCache 等池另行分配。

PLE 直接注册 tmpfs 映射给 CUDA，不再创建另一份 pinned 表。绑定在 embedding
上的 owner 持有映射、原始 tensor 和设备指针；关闭时等待 GPU 读完、注销并
释放 tensor 别名后关闭映射。空间不足、内存/权限等资源错误回退到原路径。
PLE 注册失败时先释放异常栈持有的映射，再清理无人使用的已发布缓存，避免
回退同时保留两份表；源文件变化或损坏等数据错误不会静默吞掉。

生产已绑定专用主机目录：

```yaml
environment:
  SGLANG_QWEN38_STARTUP_CACHE: auto
  SGLANG_QWEN38_SOURCE_CACHE_DIR: /dev/shm/sglang-qwen38-startup
volumes:
  - /dev/shm/sglang-qwen38-startup:/dev/shm/sglang-qwen38-startup
```

该目录专门存放生成的缓存，不能存放原 checkpoint 或其他业务数据。启动核对
当前输入文件身份、表形状、文件名与大小；匹配数据保留，其他数据及中断写入
先清理再重建，再根据释放后的空间选择档位。没有镜像 tag/version 判断；同一
份输入在更换镜像或挂载路径后仍可复用。输入变化才产生不同的文件身份。

每份映射持有共享 lease，构建者持有独占 writer lock。清理只在两个锁都能
独占取得时进行；忙的文件跳过。发布前使用临时文件，完整填充后原子改名。
保留空的 `.lock` / `.lease` 协调文件，避免等待者与新锁文件使用不同 inode；
`.cache-manifest.json` 记录当前预期文件清单。禁止把 `/dev/shm`、`/tmp`、`/run`
等公共目录作为可清理目录，也禁止缓存目录包含任何输入模型文件。

首次部署预置的 `obsolete-container-data` 已自动清理，原专家缓存 inode=1470
被保留且命中；旧专用目录直接改名，没有再次复制 63 GiB。新 PLE inode=1695，
连续重建保持不变。实际服务中分别尝试清理这两份活跃缓存，均被 lease 拒绝，
见 live-cache-leases.json；单元测试覆盖任意垃圾文件、中断写入、活跃 writer、
活跃 GPU reader、旧缓存退出后的回收及匹配数据的无 I/O 复用。

| 观测 | PLE 准备秒数 | PLE 注册秒数 | load_weight | tokenizer_e2e |
|---|---:|---:|---:|---:|
| 首次填充 PLE，专家源已缓存 | 18.206 | 0.399 | 31.12 | 52.64 |
| 两份缓存均命中 | <0.001 | 0.217 | 5.33 | 26.60 |
| 再次命中 | <0.001 | 约 0.22 | 5.46 | 26.52 |
| 部署复测 | <0.001 | 约 0.22 | 5.45 | 26.47 |

此前仅专家缓存的 tokenizer_e2e 为 61.42 秒；更早重建匿名源时为 177.37 秒。
上表是实际缓存冷热条件下的观测，不保证首次主机启动也有同样速度。两份缓存
会在停止容器后继续占约 90.10 GiB RAM，主机重启后需要重新填充。tmpfs 内容
与 CPU/CUDA 映射使用同一份物理页；memory.json 中两份映射的 Anonymous 都为 0。

当前配置仍为 6340 slots、8192 chunk、MTP=2、context=220000、静态比例 0.90。
KV 自动计算为 230144 tokens（此前 229376），独立 arena 为 0，借用 1350 MiB。
ViT 与其他服务未重建。自动策略不需要人工设定最大缓存；上限参数仅用于约束
持久驻留规模。`SGLANG_QWEN38_STARTUP_CACHE=off` 可恢复原手动源模式和 PLE
加载；auto 下 MAX_GIB=0 会选择非持久路径并回收未使用的当前缓存。

验证包括 309 项 CPU 回归、64 项 GPU 回归（无跳过），其中新 PLE GPU 测试比较
原 pinned 表逐字节输出、非法 IDs、跨 stream、CUDA Graph、重新打开缓存和
空间不足回退；CPU 覆盖四档边界、cgroup 预算、已有缓存加回及注册失败清理。
按用户要求，本轮没有运行额外长请求。

最终部署重建：load_weight=5.40 秒，tokenizer_e2e=26.41 秒，详见
deployed-final.json。最终 health=200，重启次数 0。
