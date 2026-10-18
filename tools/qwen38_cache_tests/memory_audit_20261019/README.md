# Qwen38 显存账本复核（2026-10-19）

对当前工作区的三项显存收益改动做一次“分配能对上、释放能对上、映射能对上”的
闭环复核，并把基础槽位从 6388 调到 6426 的依据留在本目录。所有结论都只引用本
目录内的产物；profiler 下的耗时一律不当成吞吐结论。

## 复核对象

1. `python/sglang/srt/eplb/expert_location.py`：host cache 且 EP1、无 EPLB/无
   recorder 时不再构造 `[48, 512, 512]` 的恒等 expert location 图（GPU 侧
   96 MiB，权重加载与 KV 定容之前）。
2. `python/sglang/srt/layers/quantization/fp8.py`：MXFP8 CUTLASS 补齐后让逻辑
   `weight` 直接指向已安装的 `weight_mxfp8_padded` 视图，权重更新走复制而非
   重新绑定，避免同一份权重同时挂在两份 storage 上。
3. `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py`：
   QSA + hybrid GDN 组合不再先构造 dense backend 再丢弃（FlashInfer 的
   进程级 workspace 会跟着丢弃的实例留下，256 MiB）。
4. `launcher-before.sh` / `launcher-final.sh`：`CACHE_SLOTS_TOTAL` 6388 -> 6426，
   用于覆盖最终 220032 token 阶段的 VMM handle 对齐。`MAX_TOTAL_TOKENS` 保持
   注释状态，KV 上限不变。

正式 launcher 已在 01:19 同步该配置并重启：`max_total_num_tokens=220032`、
`context_len=220000`、`chunked_prefill_size=8192`、`max_running_requests=1`、
`mem_fraction_static=0.90`、`port=8000`，01:20:38 ready。三档真实预算为
`(80000, 160000, 220032) -> [7045, 6681, 6426]`，两轮 74K/170K 扩容后都回到
`returned expert loan: KV=220032 expert slots=6426`。

## 采集方式

* `hook/sitecustomize.py`：把诊断 sidecar 挂到既有的 `/start_profile`、
  `/stop_profile` 路径上，只改证据采样，不改推理、权重加载和 allocator 策略。
  需要 `QWEN38_AUDIT_OUTPUT` 才生效。
* `capture.py`：只读侧车（Python 所有权遍历、VMM arena、`mem_get_info`、
  `nvidia-smi`、`memory_stats`），同时把 history ring 提到 2000000 条，
  74K/170K 请求不会截断历史。
* `run_capture.py`：8 个阶段（short/long 74K/long 170K 各两轮，加两次 flush 后
  短请求与收尾 idle），`temperature=0`、`max_new_tokens=32`、`ignore_eos`，
  阶段结果见 `clean/requests.json`（170K 单请求 17.2 s）。
* 诊断跑在隔离端口 8001，不与正式 8000 端口混用。

## 账本对账

`reconcile_lifetimes.py` 用同一阶段的 start/stop 快照做闭合检查：阶段内
`requested_size` 增量必须等于增量 history 里 alloc 减去 free_completed。
9 个阶段全部 `unexplained_bytes = 0`（`clean-lifetimes.json`）。

| 阶段 | 事件数 | alloc 减 free_completed | requested 增量 | 无法解释 |
| --- | ---: | ---: | ---: | ---: |
| short-first | 16296 | 164691 | 164691 | 0 |
| long-74000 | 130492 | 26290783 | 26290783 | 0 |
| long-170000 | 167308 | 116674448 | 116674448 | 0 |
| short-after-flush | 16562 | -121308671 | -121308671 | 0 |
| repeat-74000 | 130542 | 4612935 | 4612935 | 0 |
| repeat-170000 | 167307 | 116674448 | 116674448 | 0 |
| short-after-second-flush | 16560 | -121308671 | -121308671 | 0 |
| short-repeat | 16249 | -1720 | -1720 | 0 |
| idle-final | 1 | 0 | 0 | 0 |

两轮 170K 的事件数（167308 与 167307）与净增字节完全一致，最后一轮短请求净变化
-1720 B，即整轮周期跑完不残留增长；flush 阶段净返还约 115.7 MiB。

脚本原先用 mtime 配对 start/stop，同一秒内写完会配错（会把带 74050 条启动历史的
那份快照当成前缀）。现按文件名里的时间戳配对，并且同一份 start 不允许被两个阶段
重复认领。前缀校验忽略 `time_us`：同一份 history 两次 dump 的该字段本身会有 1
微秒抖动（`short-first` 的 74050 条里有 3550 条差 1 微秒），其余字段与顺序仍严格
相等。

## 采集器自身造成的假象

`capture.py` 之前从自身 frame 开始遍历，Python frame 自引用形成引用环，在服务端
GC 冻结期间额外留住了 HC 中间结果。修正后只遍历 caller 链，
`check_capture_lifetime.py` 三项断言同时成立（修正版即放即释、旧写法可复现、旧写
法确属可回收环），结果见 `capture-lifetime.txt`。

同一阶段修正前后对比（`before` 与 `clean`，`repeat-170000`）：

| 字段 | 修正前 | 修正后 |
| --- | ---: | ---: |
| `unknown_owner_bytes` | 241513472 | 169281024 |
| `unknown_owner_count` | 123 | 103 |
| idle 态 `physical_reserved_union` | 10263461888 | 9873391616 |
| idle 态 `driver_free` | 2108162048 | 2395406336 |

也就是说这 68.9 MiB / 20 个张量（idle 态 allocator 保留量差 372 MiB）是诊断工具
自己留住的，既不算模型泄漏，也不算显存回收收益。`clean/` 是本轮唯一权威数据，
`before/`、`final/` 保留用于说明该假象的量级。

## Nsight 与 CUPTI 映射对齐

`analyze_native.py` 读本地 Nsight SQLite 导出，逐地址维护活跃映射，再与同一进程
idle 账本里的 allocator 段加 VMM backing 对齐（`native/native-reconciliation.json`）：

* `mapped_live_bytes` 等于 `allocator_plus_vmm_bytes` 等于 30255611904，
  `unexplained_mapped_bytes` 为 0。
* 成功创建且未释放的 handle 数与仍在映射的数量一致：`cuMemCreate` 999 次成功减
  `cuMemRelease` 416 次得 583，`cuMemMap` 998 次减 `cuMemUnmap` 415 次得 583
  （另有 1 次返回 1 的 `cuMemCreate` 失败尝试，不计入活跃 handle）。
* 非 VMM 的常驻分配只剩 3 笔 `cudaMalloc`，共 8520704 B。

复核命令只读 sqlite 与账本，不碰 GPU：

    python3 analyze_native.py --sqlite /tmp/qwen38-native-memory.sqlite \
      --idle-ledger native/idle-final/ledger.json.gz --output /tmp/native-recheck

`native/` 这一轮是在修正采集器之前采集的，其 `unknown_owner_*` 仍是修正前的值；
allocator 段与 VMM backing 来自 torch 快照本身，CUPTI 对齐不受该假象影响。

## checkpoint 头对账

`checkpoint_audit.py` 只读 safetensors 头，把已加载的 GPU 参数逐个对上本地
checkpoint（`checkpoint-reconciliation.json.gz`、`checkpoint-headers.json.gz`）：
parameters 1195、casts 72、derived 277、buffers 159、unmatched 0；GPU 具名
storage 合计 7547152030 B，覆盖 6087 个 checkpoint 张量（checkpoint 共 375487 个
header、102138950916 B）。

## 测试

* `cpu-suite.txt`：`run_cpu.py` 347 项通过。
* `registered-cpu.txt`：注册 CPU 用例 7 passed / 5 subtests。
* `gpu-reload.txt`：`test_cutlass_padded_weight_shares_logical_storage_after_reload`
  在 GPU 上通过（需要真实 CUDA，本轮未重跑，沿用 01:13 记录）。
* 本次补跑（CPU-only 容器：`--network none`、不传 GPU、源码只读挂载）：
  `test/registered/unit/model_executor/model_runner_components/test_attention_backend_setup.py`
  3 passed / 2 subtests；`test/registered/unit/test_runtime_context.py` 以
  `-k "expert_location or host_cached_identity"` 过滤后 2 passed / 3 subtests。
  这两个新用例（QSA 不留下被丢弃的 dense workspace、host cache 恒等 expert map
  不需要 location storage）此前没有任何日志记录，故补跑并留档。

## 产物与复现

* 每阶段三份 gzip 账本：`clean/<stage>/{ledger,segments,owners}.json.gz`，汇总在
  `clean/summary.json`，idle 显存见 `clean/nvml-idle.json`。
* 超过 GitHub 单文件上限的 torch 快照按 SHA256 切片存放于 `profile-parts/`，
  `python3 restore_profiles.py` 校验哈希后还原
  `repeat-170000-memory.pickle.gz`（清单见 `profiles.json`，还原产物已
  `.gitignore`）。原始 pickle 仍在
  `/prod/homelab-llm/models/startup-memory/full-audit-clean-20261019/`。
* 账本复算：`python3 reconcile_lifetimes.py --captures
  /prod/homelab-llm/models/startup-memory/full-audit-clean-20261019 --output
  clean-lifetimes.json`。

## 本轮关闭状态

诊断 hook 已撤：正式服务进程环境只有 `PYTHONPATH=/workspace/sglang/python`、
`SGLANG_QWEN38_OBSERVE=0`、`CACHE_SLOTS_TOTAL=6426`、`MEM_FRACTION_STATIC=0.90`，
没有 `QWEN38_AUDIT_OUTPUT`，也不含 hook 目录；端口恢复为 8000（宿主侧
`127.0.0.1:8329` 转发），诊断端口 8001 无残留进程。

## 未覆盖

* 未重跑 GPU 端到端吞吐，profiler 下的阶段耗时不作为性能结论。
* `native/` 的 Python 所有权列是修正采集器之前的数值，如需同批次证据须重跑
  Nsight 采集（需独占 GPU）。
