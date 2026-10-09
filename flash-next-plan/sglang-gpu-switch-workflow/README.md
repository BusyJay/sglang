# SGLang 5090 异步测试切换器

这套脚本在同一张 RTX 5090 上切换：

- 开发后端：目标机器已有的常规 SGLang Docker 服务（容器名、镜像和模型自行配置）
- 测试后端：修改后的 Qwen3.8-Flash-Next SGLang

Codex 不同步等待测试。正常流程是：

```text
Codex 更新 .agent-workset 和测试 job
→ dispatch 将 job 写入本地队列并立即返回
→ 当前 Codex turn 立即结束
→ 独立 supervisor 延迟数秒后接管
→ 停止常规 SGLang后端
→ 启动测试后端并执行整批测试
→ 停止测试后端
→ 恢复常规 SGLang后端
→ /health_generate 返回成功，写入 latest-test.json
→ 立即执行 completion.command 唤醒同一个 Codex session
→ Codex 读取结果、日志和工作集合进行验证
```

全部实现只使用 Python 标准库、磁盘队列、PID 文件和 `nvidia-smi`，不依赖 systemd 或第三方 Python 包。

## 1. 安装

归档恢复补丁后可直接使用，无需安装旧 `tools/qwen38_dev` 模板：

```bash
cp flash-next-plan/sglang-gpu-switch-workflow/config.example.json \
   flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

命令均直接使用补丁恢复后的归档脚本。先初始化工作集合（已有文件不会覆盖）：

```bash
mkdir -p .agent-workset
cp -an flash-next-plan/agent-workset-template/. .agent-workset/
```

将工作集合中的占位符替换为实际环境、状态和下一步。

如果配置包含本机路径或密钥，将 `config.local.json` 加入 `.gitignore`。

## 2. 配置

编辑 `config.local.json`：

- `worktree`：修改后的 SGLang 工作树。
- `workset`：工作树内的 `.agent-workset`。
- `development.stop_command`：停止常规 SGLang后端的命令。
- `development.force_stop_command`：可选；正常停止超时后的强制停止命令。
- `development.start_command`：启动常规 SGLang后端的命令。
- `development.health_url`：常规 SGLang后端健康地址。
- `preflight_commands`：替换 `LOCAL_TEST_IMAGE` 和 `NORMAL_SGLANG_CONTAINER`，投递前检查本地镜像和常规容器。
- `test.batch_command`：当前模板直接运行 `docker_gpu.sh`，镜像和 GPU 显式指定；其内部 watchdog 管理自己的唯一测试容器并清理。
- `test.batch_timeout_seconds`：设为 0，避免外层超时杀掉负责清理的 watchdog；小 GPU 测试由内层执行 300 秒硬上限，以及 60 秒疑似停滞 + 至少 30 秒连续确认后自动 kill。
- `completion.command`：必须替换为目标机器真实可用的同一 session 唤醒命令；模板中的路径是占位符，不是提供的可执行程序。接收 `latest-test.json` 路径，唤醒后的 Codex 应先读取该文件及日志。
- `completion.required`：模板设为 true，空命令会在停服前拒绝投递。
- `completion.delay_seconds`：设为 0，ready 后写入结果即调用；唤醒机制实际调度延迟取决于本机集成。

模型转换或 canary 时将 batch_command 改为 `bash tools/qwen38_cache_tests/docker_model.sh ...`，参数见 `RUN_ON_GPU.md`；仍保留外层超时 0。旧 `docker-job` / `jobs/template.json` 属于历史方案，不是当前测试入口。

这些模板提供调度与唤醒钩子，尚未配置目标机器的 session 唤醒集成。仅恢复 Docker 服务不会自行触发 Codex。

命令使用 JSON 数组，不使用 shell 字符串。支持占位符：

```text
{worktree}
{workset}
{result_dir}
```

## 3. 启动常驻 supervisor

从普通终端启动一次，确保它不是常规 SGLang后端的子进程：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test daemon-start \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

查看状态：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test daemon-status \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

Supervisor使用双fork脱离终端，轮询本地磁盘队列。

无待执行或运行任务时可手动停止（活动任务会拒绝停止）：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test daemon-stop \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

## 4. Codex投递测试

Codex在当前turn中只运行：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test dispatch \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

`dispatch`先做只读前置检查，再保存配置快照、写入job文件，输出接收JSON并退出，例如：

```json
{
  "accepted": true,
  "phase": "dispatched",
  "job_id": "20261009-180000-1234",
  "supervisor_pid": 5678
}
```

随后Codex必须立即结束当前turn。不要在同一turn中：

- 停止常规 SGLang后端；
- 查询status；
- 轮询测试；
- 等待健康检查；
- 读取测试结果。

这些工作全部由supervisor执行。

## 5. 工作集合契约

投递前必须存在：

```text
.agent-workset/RESUME.md
.agent-workset/COMPACT.md
.agent-workset/STATE.md
.agent-workset/NEXT.md
.agent-workset/TESTS.md
```

Codex在dispatch前更新这些文件。Supervisor将结果写入：

```text
.agent-workset/artifacts/gpu-switch/<timestamp>/
.agent-workset/artifacts/latest-test.json
```

恢复后的Codex首先读取：

```text
RESUME.md
COMPACT.md
STATE.md
NEXT.md
artifacts/latest-test.json
```

## 6. Supervisor行为

Supervisor会：

1. 延迟 `dispatch.delay_seconds`，让当前Codex请求结束；
2. 运行常规 SGLang停止命令；
3. 正常停止超时后，运行可选的强制停止命令；
4. 等待5090达到配置的空闲显存；
5. 运行带活动 watchdog 的 Docker 测试批次；
6. 保存通过、失败或 suspected_stall 的详细结果；
7. 测试脚本清理自己的唯一测试容器；
8. 启动常规 SGLang并等待健康；
9. 写入结构化结果；
10. 无额外延时调用 `completion.command`；测试失败也会唤醒以供审查，恢复失败则保存错误且不唤醒。

无论测试成功或失败，清理阶段都会尝试恢复常规 SGLang后端。

## 7. 手动恢复

如果机器或测试进程异常中断：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test recover \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

该命令拒绝与活动任务并行；后台已退出时会清理保存的唯一测试容器，启动常规 SGLang并等待 ready。成功后将遗留队列标记为 recovered，允许重新投递。手动 recover 不自动唤醒 session。

## 8. 调试命令

只做前置检查，不停服：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test check \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

查看最近状态：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test status \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```

前台运行完整流程，仅用于人工调试：

```bash
bash flash-next-plan/sglang-gpu-switch-workflow/switch-test run \
  --config flash-next-plan/sglang-gpu-switch-workflow/config.local.json
```


后台 supervisor 必须运行在宿主机，具有 Docker socket 权限及 `nvidia-smi`
访问能力，并独立于常规 SGLang 容器。当前 app 若未刷新 docker 用户组，可从
具有权限的普通终端启动。常规服务恢复健康检查的模板上限为 300 秒；它没有
套用测试容器的活动 watchdog，超时会保存 DevelopmentRestoreError。
SIGKILL、断电和系统重启不能执行 finally；随后使用 recover。单纯双 fork
不提供开机自动恢复，需目标机器的进程管理器另行托管。


自动 kill 默认开启。确认要求原始累计 CPU/I/O 计数无变化，GPU 采样为 0，
日志持续安静；任何活动、指标未知或采样中断都重新计时。格式化的 Docker
CPU%/BlockIO 只作诊断。原始 I/O 不可用时不会强行补零，空闲 kill 暂不可用，
硬超时仍有效。可用 --stall-action warn 显式关闭空闲 kill；详见 RUN_ON_GPU.md。
