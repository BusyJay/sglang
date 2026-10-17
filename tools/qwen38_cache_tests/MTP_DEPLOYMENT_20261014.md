# 线上 MTP 切换记录（2026-10-14）

用户授权将线上部署切为 MTP，并确认暂时只处理文本。
修改实际启动脚本：
`/home/user/workspace/prod/homelab-llm/home-jay/remote-mapping/llm/sglang-flash-next.sh`。
通过 `/home/user/workspace/prod/homelab-llm/docker-compose.yml` 的 `llm` 服务启动，
容器 `qwen3.8-nvfp4-llm`，端口 `127.0.0.1:8329`；其他服务未重建。

## 生效配置

- `CACHE_SLOTS_TOTAL=5040` 对应 loader 的 `cache_slots_total`，替代旧的
  `CACHE_SLOTS=105` / `cache_slots_per_layer`。48 层初始均分为每层 105，
  adaptive 按总预算调整。脚本遇到旧 `CACHE_SLOTS` 环境变量明确报错。
- NEXTN 同卡 MTP，draft load format `qwen38_mtp`，steps=2、top-k=1、
  draft tokens=3、BF16 draft KV、BF16 GDN。
  实际 scheduler 将 NEXTN 规范化为 EAGLE，这是该实现的正常别名。
- `mem-fraction-static=0.92`，单请求，prefill chunk=1024，
  breakable decode graph 的 bs/max-bs 均为 1。
- 保留 `PLE_OFFLOAD_BACKEND=lazy`、3 GiB 主机 PLE row cache、
  adaptive cache、packed prefill/decode、projection 预分配及 HiCache ratio=6。
- 使用 `--language-model-only`，移除语言 worker 的 EPD/encoder 接入。
  ViT 容器仍运行；当前 LLM 不处理图片。

原脚本的 context 设置仍为 220000；实际分配 KV 为 **82496 token**，
接口公布 `max_req_input_len=82490`，所以不能把模型 context 设置视为单次
请求可用容量。尚未测试该实际容量的长上下文边界。

## 验证

通过线上接口读取 `/server_info` 检查上述配置及总槽位键；`/ready` 成功。
OpenAI chat 接口在关闭 thinking 的算术请求中返回 `323`。
原吞吐对照的 512 / 4096 token 输入分别连续输出 256 token，实际 verify
轮数为 127 / 109，draft proposed 计数均大于零，HTTP 200，进程正常运行。
加载、target verify、draft decode 和 draft extend graph 捕获、启动 warmup
均完成。这里是线上功能验收，未重新完成 lazy 的 target/MTP 吞吐对照；
此前 pinned 测量的增益不可直接作为 lazy 部署的性能结论。

曾短暂尝试 pinned 启动，核对源码后恢复 lazy：lazy 的 3 GiB cache 是
CPU pinned 内存，并非 GPU 缓存；pinned 则装载完整 PLE 表到锁页主机内存。
两者都 offload PLE，启用 MTP 不要求切换 pinned。

原脚本备份、最终脚本、验证请求/响应、配置和日志保存在本地
`results/mtp-deployment-20261014/`。回滚时恢复 `sglang-flash-next.before.sh`
到上述实际启动路径，再仅重建 compose 的 `llm` 服务。
