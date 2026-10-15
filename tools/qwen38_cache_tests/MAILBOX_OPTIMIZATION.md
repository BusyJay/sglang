# Mailbox direct mmap transport

单 token mailbox decode 的 miss 直接从注册的私有 expert-pack mmap 读入 cache slot，
删除 mailbox Python payload staging；all-hit 保留无需 CPU 确认的路径。
prefill、warm-up、adaptive sampling 继续使用原普通 transport。NVFP4 专家计算、
router 权重和累加顺序保持原样，不采用分组 pipeline。

RTX 5090 D 的只读 host registration 返回 CUDA 801。本实现使用独立的
`mmap.ACCESS_COPY` 私有可写映射，并以 `cudaHostRegisterMapped` 注册；应用只读
这些源字节，原文件及 pack reader 的只读映射不变。CUDA pinning 可能物化整份
pack 的私有页，需为 pack 大小的额外主机内存预留预算。注册失败明确报错。
关闭先排空 GPU reader，再 unregister、释放 frombuffer view 并关闭私有映射；
初始化异常同样执行清理。所有初始化控制分配在 CPU 默认设备上下文中完成。

## 验证

在对应 GPU 环境中运行：

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 tools/qwen38_cache_tests/run_cpu.py
python3 tools/qwen38_cache_tests/run_gpu.py --require-sm120 --output /tmp/mailbox-gpu.json
```

GPU 回归覆盖 required tensors、resident alpha 开关、单/多 miss、重复 expert、
跨 stream、缓存迁移、sampling、失败路径，以及对齐和非对齐尾块字节。
`test_gpu_mailbox_registration.py` 额外验证 CUDA 默认设备上下文、原文件不变与
私有映射关闭。已在 RTX 5090 D / CUDA 13.0 生产镜像中运行。

真实模型测量、与三个互斥方案的对照、内存与启动成本、部署及回滚说明见
[MAILBOX_DEPLOYMENT_20261015.md](MAILBOX_DEPLOYMENT_20261015.md)。
前 512 个 decode token 的 adaptive sampling 会绕过 mailbox，短回复不能套用
稳定 mailbox 段的性能增益。
