# Runtime switches

The authoritative runtime settings are explicit exports in
`/home/user/workspace/prod/homelab-llm/home-jay/remote-mapping/llm/sglang-fp4-local.sh`.
The copy alongside this file is a deployment snapshot. Qwen runtime settings
must not be set in the Compose YAML or defaulted from the inherited container
environment: script edits must take effect on an ordinary restart.

Edit these two lines in the authoritative script:

```bash
export SGLANG_OPT_QWEN38_B12X_PREPARED_LAUNCH="0"  # 0 / 1
export SGLANG_QWEN38_FILL_CTA="8"                # 1..8 / adaptive
```

Apply with:

```bash
docker restart homelab-llm
```

This interrupts current language requests and leaves ViT running.
Container recreation is not required. Explicit exports override even stale
values in a container created with the earlier YAML.

Inspect configured launch values without starting a model:

```bash
bash /home/user/workspace/prod/homelab-llm/home-jay/remote-mapping/llm/sglang-fp4-local.sh --print-launch-config
```

`docker exec ... printenv` shows the environment inherited by the new exec
process, not exports made inside the serving process's startup script. It is
not a reliable check of script-controlled settings.

`service_ab.py` now edits these script exports, records launcher snapshots and
uses ordinary language-container restarts. It leaves the last candidate in
the script. Historical result `override.json` files describe the old experiment;
they are retained as evidence, not current configuration instructions. The old
free-generation benchmark remains insufficient to identify optimization effect;
see `../CONTROLLED_PROTOCOL.md` before treating its differences as causal.
