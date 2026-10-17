# Docker patch experiment runner

`run.py` is a Python-stdlib runner shared by patch bundles. The bundle's
`manifest.json` selects patches, container steps and the host assessment script.
No content checksums are required. Patch state is determined with forward and
reverse `git apply --check`; unrelated workspace edits can be retained.

```bash
python3 tools/patch_experiments/run.py all \
  --bundle /path/to/bundle --repo /path/to/checkout \
  --image YOUR_IMAGE --gpus device=0 \
  --mount 'type=bind,src=/models,dst=/models,readonly' \
  -v 'experiment-cache:/cache' --output /path/to/results
```

Actions: `apply`, `experiment`, `assess`, `all`, `rollback`.
Repeated `--mount`, `-v`, and `--env KEY=VALUE` pass explicit settings to Docker.
Images must already exist. Source and bundle mounts are read-only; results are
writable. The container uses the host UID/GID and overrides the image entrypoint
with the selected Python executable. Each step has a bounded timeout and its
own temporary container, removed on completion/failure. A host Docker daemon
must see the same absolute bind-mount paths as the runner.

Manifest schema version 1:

- `patches`: relative patch file paths in application order.
- `steps`: entries containing `name`, `argv` (string array), optional
  `timeout_seconds`. Commands use `{python}` as their first argument.
- `assessment`: `report` filename and `argv` for a host stdlib script that writes
  `output/result.json`; exit 0 means pass, 2 means thresholds were not met.
- `container_env`: optional environment mapping; `parameters`: optional tokens.
- `base_commit` and `source_files`: optional provenance, not checksum gates.

Tokens: `{repo}`, `{bundle}`, `{output}`, `{python}`, `{iterations}`, `{rounds}`.
Arguments do not pass through a shell; literal Python/JSON braces are preserved.
The assessment additionally receives `{report}`, `{min_speedup}`,
`{max_p95_ratio}`, `{max_scratch_mib}`. Assessment must work on the host Python.

`execution.json` saves the image ID, additional mounts and per-step commands;
logs and the assessment result are retained. A microbenchmark pass is not a
production deployment decision. The runner never edits production launch
configuration or restarts a service.
