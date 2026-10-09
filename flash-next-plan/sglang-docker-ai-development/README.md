# Historical Docker development scaffold

This directory is retained to explain earlier design decisions. Its old
`docker-job`, `check-job`, `init-worktree`, `test-cpu` and `test-gpu-small`
entry points now exit with a pointer to the current manual. Do not install
its `qwen38_cache_tests_template` over the restored implementation.

Apply both patches using `flash-next-plan/README-ARCHIVE.md`, then run the
current `tools/qwen38_cache_tests/` wrappers. The updated
`gpu-switch.docker.example.json` is a copy of the current workflow configuration;
use `flash-next-plan/sglang-gpu-switch-workflow/README.md` to configure it.
No old `env.local` or `jobs/current.json` is required by that workflow.
