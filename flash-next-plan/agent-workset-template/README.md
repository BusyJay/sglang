# Agent Workset

This directory is the persistent project memory for AI-assisted development.

After any context compact, read files in this order:

1. `RESUME.md`
2. `COMPACT.md`
3. `STATE.md`
4. `DECISIONS.md`
5. `NEXT.md`
6. `FILES.md`
7. `TESTS.md`
8. `ENV.md` when environment details are needed
9. `artifacts/latest-test.json` when a test has just completed

Rules:

- Verify the actual Git worktree before editing.
- `STATE.md` is the source of truth for current facts.
- `DECISIONS.md` is the source of truth for accepted design decisions.
- `NEXT.md` is the only short-term task queue.
- `RESUME.md` stays short and points to the exact next action.
- `COMPACT.md` is overwritten before each expected compact.
- If the compact summary conflicts with this workset, inspect the worktree and update the workset before continuing.

