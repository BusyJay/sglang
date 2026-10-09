# Resume Here

Last updated: 2026-10-09
Branch: `<branch>`
Commit: `<commit>`

## Objective

<One sentence describing the current project objective.>

## Current Task

<The single task currently being implemented.>

## Next Action

<Must be identical to the first unchecked item in NEXT.md.>

## Read Before Editing

- `DECISIONS.md`: <decision IDs>
- `FILES.md`: <relevant file entries>
- `TESTS.md`: <relevant test entry>

## Latest Evidence

- Test result: `artifacts/latest-test.json`
- Detailed log paths are recorded inside `artifacts/latest-test.json`.

## Recovery Check

Before editing, run:

```bash
git status --short
git rev-parse HEAD
git diff --stat
```

If these disagree with `STATE.md`, update the workset first.
