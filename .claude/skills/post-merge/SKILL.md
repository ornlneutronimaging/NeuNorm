---
name: post-merge
description: Post-merge integration gate for NeuNorm — clean rebuild, full test run, and housekeeping (close addressed issues, update the task tracker / memory) after a PR lands on next. Use right after merging a PR.
---

# /post-merge — NeuNorm post-merge gate

Run after a PR merges into `next` to confirm clean integration and tidy up.

## Steps

1. **Sync.** `git switch next && git pull`.
2. **Clean rebuild from the lockfile.** `pixi install --frozen` (or `pixi install`
   if the lock changed); confirm `pixi lock --check` passes.
3. **Full suite.** `pixi run test` — must be green. If docs changed,
   `pixi run build-docs` (expect 0 warnings).
4. **Import smoke.** `python -c "import neunorm; print(neunorm.__version__)"`.
5. **Housekeeping.**
   - Close issues the merged PR addressed (reference the PR number).
   - Update the task tracker / project board.
   - If something non-obvious was learned (a constraint, a gotcha), record it.
6. **Report** integration status and anything still open.

Stop and report if any step fails — never paper over a red build.
