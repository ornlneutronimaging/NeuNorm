---
name: bughunt
description: Hunt for bugs in NeuNorm's src/neunorm using parallel Claude auditors plus, when installed, an independent Codex review, then consolidate verified findings. Use before a release or when reviewing risky changes.
---

# /bughunt — NeuNorm defect hunt

Goal: surface real, *verified* defects in `src/neunorm` (or a given scope), via a
Claude-native audit plus an automatic second opinion from the Codex CLI when it
is available.

## Steps

1. **Scope.** Default to all of `src/neunorm`. If the user named modules or a
   diff range, scope to those. List the files in scope.

2. **Claude audit (always).** Launch the `bug-hunter` subagent (Agent tool,
   `subagent_type: "bug-hunter"`). For a large scope, run several in parallel —
   one per subsystem (`loaders`, `processing`, `tof`, `pipelines`,
   `exporters`+`filters`) — for breadth. Collect their structured findings.

3. **Codex second opinion (only if available).** Check `command -v codex`.
   - If present, run an independent, read-only review, e.g.:
     `codex exec "Review <files> in src/neunorm for correctness, numerical, and edge-case bugs. Output: severity | file:line | issue. Do NOT edit files."`
     (write only to /tmp if it needs scratch space).
   - If absent, state that Codex was not found and continue — never fail on this.

4. **Consolidate & verify.** Merge Claude + Codex findings, de-duplicate by
   `file:line`. Re-confirm each candidate against the actual code before keeping
   it; drop anything you cannot ground. Diverging Claude/Codex verdicts are worth
   highlighting.

5. **Report.** Verified findings grouped by severity (P0/P1/P2/nit) with
   `file:line` and a fix suggestion, plus a short "what was checked / what each
   reviewer contributed" summary.

Do **not** fix anything in this skill — fixing is a separate, explicit step.
