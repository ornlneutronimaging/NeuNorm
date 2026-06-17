---
name: review-pipeline
description: Detection-only two-LLM-family review of the current branch's diff vs next (or a named branch/scope). Runs Claude + Codex finders through the dual-family-review engine, cross-family verifies P0/P1 findings, and reports at a human gate. Use to review a PR/branch before pushing or merging. Never edits, commits, or pushes.
user-invokable: true
---

# /review-pipeline — NeuNorm diff-scoped review

Run a **diff-scoped, two-LLM-family** review of a branch and report verified
findings at a human gate. This is **detection-only**: it finds, cross-family
verifies, and reports — it never edits files, commits, or pushes. Fixing is a
separate, explicit step you drive after the gate.

**When to use which audit skill:**

- `/review-pipeline` (this) — review the **changes** on a branch/PR (`git diff
  next...HEAD`). The right tool before pushing or merging a feature branch.
- `/bughunt` / `bughunt-deep` — audit the **whole** `src/neunorm` codebase
  (release gate), not a diff.

## Architecture

The non-interactive heart — **Claude finder + Codex finder per target →
cross-LLM-family adversarial verify → structured findings** — is the
`dual-family-review` **Workflow** (`.claude/workflows/dual-family-review.js`).
This skill is the **orchestrator**: it builds the preflight + target, invokes the
engine once, and renders the result at a STOP gate. Invoking the engine here is
the sanctioned Workflow opt-in (a skill whose instructions call the Workflow tool).

Why the cross-family pass matters: a parent agent re-deriving a subagent's claim
is **not** independent confirmation (same LLM family). A finding is VERIFIED only
when a *second* family (Codex verifying Claude, or vice-versa) independently
confirms it. Codex is supplementary — if it is absent, the round is single-family
and every P0/P1 comes back NEEDS-VERIFICATION (it does not silently pass).

## Arguments

- **No args** — review `git diff next...HEAD` (the current branch vs `next`).
- **A branch name** — review `git diff next...<branch>`.
- **`--base <ref>`** — diff against `<ref>` instead of `next`.
- **`--skip-codex`** — Claude-only (single-family; P0/P1 → NEEDS-VERIFICATION).
- **A path / glob** — scope the audit to those files instead of a diff.

---

## Step 1: Preflight (read-only git)

Gather, with read-only git from the repo root:

- `repoRoot` — `git rev-parse --show-toplevel`
- `headSha` — `git rev-parse --short HEAD`
- `isWorktree` — true if `git rev-parse --git-common-dir` differs from `--git-dir`
- `codexAvailable` / `codexVersion` — `command -v codex && codex --version`
  (false + empty if absent; do NOT fail if missing)
- base ref — `next` unless `--base` was given
- **changed files** — `git diff --name-only <base>...HEAD` (three-dot merge-base
  diff), then keep the Python files under `src/neunorm/` (and any changed
  `tests/` files, which the finder may read for circular-validation checks).
- **diverged?** — `git rev-list --count <base>..HEAD`; if `0` and no explicit
  path scope, report "no changes vs <base> — nothing to review" and STOP.

If a branch name was passed, use `<branch>` in place of `HEAD` above and read
changed files via `git show <branch>:<path>` (the branch may not be checked out).

## Step 2: Build the target and invoke the engine

Build **one** target from the changed source files:

- `key`: a short slug of the branch/scope (e.g. `diff-<headSha>`).
- `name`: `"changes on <branch> vs <base>"`.
- `paths`: the deduped **package dirs** of the changed `src/neunorm/` files
  (e.g. `src/neunorm/pipelines/`, `src/neunorm/processing/`). If a path scope was
  given instead of a diff, use those paths.
- `blurb`: `"the diff <base>...HEAD (N files in: <packages>)"`.
- `p0`: `"A correctness regression introduced by these changes: a logic error, an
  exception on valid input, a wrong normalization/TOF result, an uncertainty
  (variance) error that mis-states the reported error, data corruption in HDF5/TIFF
  output, or a silently-masked whole-array error."`
- `lookFor`: `[]` (the engine appends its NeuNorm CORE_CHECKLIST).
- `scopeDirective`: `"Audit ONLY the changes in 'git diff <base>...HEAD'. Run that
  diff, then read each changed file IN FULL. Report findings only for code
  introduced or altered by this diff, plus any pre-existing call site the diff
  newly breaks."` (For a path scope, audit those files in full instead.)

Then call the **Workflow** tool:

```
Workflow(name="dual-family-review", args={
  pf: { repoRoot, codexAvailable, codexVersion, headSha, isWorktree },
  targets: [ <the target above> ],
  config: {
    contextNote: "Per-PR review of branch <branch> vs <base>.",
    roundNote: "review",
    skipCodex: <true if --skip-codex>,
    hardEnforce: false
  }
})
```

It runs in the background and returns:

```
{ perTarget: [{ domainKey, domainName, claudeAssessment, codexAssessment,
               tierCounts:{claude,codex}, verified[], needsVerification[],
               refuted[], p2s[], circular[] }],
  droppedTargets, codexUsable }
```

**Fail closed:** if `perTarget` is missing/empty for a non-empty target list, or
`droppedTargets > 0`, report that the engine under-ran — do NOT present a false
all-clear.

Each finding carries `tier` (P0/P1/P2), `file`, `line`, `claim`, `reasoning`,
`suggestedFix`, `confidence`, and a `status` (VERIFIED = a 2nd family confirmed;
NEEDS-VERIFICATION = single-family / split / codex unavailable; REFUTED =
cross-family refuted). VERIFIED findings may carry a `verifierTier` (the verifier
downgraded the original tier).

## Step 3: Consolidation gate

Render the returned payload as a report and **STOP** for the user:

1. **Header**: branch, base, headSha, and whether codex was usable
   (`codexUsable`). If codex was disabled/absent, say so and note that P0/P1
   findings are NEEDS-VERIFICATION, not cross-confirmed.
2. **Counts**: VERIFIED P0 / VERIFIED P1 / NEEDS-VERIFICATION / REFUTED / P2,
   plus Claude vs Codex `tierCounts`.
3. **Findings detail**, grouped by confidence, each with `file:line`, claim,
   reasoning, and `suggestedFix`:
   - **VERIFIED** (cross-confirmed) — highest confidence; fix candidates.
   - **NEEDS-VERIFICATION** (single-family) — call out explicitly that these did
     NOT meet the ≥2-family bar.
   - **REFUTED** — list with the refuter's reasoning so they are not re-litigated.
   - **P2** — trivial; list briefly.
4. **Circular-validation risks**: surface every non-empty `circular[]` entry
   (its `note`, `file:line`, family) — a test whose oracle may mirror buggy
   behaviour is a HIGH-PRIORITY signal; never drop it silently.
5. **Suggested disposition** (you propose; the **user decides**): fix-now for
   VERIFIED P0/P1, consider for NEEDS-VERIFICATION, dismiss for REFUTED.

**MANDATORY GATE — present the report and STOP. Detection-only: do NOT fix,
commit, or push.** If the user then asks for fixes, apply them as a normal,
separate editing task (run `pixi run test` and `pixi run pre-commit run
--all-files` after), and re-run `/review-pipeline` to confirm.

## Notes

- The engine is **read-only** and enforces a DETECTION-ONLY contract on every
  subagent; it writes only `/tmp` scratch files for the Codex harness.
- Codex must be able to authenticate non-interactively. If it consistently fails,
  check `codex --version` / the codex login; do not work around with model
  overrides — codex is supplementary, not blocking.
- Compare floating-point expectations with tolerances, never bit-exact (per
  AGENTS.md): a finding that demands bit-exact equality across arches is itself
  likely a false positive — weigh that at the gate.
