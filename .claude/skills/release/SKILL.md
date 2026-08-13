---
name: release
description: Cut a NeuNorm release — pre-flight checks, promote next->qa->main, tag the version, watch the publish pipeline, and verify published artifacts. Use when shipping a new NeuNorm version to PyPI + conda.
---

# /release — NeuNorm release

Drives a NeuNorm release end to end. NeuNorm uses dynamic versioning
(**versioningit**): the **git tag** `vX.Y.Z` is the single source of the version,
and CI (`.github/workflows/test_and_deploy.yaml`) publishes on `v*` tags to PyPI
(trusted publishing) and the `neutronimaging` Anaconda channel. Promotion path:
`next -> qa -> main`.

Ask the user for the target version (e.g. `2.0.0`) if not provided, then:

## 1. Pre-flight (do not skip)

- Working tree clean and synced with the intended source branch.
- `pixi run test` green locally; `pixi lock --check` passes.
- Version sanity — confirm an annotated `vX.Y.Z` tag resolves to exactly
  `X.Y.Z` (versioningit's `next-version = minor` computes a dev version off the
  *previous* tag, so only the explicit tag yields the intended release number).
- Prerequisites (ask the user to confirm — do NOT assume):
  - PyPI **trusted publisher** configured for `ornlneutronimaging/NeuNorm` +
    workflow `test_and_deploy.yaml`.
  - `ANACONDA_TOKEN` repository secret present. (No `CODECOV_TOKEN` — Codecov
    was dropped in #138; see #139.)
- **Run a pre-release audit — this pays for itself.** The 2.3.0 audit found three
  release-blocking defects that all of CI, green tests and per-PR review had
  missed, including a conda recipe that had been shipping unimportable pipelines
  since 2.0. Audit the whole `<last-tag>..HEAD` surface, not just the newest PR,
  and include these dimensions explicitly — they are where the 2.3.0 blockers
  actually were, and none is a "bug" a bug-hunter looks for:
  - **CHANGELOG completeness**: map every commit in the range to an entry. A
    user-facing change with no entry ships undocumented.
  - **CHANGELOG accuracy**: verify each claim against the code by *executing* it.
    Claims of the form "unchanged, bit-for-bit" / "bit-identical" are the ones
    that turn out false, and a downstream reads exactly those to decide whether
    to re-validate.
  - **Packaging**: the conda run-dependencies versus what the code actually
    imports at module level.
  - **Docs examples**: run them; a copied example that raises is worse than a
    missing one.
- Anything landing on `next` during release prep is a normal change and goes
  through a **PR**, not a direct push — only the CHANGELOG stamp itself is a
  direct commit (see step 3).

## 2. Check out all three branches locally FIRST (mandatory)

**Before any push, `next`, `qa` and `main` must all exist as local branches and
all be up to date with their remotes.** Do this even though only `next` is
normally worked on.

```bash
git fetch origin --tags --prune
for b in next qa main; do
  git checkout "$b" && git pull --ff-only origin "$b"
done
git checkout next
```

Then print the state and confirm it before pushing anything:

```bash
git log --oneline -1 next; git log --oneline -1 qa; git log --oneline -1 main
```

Do **not** substitute a raw-SHA refspec (`git push origin <sha>:refs/heads/qa`)
for a missing local branch. That form is valid git and does work, but it hides
which commit is actually moving and it reads to the maintainer like an invented
workaround. Create the local branch instead. The only acceptable reason to name
a SHA is when the maintainer explicitly asks for it.

**Why this is mandatory.** During the 2.3.0 release the promotion pushed a stale
`qa`: the maintainer's local `next` was still several commits behind (the release
PR had been merged in the GitHub web UI, so the local ref never advanced), and
`git push origin next:qa` faithfully pushed that stale local `next`. `qa` landed
3 commits behind `main`, which had been pushed by explicit SHA and was therefore
correct. The branches disagreed, and nobody noticed until after the tag. A
`git push` of `<local-branch>:<remote-branch>` publishes **the local ref**, never
the remote one — so a stale local branch silently promotes the wrong commit.

## 3. Stamp the CHANGELOG on `next`

- Convert the accumulated notes into a dated release section: keep an empty
  `## [Unreleased]` heading and insert `## [X.Y.Z] - YYYY-MM-DD` directly beneath
  it, so the existing entries fall under the new version. Add the matching
  `[X.Y.Z]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/vX.Y.Z`
  link reference at the bottom — this gets forgotten (the 2.2.3 stamp omitted it
  and 2.3.0 had to restore it), so check the tail of the file.
- Commit message: `docs: stamp CHANGELOG [X.Y.Z] - YYYY-MM-DD for release`.
- This one commit is a **direct commit on `next`**, matching every prior stamp —
  it touches nothing but CHANGELOG date lines, so there is nothing to review.
  Anything else must have gone through a PR first (step 1).
- Use the date the release actually goes out. If a fix PR delays things, re-stamp
  rather than carrying a stale date.

## 4. Promote

- Promotion is **fast-forward only** (see AGENTS.md): `qa` and `main` carry no
  unique commits, so promotion is an admin fast-forward push, not a merge.
- The maintainer runs these pushes (protected branches); the agent prepares and
  verifies but does not force-push.

```bash
git push origin next:qa      # then WAIT for CI green on qa
git push origin qa:main
```

- After **each** push, re-verify that the remote actually moved where intended,
  rather than assuming the command implied it:

```bash
git fetch origin
git rev-parse origin/next origin/qa origin/main   # all three must match
```

- If the push is rejected as non-fast-forward, STOP: something landed directly
  on `qa`/`main`. Reconcile it back into `next` first — never force-push or
  merge to work around it.
- If a branch is merely **behind** (an ancestor of the release commit, no
  divergence), that is not a reconcile — just fast-forward it again. Confirm
  which case you are in with
  `git merge-base --is-ancestor origin/qa <release-sha>`.

## 5. Tag

- Tag only once all three remotes point at the same commit (the check above).
- Annotated tag on `main`: `git tag -a vX.Y.Z -m "NeuNorm vX.Y.Z"` then
  `git push origin vX.Y.Z`.
- Pre-releases: use `vX.Y.ZrcN` — CI routes these to the conda `rc` label and
  PyPI marks them as pre-releases. Note the conda label is chosen by
  `contains(github.ref, 'rc')`, so any tag containing "rc" anywhere routes to the
  `rc` label.

## 6. Watch the pipeline

- `gh run watch` / `gh pr checks`. Confirm unit tests, conda-build,
  `Upload package to anaconda`, and `Upload release to PyPI` all succeed.
- A green `Upload release to PyPI` on a **non-tag** ref means nothing was
  published: the job always runs and builds, but its upload step is gated on
  `startsWith(github.ref, 'refs/tags/v')`. Check the tag run, not a branch run.
- Confirm the conda label from the job log: it should print
  `pushing refs/tags/vX.Y.Z with label main` (`rc` only for pre-releases).

## 7. Verify artifacts

- **Use a Python >= 3.11 interpreter to test the PyPI install.** `requires-python
  = ">=3.11"` means an older interpreter makes pip report only the 1.x versions
  and say the release "does not exist" — a false alarm that looks exactly like a
  failed publish. macOS system `python3` is 3.9, so
  `python3 -m venv` produces a misleading result. Build the venv from the pixi
  env's interpreter instead.

```bash
pixi run python -m venv /tmp/relcheck
/tmp/relcheck/bin/pip install "NeuNorm==X.Y.Z"
/tmp/relcheck/bin/python -c "import neunorm; print(neunorm.__version__)"
```

- Verify the **conda** package separately, in one shot:

```bash
pixi exec --spec "neunorm=X.Y.Z" --spec python=3.12 \
  --channel neutronimaging --channel conda-forge \
  python -c "import neunorm; print(neunorm.__version__)"
```

- Import smoke must import a **pipeline**, not just `neunorm`.
  `src/neunorm/__init__.py` imports nothing but `_version`, so
  `import neunorm` succeeds even when the conda recipe is missing a real runtime
  dependency — which is exactly how a missing `scitiff` shipped in every 2.x
  conda package up to 2.2.3:

```bash
python -c "import neunorm.pipelines.mars_ccd, neunorm.pipelines.venus_tpx1"
```

- Create the GitHub Release (notes from `CHANGELOG.md`; `.github/release.yml`
  filters bot PRs). Confirm Read the Docs built the new version — RTD reads
  `.readthedocs.yaml` from each version's own ref, so `main`/`stable` only heal
  once the release commit reaches them.

Report what shipped, where, and any follow-ups.
