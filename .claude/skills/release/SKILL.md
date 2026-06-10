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
- Recommended: run `/bughunt`; ensure `CHANGELOG.md` is updated.

## 2. Promote

- Promotion is **fast-forward only** (see AGENTS.md): `qa` and `main` carry no
  unique commits, so promotion is an admin fast-forward push, not a merge:
  `git push origin next:qa`, verify CI green on `qa`, then
  `git push origin qa:main`. The maintainer runs these pushes (protected
  branches); the agent prepares and verifies but does not force-push.
- If the push is rejected as non-fast-forward, STOP: something landed directly
  on `qa`/`main`. Reconcile it back into `next` first — never force-push or
  merge to work around it.

## 3. Tag

- Annotated tag on `main`: `git tag -a vX.Y.Z -m "NeuNorm vX.Y.Z"` then
  `git push origin vX.Y.Z`.
- Pre-releases: use `vX.Y.ZrcN` — CI routes these to the conda `rc` label and
  PyPI marks them as pre-releases.

## 4. Watch the pipeline

- `gh run watch` / `gh pr checks`. Confirm unit tests, conda-build,
  `Upload package to anaconda`, and `Upload release to PyPI` all succeed.

## 5. Verify artifacts

- `pip install NeuNorm==X.Y.Z` (PyPI) and `conda install neutronimaging::neunorm=X.Y.Z`.
- Import smoke: `python -c "import neunorm; print(neunorm.__version__)"`.
- Create the GitHub Release (notes from `CHANGELOG.md`; `.github/release.yml`
  filters bot PRs). Confirm Read the Docs built the new version.

Report what shipped, where, and any follow-ups.
