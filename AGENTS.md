# NeuNorm — AI agent guidance

Guidance for AI coding agents (Claude Code, Codex, etc.) working in this
repository. `CLAUDE.md` imports this file, so both tools share one source.

## What this is

NeuNorm 2.0 is a scipp-based library for **neutron imaging normalization and
time-of-flight (TOF) data processing** at ORNL facilities (MARS at HFIR, VENUS at
SNS). It is a complete rewrite of the 1.x `NeuNorm.normalization` API.
**HDF5 is the primary output format; TIFF is secondary.**

## Environment: Pixi only

This is a **Pixi** project (`pyproject.toml` `[tool.pixi.*]` + `pixi.lock`).

- **Never** run `pip install` to manage the environment — use Pixi.
- Run everything through Pixi: `pixi run <task>`, `pixi run python ...`.
- Add dependencies by editing `pyproject.toml` (`[tool.pixi.dependencies]` for
  conda packages, `[tool.pixi.pypi-dependencies]` for PyPI), then `pixi install`.
- CI installs with `pixi install --frozen`, pinned to a specific pixi version via
  `setup-pixi`. The pinned pixi must be able to **read** the `pixi.lock` format
  that local pixi **writes** — when regenerating the lock, use a pixi whose format
  the CI pin can read, and keep the CI `pixi-version` in step with local dev. The
  `update-lockfiles` workflow is pinned to the same pixi for this reason.

Key tasks (`[tool.pixi.tasks]`): `test`, `build-docs`, `conda-build`,
`pypi-build`, `audit-deps`, `clean*`.

## Layout

```text
src/neunorm/
  data_models/   scipp / pydantic data containers
  loaders/       TIFF, FITS, event (HDF5), metadata (NeXus) loaders
  processing/    dark/air correction, normalization, ROI, run combine, uncertainty
  tof/           binning, event/pulse reconstruction, resonance, statistics
  pipelines/     end-to-end MARS/VENUS detector pipelines
  exporters/     HDF5 (primary) + TIFF writers
  filters/       gamma-spike removal
  utils/         constants, _numba_compat (optional numba shim)
tests/unit/      pytest suite
docs/            Sphinx (autodoc) + MyST workflow guides
```

## Conventions

- Data are **scipp DataArrays** carrying variances (uncertainty tracking).
- Configuration objects are **pydantic** models; logging is **loguru**.
- `numba` is **optional**, accessed via `utils._numba_compat` (`jit`/`njit`
  degrade to no-ops when numba is absent). Do not import `numba` at module level.
- Public API uses NumPy-style docstrings (Sphinx autodoc renders them; keep
  coverage high).

## Code style & checks

- Ruff (line length 120, double quotes). Run `pixi run pre-commit run --all-files`
  before committing. Hooks: ruff (check + format), codespell, gitleaks, yamllint,
  taplo. Note: `.github/**` is excluded from pre-commit — validate workflow YAML
  manually (e.g. `python -c 'import yaml; yaml.safe_load(open(PATH))'`).

## Testing

- `pixi run test` (pytest + coverage). The suite must pass on **both `linux-64`
  and `osx-arm64`** (CI matrix; primary development is on Apple Silicon).
- Compare floating-point arrays with **`np.testing.assert_allclose`, not
  `assert_equal`** — bit-exact comparisons differ by ~1 ULP across x86_64/arm64.

## Git, branches, releases

- Promotion path: **`next` → `qa` → `main`**. `next` is the default branch;
  protected branches require PRs.
- **Promotion is fast-forward only.** `qa` and `main` never receive direct
  commits or merge commits — they are promoted by fast-forwarding to `next`
  (`git push origin next:qa`, then `git push origin qa:main`, admin push).
  This keeps all three branches on one linear history with no merge bubbles.
  If a fast-forward is rejected, something was committed directly to `qa`/`main`
  — stop and reconcile that commit back into `next` first; never force-push
  or merge to "make it fit".
- Commit attribution: end AI-assisted commits with an `Assisted-With: <model>`
  trailer. Do **not** use `Co-Authored-By:` for AI assistants. Real human
  co-authors keep their `Co-Authored-By:` trailers.
- Coordinate with the maintainers (Jean Bilheux, Chen Zhang) before large changes.
- Versioning is dynamic (**versioningit**): the release tag (`vX.Y.Z`) drives the
  version. Never hand-edit `src/neunorm/_version.py` (it is generated).
- Releases are driven by the `/release` skill; CI publishes on `v*` tags to PyPI
  (trusted publishing) and the `neutronimaging` Anaconda channel
  (`conda install neutronimaging::neunorm` — same channel as the 1.x package).
