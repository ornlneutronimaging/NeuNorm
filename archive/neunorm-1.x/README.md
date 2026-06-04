# NeuNorm 1.x (Archived)

This directory contains the archived NeuNorm 1.x codebase.

**For NeuNorm 2.0 development, see the root directory.**

## What's here

- `NeuNorm/` - Original source code
- `tests/` - Original test suite
- `notebooks/` - Tutorial notebooks
- `documentation/` - Sphinx documentation
- `environment.yml` - Conda environment (deprecated in 2.0)
- `conda.recipe/` - Conda build recipe (replaced with pixi in 2.0)

## Installing NeuNorm 1.x

If you need to use NeuNorm 1.x for legacy projects:

```bash
conda create -n neunorm1 -c conda-forge neunorm
```

Or from this archived source:

```bash
cd archive/neunorm-1.x
conda env create -f environment.yml
conda activate NeuNorm
pip install -e .
```

## Migrating to NeuNorm 2.0

See the main repository README for migration guide.
