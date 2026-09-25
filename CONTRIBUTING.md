# Contributing to NeuNorm

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, NeuNorm version and python version. Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

## Contributing code

Thanks for your interest in contributing code to NeuNorm!

We are currently working on new features such as:
 - use of a mask to block pixels and not allow normalization of those pixels
 - speed up loading of data by working in parallel

For any contribution you would like to add, please fork the NeuNorm project and use the pull request to bring them back to master.

## Building the documentation

Build the docs with `pixi run -e docs build-docs`. Warnings are errors (`-W`) here,
in CI and on Read the Docs.

The build links to the Python, NumPy, SciPy and scipp docs through intersphinx. It
fetches each project's `objects.inv` from the live site first and falls back to a
copy committed under `docs/_inventory/`, so an outage of one of those sites does
not fail the build. Refresh the committed copies now and then, and whenever a link
to one of those projects stops resolving, with:

```bash
pixi run update-inventories
```

Then commit the updated `docs/_inventory/*.inv` files. If you add or change an
intersphinx mapping in `docs/conf.py`, update the `update-inventories` task in
`pyproject.toml` to match.
