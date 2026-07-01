# NeuNorm Streamlit app

An interactive front-end (a "notebook" in app form) for the NeuNorm reduction
pipelines in `neunorm.pipelines`. It lets you run any of the MARS/VENUS
normalization workflows from a browser, then inspect and download the result —
without writing a script.

The app is a *thin* UI: every reduction step is delegated to the library, so it
always reflects the installed NeuNorm version and adds no physics of its own.

## Run it

```bash
pixi run streamlit-app
# equivalent, explicit:
pixi run -e streamlit streamlit run apps/streamlit/neunorm_app.py
```

Then open the printed Local URL (default <http://localhost:8501>). On an analysis
node, forward the port (e.g. `ssh -L 8501:localhost:8501 <node>`) or pass
`--server.address`/`--server.port`.

## What it does

1. **Pick a pipeline** — MARS CCD/CMOS, MARS TPX3, VENUS CCD/CMOS,
   VENUS TPX1 (TOF), VENUS TPX3 histogram (TOF), VENUS TPX3 event (TOF).
2. **Specify input files** as server-side glob patterns. For the stack
   pipelines, **one glob per line = one acquisition run** to combine; the event
   pipeline takes a flat list of HDF5 event files. (Globs, not uploads, because
   imaging runs are thousands of files already on the cluster filesystem.)
3. **Set corrections** in the sidebar — ROI, air ROI, gamma filter, TOF/spatial
   rebinning, detector shape, and the event-mode `BinningConfig` — exposed only
   where the selected pipeline supports them.
4. **Run** the pipeline. The normalized transmission is written to HDF5 (primary)
   or TIFF at the output path you choose.
5. **Inspect** the transmission: a slice viewer (with a slider over the
   TOF/wavelength/energy or image-stack dimension), a 1σ-uncertainty toggle,
   adjustable contrast, and a spatially-averaged spectrum for TOF data.
   Then **download** the output file.

## Environment

`streamlit` is an **optional** Pixi feature/environment — it is deliberately
kept out of the core library dependencies. The Streamlit package is installed
from **PyPI** (not conda-forge) because conda-forge's `pydeck` pins
`ipywidgets<8`, which conflicts with the base `plopp` dependency
(`ipywidgets>=8.1`); the PyPI wheel carries no such pin. See
`[tool.pixi.feature.streamlit]` in `pyproject.toml`.
