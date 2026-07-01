"""Streamlit front-end for the NeuNorm reduction pipelines.

An interactive "notebook" that wraps the end-to-end pipelines in
:mod:`neunorm.pipelines` (the NeuNorm main library) so a beamline user can:

1. pick a detector/facility pipeline,
2. point it at sample / open-beam / dark files on the analysis filesystem,
3. set the corrections (ROI, gamma filter, TOF/spatial rebin, binning, ...),
4. run the normalization, and
5. inspect the resulting transmission (image, uncertainty, TOF spectrum) and
   download the HDF5 / TIFF output.

This is a *thin* UI: every reduction step is delegated to the library. The app
adds no physics of its own. Because neutron-imaging runs are thousands of TIFFs
living on the cluster, input is specified as server-side glob patterns rather
than browser uploads.

Run it with::

    pixi run streamlit-app
    # or directly:
    pixi run -e streamlit streamlit run apps/streamlit/neunorm_app.py
"""

from __future__ import annotations

import glob
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
import streamlit as st

from neunorm import __version__
from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.pipelines.mars_tpx3 import run_mars_tpx3_pipeline
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline


# --------------------------------------------------------------------------- #
# Pipeline registry
# --------------------------------------------------------------------------- #
@dataclass
class PipelineSpec:
    """Declarative description of one NeuNorm pipeline for the UI.

    The flags drive which input widgets are rendered; ``sample_kw`` / ``ob_kw``
    carry the (differing) keyword names each pipeline expects for its file lists.
    """

    func: Callable
    facility: str
    detector: str
    tof: bool
    sample_kw: str = "sample_paths"
    ob_kw: str = "ob_paths"
    nested: bool = True  # Sequence[Sequence[path]] (per-run lists) vs. flat Sequence[path]
    supports_dark: bool = False
    supports_air_roi: bool = False
    supports_rebin: bool = False
    supports_binning: bool = False
    supports_detector_shape: bool = False
    detector_shape_default: tuple[int, int] = (514, 514)
    notes: str = ""


PIPELINES: dict[str, PipelineSpec] = {
    "MARS CCD/CMOS": PipelineSpec(
        func=run_mars_ccd_pipeline,
        facility="HFIR",
        detector="CCD/CMOS",
        tof=False,
        supports_dark=True,
        notes="Continuous-source CCD/CMOS. Dark correction supported (optional).",
    ),
    "MARS TPX3": PipelineSpec(
        func=run_mars_tpx3_pipeline,
        facility="HFIR",
        detector="Timepix3",
        tof=False,
        supports_detector_shape=True,
        notes="Continuous-source Timepix3 histograms. Hot-pixel masking applied.",
    ),
    "VENUS CCD/CMOS": PipelineSpec(
        func=run_venus_ccd_pipeline,
        facility="SNS",
        detector="CCD/CMOS",
        tof=False,
        supports_dark=True,
        supports_air_roi=True,
        notes="Pulsed-source CCD/CMOS. Dark correction + optional air-region beam correction.",
    ),
    "VENUS TPX1 (TOF)": PipelineSpec(
        func=run_venus_tpx1_pipeline,
        facility="SNS",
        detector="Timepix1",
        tof=True,
        sample_kw="sample_tiff_paths",
        ob_kw="ob_tiff_paths",
        supports_air_roi=True,
        supports_rebin=True,
        notes="Pulsed-source Timepix1 TOF stacks (per-shutter TIFFs).",
    ),
    "VENUS TPX3 histogram (TOF)": PipelineSpec(
        func=run_venus_tpx3_histogram_pipeline,
        facility="SNS",
        detector="Timepix3",
        tof=True,
        sample_kw="sample_tiff_paths",
        ob_kw="ob_tiff_paths",
        supports_air_roi=True,
        supports_rebin=True,
        notes="Pulsed-source Timepix3 pre-histogrammed TOF stacks.",
    ),
    "VENUS TPX3 event (TOF)": PipelineSpec(
        func=run_venus_tpx3_event_pipeline,
        facility="SNS",
        detector="Timepix3",
        tof=True,
        nested=False,  # flat list of event HDF5 files
        supports_air_roi=True,
        supports_rebin=True,
        supports_binning=True,
        supports_detector_shape=True,
        notes="Pulsed-source Timepix3 event mode (HDF5). Requires a BinningConfig.",
    ),
}


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedInput:
    """Result of expanding one multi-line glob spec into concrete file lists."""

    runs: list[list[str]] = field(default_factory=list)  # one inner list per non-empty line
    warnings: list[str] = field(default_factory=list)

    @property
    def flat(self) -> list[str]:
        return [p for run in self.runs for p in run]

    @property
    def n_files(self) -> int:
        return sum(len(run) for run in self.runs)


def resolve_globs(text: str) -> ResolvedInput:
    """Expand a multi-line glob spec.

    Each non-empty line is treated as one acquisition *run*: its glob is
    expanded and sorted, producing one inner list. Lines that match nothing are
    reported as warnings (and contribute no run).
    """
    out = ResolvedInput()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        matches = sorted(glob.glob(line))
        if not matches:
            out.warnings.append(f"No files matched: `{line}`")
            continue
        out.runs.append(matches)
    return out


def parse_roi(text: str) -> Optional[tuple[int, int, int, int]]:
    """Parse ``"x0, y0, x1, y1"`` into an int tuple, or ``None`` if blank."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError("ROI needs exactly 4 integers: x0, y0, x1, y1")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def spectral_dim_of(da: sc.DataArray) -> Optional[str]:
    """Return the non-spatial dimension (tof/wavelength/energy), if any."""
    extra = [d for d in da.dims if d not in ("x", "y")]
    return extra[0] if extra else None


def slice_image(da: sc.DataArray, spectral_dim: Optional[str], idx: int, values: bool = True) -> np.ndarray:
    """Return a 2D ``[y, x]`` numpy array for display (data values or stderr)."""
    da2 = da[spectral_dim, idx] if spectral_dim is not None else da
    da2 = da2.transpose(["y", "x"])
    if values:
        return np.asarray(da2.values, dtype=float)
    var = da2.variances
    if var is None:
        return np.zeros(da2.shape)
    return np.sqrt(np.asarray(var, dtype=float))


def spectrum_over(da: sc.DataArray, spectral_dim: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean transmission vs. the spectral coordinate (averaged over x and y)."""
    arr = np.asarray(da.transpose([spectral_dim, "y", "x"]).values, dtype=float)
    spectrum = np.nanmean(arr, axis=(1, 2))
    if spectral_dim in da.coords:
        coord = np.asarray(da.coords[spectral_dim].values, dtype=float)
        if coord.size == spectrum.size + 1:  # bin edges -> centers
            coord = 0.5 * (coord[:-1] + coord[1:])
    else:
        coord = np.arange(spectrum.size, dtype=float)
    return coord, spectrum


def render_results(da: sc.DataArray) -> None:
    st.subheader("Transmission")
    spectral = spectral_dim_of(da)
    st.caption(f"dims `{da.dims}` · shape `{tuple(da.shape)}` · unit `{da.unit}`")

    idx = 0
    if spectral is not None:
        n = da.sizes[spectral]
        idx = st.slider(f"{spectral} index", 0, n - 1, n // 2)
        if spectral in da.coords:
            try:
                coord = da.coords[spectral].values
                st.caption(f"{spectral} ≈ {float(coord[idx]):.4g} {da.coords[spectral].unit}")
            except (IndexError, TypeError, ValueError):
                pass

    show_unc = st.checkbox("Show uncertainty (1σ) instead of transmission", value=False)
    img = slice_image(da, spectral, idx, values=not show_unc)

    finite = img[np.isfinite(img)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [1, 99])
    else:
        lo, hi = 0.0, 1.0
    col1, col2 = st.columns(2)
    vmin = col1.number_input("vmin", value=float(lo), format="%.4g")
    vmax = col2.number_input("vmax", value=float(hi), format="%.4g")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("1σ uncertainty" if show_unc else "Transmission")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)

    if spectral is not None:
        st.markdown("**Spatially-averaged spectrum**")
        coord, spectrum = spectrum_over(da, spectral)
        figs, axs = plt.subplots(figsize=(6, 3))
        axs.plot(coord, spectrum, lw=1)
        axs.axvline(coord[idx], color="tab:red", lw=1, ls="--", label="current slice")
        axs.set_xlabel(f"{spectral} [{da.coords[spectral].unit if spectral in da.coords else 'index'}]")
        axs.set_ylabel("mean transmission")
        axs.legend(loc="best", fontsize="small")
        st.pyplot(figs)
        plt.close(figs)


# --------------------------------------------------------------------------- #
# Sidebar / form
# --------------------------------------------------------------------------- #
def build_binning_form() -> BinningConfig:
    st.markdown("**Binning (event mode)**")
    bins = st.number_input("bins", min_value=1, value=5000, step=100)
    bin_space = st.selectbox("bin_space", ["energy", "wavelength", "tof"], index=0)
    use_log_bin = st.checkbox("logarithmic spacing", value=True)

    kwargs: dict = {"bins": int(bins), "bin_space": bin_space, "use_log_bin": use_log_bin}
    if bin_space == "energy":
        c1, c2 = st.columns(2)
        emin = c1.number_input("E_min [eV]", value=1.0, format="%.6g")
        emax = c2.number_input("E_max [eV]", value=100.0, format="%.6g")
        kwargs["energy_range"] = (float(emin), float(emax))
    elif bin_space == "wavelength":
        c1, c2 = st.columns(2)
        wmin = c1.number_input("λ_min [Å]", value=0.5, format="%.6g")
        wmax = c2.number_input("λ_max [Å]", value=6.0, format="%.6g")
        kwargs["wavelength_range"] = (float(wmin), float(wmax))
    else:  # tof
        use_range = st.checkbox("limit TOF range", value=False)
        if use_range:
            c1, c2 = st.columns(2)
            tmin = c1.number_input("t_min [ns]", value=0.0, format="%.6g")
            tmax = c2.number_input("t_max [ns]", value=16_667_000.0, format="%.6g")
            kwargs["tof_range"] = (float(tmin), float(tmax))
    return BinningConfig(**kwargs)


def main() -> None:  # noqa: C901  (UI assembly is intentionally one linear flow)
    st.set_page_config(page_title="NeuNorm", page_icon="🔬", layout="wide")
    st.title("🔬 NeuNorm — Neutron Imaging Normalization")
    st.caption(f"Interactive front-end for the `neunorm.pipelines` library · NeuNorm {__version__}")

    # ---- Sidebar: pipeline + options -------------------------------------- #
    with st.sidebar:
        st.header("Pipeline")
        name = st.selectbox("Detector / facility", list(PIPELINES))
        spec = PIPELINES[name]
        st.info(f"**{spec.facility} · {spec.detector}**\n\n{spec.notes}")

        st.header("Corrections")
        gamma_filter = True
        if not spec.tof:  # gamma_filter is a CCD/TPX3-histogram (non-TOF) pipeline knob
            gamma_filter = st.checkbox("Gamma-spike filter", value=True)

        roi_text = st.text_input("ROI  (x0, y0, x1, y1)", value="", help="Blank = full detector")
        air_roi_text = ""
        if spec.supports_air_roi:
            air_roi_text = st.text_input(
                "Air ROI  (x0, y0, x1, y1)", value="", help="Open-beam air region for beam correction (blank = off)"
            )

        rebin_by_tof: bool | int = False
        rebin_by_spatial: Optional[int] = None
        if spec.supports_rebin:
            st.subheader("Rebinning")
            tof_factor = st.number_input("TOF rebin factor (0 = off, 1 = auto)", min_value=0, value=0, step=1)
            if tof_factor == 1:
                rebin_by_tof = True
            elif tof_factor > 1:
                rebin_by_tof = int(tof_factor)
            spatial_factor = st.number_input("Spatial rebin factor (0 = off)", min_value=0, value=0, step=1)
            if spatial_factor > 0:
                rebin_by_spatial = int(spatial_factor)

        detector_shape = spec.detector_shape_default
        if spec.supports_detector_shape:
            c1, c2 = st.columns(2)
            h = c1.number_input("detector rows", min_value=1, value=spec.detector_shape_default[0])
            w = c2.number_input("detector cols", min_value=1, value=spec.detector_shape_default[1])
            detector_shape = (int(h), int(w))

        binning_cfg: Optional[BinningConfig] = None
        if spec.supports_binning:
            st.subheader("TOF binning")
            binning_cfg = build_binning_form()

        st.header("Output")
        out_fmt = st.selectbox("Format", ["hdf5", "tiff"], index=0)
        default_out = f"neunorm_{spec.facility.lower()}_{spec.detector.split('/')[0].lower()}.{out_fmt}"
        out_path = st.text_input("Output path", value=str(Path.cwd() / default_out))

    # ---- Main: inputs ----------------------------------------------------- #
    st.subheader("Input files")
    if spec.nested:
        st.caption("One **glob pattern per line** — each line is one acquisition run to combine.")
    else:
        st.caption("Glob pattern(s) for the event HDF5 files (all matches are combined).")

    col_s, col_o = st.columns(2)
    sample_text = col_s.text_area("Sample files", height=120, placeholder="/path/to/sample/*.tiff")
    ob_text = col_o.text_area("Open-beam (OB) files", height=120, placeholder="/path/to/ob/*.tiff")

    dark_text = ""
    if spec.supports_dark:
        dark_text = st.text_area(
            "Dark files (optional)", height=80, placeholder="/path/to/dark/*.tiff — blank to skip dark correction"
        )

    sample = resolve_globs(sample_text)
    ob = resolve_globs(ob_text)
    dark = resolve_globs(dark_text) if dark_text.strip() else None

    cols = st.columns(3)
    cols[0].metric("Sample files", sample.n_files, f"{len(sample.runs)} run(s)")
    cols[1].metric("OB files", ob.n_files, f"{len(ob.runs)} run(s)")
    if spec.supports_dark:
        cols[2].metric("Dark files", dark.n_files if dark else 0, f"{len(dark.runs) if dark else 0} run(s)")

    for w in [*sample.warnings, *ob.warnings, *(dark.warnings if dark else [])]:
        st.warning(w)

    run = st.button("▶ Run pipeline", type="primary", disabled=not (sample.n_files and ob.n_files))

    # ---- Run -------------------------------------------------------------- #
    if run:
        try:
            roi = parse_roi(roi_text)
            air_roi = parse_roi(air_roi_text) if spec.supports_air_roi else None
        except ValueError as exc:
            st.error(f"ROI error: {exc}")
            st.stop()

        kwargs: dict = {"output_path": Path(out_path)}
        kwargs[spec.sample_kw] = sample.flat if not spec.nested else sample.runs
        kwargs[spec.ob_kw] = ob.flat if not spec.nested else ob.runs
        if roi is not None:
            kwargs["roi"] = roi
        if not spec.tof:
            kwargs["gamma_filter"] = gamma_filter
        if spec.supports_dark and dark is not None:
            kwargs["dark_paths"] = dark.runs
        if spec.supports_air_roi and air_roi is not None:
            kwargs["air_roi"] = air_roi
        if spec.supports_rebin:
            kwargs["rebin_by_tof"] = rebin_by_tof
            kwargs["rebin_by_spatial"] = rebin_by_spatial
        if spec.supports_detector_shape:
            kwargs["detector_shape"] = detector_shape
        if spec.supports_binning:
            kwargs["binning"] = binning_cfg

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with st.status("Running NeuNorm pipeline…", expanded=True) as status:
            st.write(f"Pipeline: **{name}**  ·  started {datetime.now():%H:%M:%S}")
            try:
                transmission = spec.func(**kwargs)
            except Exception as exc:  # noqa: BLE001 — surface any library error to the UI
                status.update(label="Pipeline failed", state="error")
                st.exception(exc)
                st.stop()
            status.update(label=f"Done → {out_path}", state="complete")
        st.session_state["result"] = transmission
        st.session_state["result_path"] = out_path
        st.success(f"Wrote normalized transmission to `{out_path}`")

    # ---- Results ---------------------------------------------------------- #
    if "result" in st.session_state:
        st.divider()
        render_results(st.session_state["result"])

        out_file = Path(st.session_state.get("result_path", ""))
        if out_file.is_file():
            with open(out_file, "rb") as fh:
                data = fh.read()
            st.download_button(
                f"⬇ Download {out_file.name}  ({len(data) / 1e6:.1f} MB)",
                data=io.BytesIO(data),
                file_name=out_file.name,
                mime="application/octet-stream",
            )


if __name__ == "__main__":
    main()
