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
living on the cluster, files are picked with a native OS multi-file dialog on
the machine hosting the server (see :func:`file_input`) rather than uploaded
through the browser.

Run it with::

    pixi run streamlit-app
    # or directly:
    pixi run -e streamlit streamlit run apps/streamlit/neunorm_app.py
"""

from __future__ import annotations

import glob
import io
import multiprocessing
import os
import pprint
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


def output_basename(runs: list[list[str]]) -> Optional[str]:
    """Derive an output base name from selected input runs.

    ``runs`` is the ``list[list[str]]`` produced by :func:`file_input` (one inner
    list per acquisition run). Returns the stem of the first selected file, or
    ``None`` when nothing has been picked yet.
    """
    for run in runs or []:
        for path in run:
            return Path(path).stem
    return None


# --------------------------------------------------------------------------- #
# Native file dialog (replaces free-text glob entry)
# --------------------------------------------------------------------------- #
def _file_dialog_worker(queue, title: str, initialdir: Optional[str]) -> None:
    """Open a native multi-file picker; push the chosen paths onto ``queue``.

    Runs in a child process so tkinter owns its own main thread (Streamlit runs
    the script in a worker thread, where tkinter is unreliable). Any failure —
    most commonly no display on the server — is returned as the exception.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        kwargs = {"title": title}
        if initialdir and Path(initialdir).is_dir():
            kwargs["initialdir"] = initialdir
        files = filedialog.askopenfilenames(**kwargs)
        root.update()
        root.destroy()
        queue.put(list(files))
    except Exception as exc:  # noqa: BLE001 — surface any GUI/display error to the caller
        queue.put(exc)


def pick_files(title: str = "Select files", initialdir: Optional[str] = None) -> list[str]:
    """Open a native OS multi-file dialog and return the chosen paths (sorted).

    ``initialdir`` sets the folder the dialog opens in (ignored if it does not
    exist). The dialog appears on the machine hosting the Streamlit server;
    raises ``RuntimeError`` when no display is available (headless / remote).
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_file_dialog_worker, args=(queue, title, initialdir), daemon=True)
    proc.start()
    proc.join()
    result = queue.get() if not queue.empty() else []
    if isinstance(result, Exception):
        raise RuntimeError(f"File dialog unavailable ({result}). No display on the server?")
    return sorted(result)


def file_input(
    label: str, state_key: str, *, nested: bool, help_text: str = "", initialdir: Optional[str] = None
) -> ResolvedInput:
    """Render a native-dialog file input and return the current selection.

    Each **Browse** click opens a multi-file picker (starting in ``initialdir``);
    the chosen files become one acquisition *run*. For nested pipelines, browse
    repeatedly to add runs; for flat pipelines every run is combined. Selections
    persist in ``st.session_state[state_key]`` as ``list[list[str]]``.
    """
    runs: list[list[str]] = st.session_state.setdefault(state_key, [])
    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    c_browse, c_clear = st.columns([3, 1])
    if c_browse.button("📁 Browse files…", key=f"browse_{state_key}", use_container_width=True):
        try:
            picked = pick_files(f"Select {label}", initialdir=initialdir)
        except RuntimeError as exc:
            st.error(str(exc))
            picked = []
        if picked:
            runs.append(picked)
            st.session_state[state_key] = runs
            st.rerun()
    if runs and c_clear.button("Clear", key=f"clear_{state_key}", use_container_width=True):
        st.session_state[state_key] = []
        st.rerun()

    resolved = ResolvedInput(runs=[list(run) for run in runs])
    if resolved.n_files:
        with st.expander(f"{resolved.n_files} file(s) · {len(resolved.runs)} run(s)", expanded=False):
            for i, run in enumerate(resolved.runs, 1):
                head = run[0]
                extra = f"  … (+{len(run) - 1} more)" if len(run) > 1 else ""
                st.caption(f"Run {i}: `{head}`{extra}" if nested else f"`{head}`{extra}")
    return resolved


# Folder the sample dialog opens in by default (the VENUS data root).
DEFAULT_BROWSE_DIR = "/SNS/VENUS/"

# --------------------------------------------------------------------------- #
# Debug mode — set NEUNORM_DEBUG=1 to auto-fill a known sample/OB/output for
# quick manual testing without clicking through the native file dialogs.
# --------------------------------------------------------------------------- #
DEBUG = bool(os.environ.get("NEUNORM_DEBUG"))
DEBUG_SAMPLE = (
    "/SNS/VENUS/IPTS-35825/images/ikonxl/raw/radiography/"
    "20260617_3EG__180_000s_2_800AngsMin/20260617_Run_23728_3EG__180_000s_2_800AngsMin_0.tiff"
)
DEBUG_OB = (
    "/SNS/VENUS/IPTS-35825/images/ikonxl/ob/"
    "20260617_OB2__300_000s_2_800AngsMin/20260617_Run_23755_OB2__300_000s_2_800AngsMin_ob_0.tiff"
)
DEBUG_OUTPUT_DIR = "/SNS/VENUS/IPTS-35825/shared/processed_data/remove_me_jean"


def apply_debug_defaults() -> None:
    """When ``NEUNORM_DEBUG`` is set, pre-select the debug sample/OB files once.

    Seeds ``st.session_state`` a single time per session so the selections can
    still be cleared or changed in the UI afterwards. No-op when debug is off.
    """
    if not DEBUG or st.session_state.get("_debug_seeded"):
        return
    st.session_state["_debug_seeded"] = True
    st.session_state["sample_files_input"] = [[DEBUG_SAMPLE]]
    st.session_state["ob_files_input"] = [[DEBUG_OB]]


def sibling_start_dir(sample_state_key: str = "sample_files_input") -> str:
    """Where OB / dark dialogs should open: the parent of the first sample file.

    Falls back to :data:`DEFAULT_BROWSE_DIR` when no sample has been picked yet.
    """
    runs = st.session_state.get(sample_state_key, [])
    if runs and runs[0]:
        return str(Path(runs[0][0]).parent)
    return DEFAULT_BROWSE_DIR


# --------------------------------------------------------------------------- #
# Pipeline call assembly + reproducible script
# --------------------------------------------------------------------------- #
def build_pipeline_kwargs(
    spec: "PipelineSpec",
    *,
    sample: ResolvedInput,
    ob: ResolvedInput,
    out_path: str,
    roi: Optional[tuple] = None,
    air_roi: Optional[tuple] = None,
    dark: Optional[ResolvedInput] = None,
    gamma_filter: bool = True,
    rebin_by_tof: "bool | int" = False,
    rebin_by_spatial: Optional[int] = None,
    detector_shape: Optional[tuple] = None,
    binning_cfg: Optional[BinningConfig] = None,
) -> dict:
    """Assemble the exact keyword arguments passed to ``spec.func``.

    A single source of truth so the reproducible-script preview and the actual
    execution use an identical call.
    """
    kwargs: dict = {"output_path": Path(out_path)}
    kwargs[spec.sample_kw] = sample.flat if not spec.nested else sample.runs
    kwargs[spec.ob_kw] = ob.flat if not spec.nested else ob.runs
    if roi is not None:
        kwargs["roi"] = roi
    if not spec.tof:
        kwargs["gamma_filter"] = gamma_filter
    if spec.supports_dark and dark is not None and dark.n_files:
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
    return kwargs


def _script_literal(value) -> str:
    """Render one kwarg value as valid, readable Python source."""
    if isinstance(value, Path):
        return f"Path({str(value)!r})"
    if isinstance(value, BinningConfig):
        args = ", ".join(f"{k}={val!r}" for k, val in value.model_dump().items())
        return f"BinningConfig({args})"
    if isinstance(value, list):
        # Re-indent pprint's continuation lines under the "    key=" column.
        return pprint.pformat(value, width=100).replace("\n", "\n    ")
    return repr(value)


def build_reduction_script(name: str, spec: "PipelineSpec", kwargs: dict) -> str:
    """Return a standalone, runnable Python script reproducing this reduction."""
    func = spec.func
    lines = [
        '"""NeuNorm reduction script — generated by the Streamlit app.',
        "",
        f"Pipeline: {name}  ({spec.facility} · {spec.detector})",
        '"""',
        "from pathlib import Path",
        "",
        f"from {func.__module__} import {func.__name__}",
    ]
    if any(isinstance(v, BinningConfig) for v in kwargs.values()):
        lines.append("from neunorm.data_models.tof import BinningConfig")
    lines += ["", f"transmission = {func.__name__}("]
    for key, value in kwargs.items():
        lines.append(f"    {key}={_script_literal(value)},")
    lines += [")", "", f'print("Wrote", {str(kwargs["output_path"])!r})']
    return "\n".join(lines)


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

    apply_debug_defaults()

    # ---- Sidebar: pipeline + options -------------------------------------- #
    with st.sidebar:
        if DEBUG:
            st.warning("🐞 DEBUG mode — sample/OB/output pre-filled (NEUNORM_DEBUG).")
        st.header("Pipeline")
        default_pipeline = list(PIPELINES).index("VENUS CCD/CMOS") if DEBUG else 0
        name = st.selectbox("Detector / facility", list(PIPELINES), index=default_pipeline)
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
        out_fmt = st.selectbox("Format", ["hdf5", "tiff"], index=1)
        # Derive the default name from the sample files picked on a previous
        # rerun (the sample widget renders later, so read it from session_state).
        base = output_basename(st.session_state.get("sample_files_input", []))
        stem = base or f"neunorm_{spec.facility.lower()}_{spec.detector.split('/')[0].lower()}"
        default_out = f"{stem}.{out_fmt}"
        out_dir = DEBUG_OUTPUT_DIR if DEBUG else Path.home()
        out_path = st.text_input("Output path", value=str(Path(out_dir) / default_out))

    # ---- Main: inputs ----------------------------------------------------- #
    st.subheader("Input files")
    if spec.nested:
        st.caption("**Browse** to pick each acquisition run's files — browse again to add more runs.")
    else:
        st.caption("**Browse** to pick the event HDF5 files (all selections are combined).")

    # OB / dark dialogs open next to the chosen sample; the sample starts at the VENUS root.
    sibling_dir = sibling_start_dir()
    col_s, col_o = st.columns(2)
    with col_s:
        sample = file_input("Sample files", "sample_files_input", nested=spec.nested, initialdir=DEFAULT_BROWSE_DIR)
    with col_o:
        ob = file_input("Open-beam (OB) files", "ob_files_input", nested=spec.nested, initialdir=sibling_dir)

    dark = None
    if spec.supports_dark:
        dark = file_input(
            "Dark files (optional)",
            "dark_files_input",
            nested=spec.nested,
            help_text="Leave empty to skip dark correction.",
            initialdir=sibling_dir,
        )

    cols = st.columns(3)
    cols[0].metric("Sample files", sample.n_files, f"{len(sample.runs)} run(s)")
    cols[1].metric("OB files", ob.n_files, f"{len(ob.runs)} run(s)")
    if spec.supports_dark:
        cols[2].metric("Dark files", dark.n_files if dark else 0, f"{len(dark.runs) if dark else 0} run(s)")

    run = st.button("▶ Run pipeline", type="primary", disabled=not (sample.n_files and ob.n_files))

    # Assemble the exact pipeline call once — feeds both the script preview and the run.
    kwargs = None
    kwargs_error = None
    if sample.n_files and ob.n_files:
        try:
            roi = parse_roi(roi_text)
            air_roi = parse_roi(air_roi_text) if spec.supports_air_roi else None
        except ValueError as exc:
            kwargs_error = f"ROI error: {exc}"
        else:
            kwargs = build_pipeline_kwargs(
                spec,
                sample=sample,
                ob=ob,
                out_path=out_path,
                roi=roi,
                air_roi=air_roi,
                dark=dark,
                gamma_filter=gamma_filter,
                rebin_by_tof=rebin_by_tof,
                rebin_by_spatial=rebin_by_spatial,
                detector_shape=detector_shape,
                binning_cfg=binning_cfg,
            )

    # ---- Advanced: exact reproducible script ------------------------------ #
    with st.expander("🧪 Advanced · reproducible script", expanded=False):
        if kwargs_error:
            st.warning(kwargs_error)
        elif kwargs is not None:
            script = build_reduction_script(name, spec, kwargs)
            st.caption("The exact standalone script this app runs — copy or download to reproduce it outside the UI.")
            st.code(script, language="python")
            st.download_button("⬇ Download script", script, file_name="neunorm_reduction.py", mime="text/x-python")
        else:
            st.caption("Select sample and open-beam files to preview the script.")

    # ---- Run -------------------------------------------------------------- #
    if run:
        if kwargs_error:
            st.error(kwargs_error)
            st.stop()

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
