"""NeuNorm Streamlit app — styled with the ORNL *Coefficient* design system.

Same functionality as ``neunorm_app.py`` (it reuses that module's pipeline
registry, input helpers, and result viewer), re-skinned to follow the ORNL UX
Coefficient guidelines documented at <https://ux.ornl.gov>:

* **Brand colour** — primary ORNL green ``#0F8723`` (the ``--ords-primary`` /
  ``--primary-rich`` token), with semantic success/danger/warning/info tokens.
* **Branded header** — a full-width "shell bar" with the ORNL leaf mark in the
  Branding slot (left), the app title, and an Actions slot (right), rendered in
  the *Branded* treatment (rich-primary background, white foreground).
* **Compiled-style typography** — an ``ords``-namespaced type scale and card /
  button / input styling applied to Streamlit's widgets via injected CSS custom
  properties (design tokens as data).

Because Streamlit renders its own DOM, the Coefficient tokens are expressed here
as CSS custom properties and mapped onto Streamlit's components rather than
pulling the ``@ornl-ux/coefficient-foundations`` package.

Run it with::

    pixi run streamlit-app-ornl
    # or:
    pixi run -e streamlit streamlit run apps/streamlit/neunorm_app_ornl_design.py
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import streamlit as st

# Reuse the tested pipeline logic + viewer from the base app (single source of truth).
from neunorm_app import (
    PIPELINES,
    BinningConfig,
    build_binning_form,
    parse_roi,
    render_results,
    resolve_globs,
)

from neunorm import __version__

# --------------------------------------------------------------------------- #
# ORNL Coefficient design tokens (light mode)
# --------------------------------------------------------------------------- #
# Semantic tokens mirror the Coefficient naming (`--ords-*`, `--primary-rich`,
# `--surface-*`, `--text-*`). Primary green `#0F8723` is the documented brand
# value; the rest are a light-mode palette consistent with the guidelines.
ORNL = {
    "primary": "#0F8723",  # --ords-primary / --primary-rich (ORNL green)
    "primary_dark": "#0A6B1A",  # hover / active
    "primary_weak": "#E7F3E9",  # tinted surface
    "surface_base": "#FFFFFF",  # --surface-base (page)
    "surface_weak": "#F4F6F5",  # --surface-weak (subtle panels)
    "text_base": "#1F2421",  # --text-base
    "text_muted": "#5A625C",  # secondary text
    "text_white": "#FFFFFF",  # --text-white
    "border": "#D6DBD7",  # hairline borders
    "success": "#0F8723",  # --ifm-color-success
    "danger": "#C4314B",  # --ifm-color-danger
    "warning": "#B26A00",  # --ifm-color-warning
    "info": "#005B94",  # --ifm-color-info
    "radius": "8px",
    "font": '"Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif',
}

# Phosphor-style leaf mark (ph:leaf) for the Branding slot, inlined as SVG.
LEAF_SVG = (
    '<svg width="30" height="30" viewBox="0 0 256 256" fill="currentColor" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M223.45 40.07a8 8 0 0 0-7.52-7.52C143.34 28.83 78.05 51 49.39 '
    "104.53c-12.4 23.17-13.8 51.19-4.62 79.32L32.6 195.93a8 8 0 0 0 11.32 "
    "11.32l12.08-12.17c14.11 4.6 27.66 6.92 40.24 6.92 13.68 0 26.26-2.74 "
    "37.32-8.26 53.54-28.66 75.71-93.95 71.99-153.67ZM128 152a8 8 0 0 1-11.32 "
    '0 8 8 0 0 1 0-11.32l50.34-50.35a8 8 0 0 1 11.32 11.32Z"/></svg>'
)


def inject_css() -> None:
    """Register the ORNL design tokens and map them onto Streamlit widgets."""
    t = ORNL
    st.markdown(
        f"""
        <style>
        :root {{
            --ords-primary: {t["primary"]};
            --primary-rich: {t["primary"]};
            --primary-dark: {t["primary_dark"]};
            --primary-weak: {t["primary_weak"]};
            --surface-base: {t["surface_base"]};
            --surface-weak: {t["surface_weak"]};
            --text-base: {t["text_base"]};
            --text-muted: {t["text_muted"]};
            --text-white: {t["text_white"]};
            --ords-border: {t["border"]};
            --ords-radius: {t["radius"]};
        }}

        html, body, [class*="css"], .stApp {{
            font-family: {t["font"]};
            color: var(--text-base);
        }}
        .stApp {{ background: var(--surface-weak); }}

        /* Compiled-style typography scale (ords-*) */
        .ords-heading-h1 {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2;
                            color: var(--text-white); margin: 0; }}
        .ords-subtitle   {{ font-size: 0.95rem; font-weight: 400; opacity: 0.92;
                            color: var(--text-white); margin: 0; }}
        .ords-section    {{ font-size: 1.15rem; font-weight: 700; color: var(--text-base);
                            border-left: 4px solid var(--ords-primary); padding-left: 0.55rem;
                            margin: 0.25rem 0 0.75rem; }}

        /* Branded header shell bar (Branding | title | Actions slots) */
        .ords-header {{
            display: flex; align-items: center; gap: 0.9rem;
            background: var(--primary-rich); color: var(--text-white);
            padding: 0.85rem 1.25rem; border-radius: var(--ords-radius);
            margin-bottom: 1.1rem; box-shadow: 0 2px 8px rgba(15,135,35,0.25);
        }}
        .ords-header .leaf {{ display: flex; color: var(--text-white); }}
        .ords-header .brand {{ display: flex; flex-direction: column; }}
        .ords-header .actions {{ margin-left: auto; font-size: 0.82rem; opacity: 0.9;
                                 text-align: right; }}

        /* Cards / containers */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--surface-base);
            border: 1px solid var(--ords-border);
            border-radius: var(--ords-radius);
        }}

        /* Primary button = ORNL green; secondary = de-emphasized outline */
        .stButton > button, .stDownloadButton > button {{
            border-radius: var(--ords-radius); font-weight: 600;
            border: 1px solid var(--ords-primary);
        }}
        .stButton > button[kind="primary"] {{
            background: var(--ords-primary); color: var(--text-white);
            border-color: var(--ords-primary);
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--primary-dark); border-color: var(--primary-dark);
        }}
        .stButton > button[kind="secondary"], .stDownloadButton > button {{
            background: var(--surface-base); color: var(--ords-primary);
        }}
        .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {{
            background: var(--primary-weak); color: var(--primary-dark);
        }}

        /* Sidebar as a distinct surface */
        [data-testid="stSidebar"] {{
            background: var(--surface-base);
            border-right: 1px solid var(--ords-border);
        }}

        /* Inputs / focus ring in brand green */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            border-radius: 6px;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {{
            border-color: var(--ords-primary) !important;
            box-shadow: 0 0 0 2px var(--primary-weak) !important;
        }}
        [data-testid="stMetric"] {{
            background: var(--surface-weak); border: 1px solid var(--ords-border);
            border-radius: var(--ords-radius); padding: 0.6rem 0.8rem;
        }}
        [data-baseweb="tag"] {{ background: var(--ords-primary) !important; }}
        a {{ color: var(--ords-primary); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ornl_header() -> None:
    """Render the Branded shell-bar header (leaf mark + title + actions slot)."""
    st.markdown(
        f"""
        <header class="ords-header">
            <span class="leaf">{LEAF_SVG}</span>
            <span class="brand">
                <span class="ords-heading-h1">NeuNorm</span>
                <span class="ords-subtitle">Neutron Imaging Normalization &middot; ORNL</span>
            </span>
            <span class="actions">MARS &middot; HFIR<br/>VENUS &middot; SNS<br/>v{__version__}</span>
        </header>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="ords-section">{title}</div>', unsafe_allow_html=True)


def apply_plot_theme() -> None:
    """Nudge matplotlib (used by the shared viewer) toward the ORNL palette."""
    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(color=[ORNL["primary"], ORNL["info"], ORNL["danger"]]),
            "axes.edgecolor": ORNL["border"],
            "axes.labelcolor": ORNL["text_base"],
            "xtick.color": ORNL["text_muted"],
            "ytick.color": ORNL["text_muted"],
            "figure.facecolor": ORNL["surface_base"],
            "axes.facecolor": ORNL["surface_base"],
        }
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: C901  (UI assembly is intentionally one linear flow)
    st.set_page_config(page_title="NeuNorm · ORNL", page_icon="🍃", layout="wide")
    inject_css()
    apply_plot_theme()
    ornl_header()

    # ---- Sidebar: pipeline + options -------------------------------------- #
    with st.sidebar:
        section("Pipeline")
        name = st.selectbox("Detector / facility", list(PIPELINES))
        spec = PIPELINES[name]
        st.info(f"**{spec.facility} · {spec.detector}**\n\n{spec.notes}")

        section("Corrections")
        gamma_filter = True
        if not spec.tof:
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
            section("Rebinning")
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
            section("TOF binning")
            binning_cfg = build_binning_form()

        section("Output")
        out_fmt = st.selectbox("Format", ["hdf5", "tiff"], index=0)
        default_out = f"neunorm_{spec.facility.lower()}_{spec.detector.split('/')[0].lower()}.{out_fmt}"
        out_path = st.text_input("Output path", value=str(Path.cwd() / default_out))

    # ---- Main: inputs ----------------------------------------------------- #
    section("Input files")
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

    run = st.button("▶ Run Pipeline", type="primary", disabled=not (sample.n_files and ob.n_files))

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
