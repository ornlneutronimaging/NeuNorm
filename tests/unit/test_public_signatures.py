"""Signature-guard tests for the released public entry points (#195, Task 2).

These pin the **positional** parameter order of every pipeline and of ``write_tiff_stack``, so a
future parameter cannot be inserted mid-signature and silently re-bind every argument after it.

That is not hypothetical. Commit 3bd2b07 inserted ``rebin_reduction`` immediately after
``rebin_by_tof`` in the three VENUS TOF pipelines, shifting every later positional parameter — five
of them in ``venus_tpx3_event`` (``rebin_by_spatial``, ``detector_shape``, ``event_id_offset``,
``bank_name``, ``flight_path``). A caller passing any of those positionally started binding it to
the wrong parameter. The existing pipeline tests could not catch it because they pass every argument
by keyword.

The lists below are the signatures as released. **Appending** is fine — a new trailing parameter,
preferably keyword-only, does not move anything. Reordering, inserting, or removing is a breaking
change: if one of these tests fails, either revert the reorder or treat it as a deliberate API break
with a CHANGELOG entry.

Pattern follows ``test_rebin_tof_flexible.py::test_rebin_tof_preserves_legacy_positional_signature``,
which guards the leaf ``rebin_tof`` the same way.
"""

import inspect

import pytest

from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.pipelines.mars_tpx3 import run_mars_tpx3_pipeline
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline

# Released positional order per entry point. Keep in sync ONLY by appending.
RELEASED_POSITIONAL = {
    "run_venus_tpx1_pipeline": (
        run_venus_tpx1_pipeline,
        [
            "sample_hdf5_paths",
            "ob_hdf5_paths",
            "sample_tiff_paths",
            "ob_tiff_paths",
            "output_path",
            "roi",
            "air_roi",
            "rebin_by_tof",
            "rebin_reduction",
            "rebin_by_spatial",
            "flight_path",
            "tiff_one_file_per_image",
        ],
    ),
    "run_venus_tpx3_histogram_pipeline": (
        run_venus_tpx3_histogram_pipeline,
        [
            "sample_hdf5_paths",
            "ob_hdf5_paths",
            "sample_tiff_paths",
            "ob_tiff_paths",
            "output_path",
            "roi",
            "air_roi",
            "rebin_by_tof",
            "rebin_reduction",
            "rebin_by_spatial",
            "flight_path",
            "tiff_one_file_per_image",
        ],
    ),
    "run_venus_tpx3_event_pipeline": (
        run_venus_tpx3_event_pipeline,
        [
            "sample_paths",
            "ob_paths",
            "binning",
            "output_path",
            "roi",
            "air_roi",
            "rebin_by_tof",
            "rebin_reduction",
            "rebin_by_spatial",
            "detector_shape",
            "event_id_offset",
            "bank_name",
            "flight_path",
            "tiff_one_file_per_image",
        ],
    ),
    "run_venus_ccd_pipeline": (
        run_venus_ccd_pipeline,
        [
            "sample_paths",
            "ob_paths",
            "dark_paths",
            "output_path",
            "roi",
            "gamma_filter",
            "air_roi",
            "background_roi",
        ],
    ),
    "run_mars_ccd_pipeline": (
        run_mars_ccd_pipeline,
        [
            "sample_paths",
            "ob_paths",
            "dark_paths",
            "output_path",
            "roi",
            "gamma_filter",
            "background_roi",
        ],
    ),
    "run_mars_tpx3_pipeline": (
        run_mars_tpx3_pipeline,
        [
            "sample_paths",
            "ob_paths",
            "output_path",
            "roi",
            "gamma_filter",
            "detector_shape",
            "background_roi",
        ],
    ),
    "write_tiff_stack": (
        write_tiff_stack,
        [
            "output_path",
            "transmission",
            "metadata",
            "daqmetadata",
            "one_file_per_image",
            "concat_stdevs_and_mask",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(RELEASED_POSITIONAL))
def test_released_positional_order_is_unchanged(name):
    """The ordered positional-capable parameters must match the released signature exactly.

    Failing here means an argument was inserted or reordered, so every caller passing a later
    argument positionally now binds it to the wrong parameter — silently, unless the types happen
    to clash.
    """
    func, expected = RELEASED_POSITIONAL[name]
    params = inspect.signature(func).parameters
    actual = [n for n, p in params.items() if p.kind is not p.KEYWORD_ONLY]

    assert actual[: len(expected)] == expected, (
        f"{name}: positional parameter order changed.\n"
        f"  released: {expected}\n"
        f"  current:  {actual}\n"
        "Append new parameters (ideally keyword-only) instead of inserting them."
    )


@pytest.mark.parametrize("name", sorted(RELEASED_POSITIONAL))
def test_no_positional_parameter_was_dropped(name):
    """Every released parameter must still exist under its own name, so keyword callers keep
    working even if the parameter later becomes keyword-only."""
    func, expected = RELEASED_POSITIONAL[name]
    params = inspect.signature(func).parameters
    missing = [n for n in expected if n not in params]

    assert not missing, f"{name}: released parameter(s) removed: {missing}"


@pytest.mark.parametrize("name", sorted(RELEASED_POSITIONAL))
def test_any_new_parameter_is_appended_not_inserted(name):
    """Growth is allowed; movement is not. Anything beyond the released list must sit after it."""
    func, expected = RELEASED_POSITIONAL[name]
    params = inspect.signature(func).parameters
    actual = [n for n, p in params.items() if p.kind is not p.KEYWORD_ONLY]
    added = actual[len(expected) :]

    assert set(added).isdisjoint(expected), (
        f"{name}: {sorted(set(added) & set(expected))} appear both inside and after the released "
        "prefix — the signature was reordered, not appended to."
    )
