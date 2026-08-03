"""The per-pipeline differences the shared VENUS TOF spine had to preserve.

``venus_tpx1``, ``venus_tpx3_histogram`` and ``venus_tpx3_event`` now run the same middle
(``_tof_spine.reduce_tof_stacks``) and differ only through a ``TofPipelineProfile``. A byte-comparison
harness pins most of that by diffing written HDF5 against the pre-refactor code, but two profile fields
never reach an HDF5 file: ``tiff_detector_model`` lands only in the scitiff DAQ metadata, and
``hdf5_hot_pixel_mask`` is unobservable on the one pipeline that leaves it unset. Those two, the mask
layout each pipeline writes, and the profile table itself are what this file covers.

Every value pinned here is PRESERVED PRE-EXISTING behaviour of the three pipelines, not a choice the
refactor made, so the tests are change detectors on purpose. Two of the values look like oversights and
are kept deliberately: ``venus_tpx3_histogram`` re-detects its masks from the SAMPLE after a spatial
rebin where its own pre-rebin detection and both sibling pipelines read the open beam, and
``venus_tpx1`` hands the HDF5 writer no ``hot_pixel_mask`` name. Editing an expected value here changes
what NeuNorm publishes to users.

One further difference is pinned in passing because the written masks expose it: the two TIFF pipelines
carry spatial dims ``(y, x)`` (the order ``load_tiff_stack`` produces), while the event pipeline
histograms into ``(tof, x, y)``, so its ``/masks/*`` datasets land TRANSPOSED relative to the other two.
That one is not in the profile — it comes from the loaders — and it is equally pre-existing.
"""

import dataclasses
import re

import h5py
import numpy as np
import pytest
from PIL import Image
from scitiff.io import load_scitiff
from test_progress_pipelines import _spectra_file, _venus_metadata_nexus

from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.venus_tpx1 import _TPX1_PROFILE, run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import _TPX3_EVENT_PROFILE, run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import _TPX3_HISTOGRAM_PROFILE, run_venus_tpx3_histogram_pipeline

# --------------------------------------------------------------------------------------
# synthetic detector: small, even-sided so rebin_by_spatial=2 divides it
# --------------------------------------------------------------------------------------

_SIZE = 8
_FRAMES = 4
_SAMPLE_COUNTS = 60.0
_OB_COUNTS = 100.0
_HOT_COUNTS = 5000.0
_SAMPLE_CHARGE = 12345
_OB_CHARGE = 24690

#: One dead and one hot pixel in the OPEN BEAM, which is the stack all three pipelines detect their
#: pre-rebin masks from. Both off-diagonal, so a transposed written mask cannot pass.
_DEAD_PIXEL = (1, 5)
_HOT_PIXEL = (6, 2)

#: Dead 2x2 blocks, placed so a spatial rebin by 2 collapses each to exactly one superpixel, at
#: different and non-symmetric positions in the sample and the open beam.
_SAMPLE_DEAD_BLOCK = (slice(0, 2), slice(2, 4))  # -> superpixel (y=0, x=1)
_OB_DEAD_BLOCK = (slice(4, 6), slice(0, 2))  # -> superpixel (y=2, x=0)

#: The detector name in the synthetic NeXus metadata (``BL10:Exp:Det``), which two of the three
#: pipelines write as the TIFF detector model while the event pipeline overrides it.
_METADATA_DETECTOR = "MCP TPX3"

_PIPELINES = {
    "venus_tpx1": run_venus_tpx1_pipeline,
    "venus_tpx3_histogram": run_venus_tpx3_histogram_pipeline,
    "venus_tpx3_event": run_venus_tpx3_event_pipeline,
}

#: The event pipeline's spatial dims are (x, y), the TIFF pipelines' are (y, x); expectations below
#: are written in the (y, x) detector frame and transposed for the event pipeline.
_XY_AXES = {"venus_tpx1": False, "venus_tpx3_histogram": False, "venus_tpx3_event": True}


# --------------------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------------------


def _frame(counts, *, dead=(), hot=(), dead_block=None):
    """One detector frame of uniform counts, with the named pixels zeroed or spiked."""
    frame = np.full((_SIZE, _SIZE), counts, dtype=np.float32)
    for y, x in dead:
        frame[y, x] = 0.0
    for y, x in hot:
        frame[y, x] = _HOT_COUNTS
    if dead_block is not None:
        frame[dead_block] = 0.0
    return frame


def _tiff_stack(directory, prefix, frame):
    """``_FRAMES`` identical TIFFs of ``frame`` — the pre-binned stack both TIFF pipelines read."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(_FRAMES):
        path = directory / f"{prefix}_{index:03}.tiff"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def _event_file(path, frame, proton_charge):
    """A TPX3 NeXus event file whose per-pixel event count, in every TOF frame, is ``frame[y, x]``.

    Same structure as the event fixtures elsewhere in the suite, but with a per-pixel count instead of
    a flat flood, so dead and hot pixels can be planted.
    """
    event_ids, tofs = [], []
    for y in range(_SIZE):
        for x in range(_SIZE):
            repeats = int(frame[y, x])
            if repeats == 0:  # a dead pixel emits nothing
                continue
            for tof_frame in range(_FRAMES):
                event_ids.extend([y * _SIZE + x + 1_000_000] * repeats)
                tofs.extend([100 + tof_frame * 5] * repeats)  # microseconds
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset("proton_charge", data=[proton_charge])
        entry.create_dataset("duration", data=[60.0])
        bank = entry.create_group("bank100_events")
        bank.create_dataset("event_time_offset", data=tofs)
        bank.create_dataset("event_id", data=event_ids, dtype=np.int32)
        daslogs = entry.create_group("DASlogs")
        daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        daslogs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[_METADATA_DETECTOR.encode()]])
    return path


def _event_kwargs(sample_frame, ob_frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "sample_paths": [_event_file(directory / "event_sample.hdf5", sample_frame, _SAMPLE_CHARGE)],
        "ob_paths": [_event_file(directory / "event_ob.hdf5", ob_frame, _OB_CHARGE)],
        "binning": BinningConfig(
            bins=_FRAMES,
            bin_space="tof",
            tof_range=(100_000, 100_000 + _FRAMES * 5_000),
            use_log_bin=False,
        ),
        "detector_shape": (_SIZE, _SIZE),
    }


def _tiff_kwargs(sample_frame, ob_frame, directory, *, spectra_sidecar):
    """Kwargs for the two TIFF-plus-NeXus pipelines. TPX1 additionally reads a spectra sidecar."""
    sample_dir = directory / "sample"
    ob_dir = directory / "ob"
    sample_tiffs = _tiff_stack(sample_dir, "sample", sample_frame)
    ob_tiffs = _tiff_stack(ob_dir, "ob", ob_frame)
    if spectra_sidecar:
        edges = [round(0.1 * (index + 1), 1) for index in range(_FRAMES)]
        _spectra_file(sample_dir / "sample_Spectra.txt", edges)
        _spectra_file(ob_dir / "ob_Spectra.txt", edges)
    return {
        "sample_tiff_paths": [sample_tiffs],
        "ob_tiff_paths": [ob_tiffs],
        "sample_hdf5_paths": [
            _venus_metadata_nexus(directory / "nexus" / "sample.nxs.h5", _SAMPLE_CHARGE, tof_bins=_FRAMES)
        ],
        "ob_hdf5_paths": [_venus_metadata_nexus(directory / "nexus" / "ob.nxs.h5", _OB_CHARGE, tof_bins=_FRAMES)],
    }


def _inputs_for(root, sample_frame, ob_frame):
    """The same synthetic sample/open-beam pair, expressed for each of the three pipelines."""
    return {
        "venus_tpx1": _tiff_kwargs(sample_frame, ob_frame, root / "tpx1", spectra_sidecar=True),
        "venus_tpx3_histogram": _tiff_kwargs(sample_frame, ob_frame, root / "histogram", spectra_sidecar=False),
        "venus_tpx3_event": _event_kwargs(sample_frame, ob_frame, root / "event"),
    }


def _strip_detector_log(source, destination):
    """Copy a synthetic NeXus metadata file without its ``BL10:Exp:Det`` detector-name log."""
    with h5py.File(source, "r") as src, h5py.File(destination, "w") as dst:
        src.copy("entry", dst)
        del dst["entry/DASlogs/BL10:Exp:Det"]
    return destination


@pytest.fixture(scope="module")
def flood_inputs(tmp_path_factory):
    """Uniform counts, with one dead and one hot pixel in the OPEN BEAM.

    A plain flood leaves every mask empty and makes a mask assertion vacuous, so the open beam — the
    stack all three pipelines detect their pre-rebin masks from — carries both kinds of bad pixel.
    """
    root = tmp_path_factory.mktemp("spine_flood")
    return _inputs_for(
        root,
        _frame(_SAMPLE_COUNTS),
        _frame(_OB_COUNTS, dead=[_DEAD_PIXEL], hot=[_HOT_PIXEL]),
    )


@pytest.fixture(scope="module")
def split_dead_inputs(tmp_path_factory):
    """Dead 2x2 blocks in DIFFERENT places in the sample and the open beam.

    This is what makes the post-spatial-rebin re-detection source observable: the two candidate sources
    disagree, so the written mask says which stack the pipeline read.
    """
    root = tmp_path_factory.mktemp("spine_split_dead")
    return _inputs_for(
        root,
        _frame(_SAMPLE_COUNTS, dead_block=_SAMPLE_DEAD_BLOCK),
        _frame(_OB_COUNTS, dead_block=_OB_DEAD_BLOCK),
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _run(name, inputs, output_path, **extra):
    return _PIPELINES[name](output_path=output_path, **inputs[name], **extra)


def _mask(size, pixels, *, xy_axes):
    """A boolean mask with the given ``(y, x)`` positions set, transposed for ``(x, y)`` output."""
    expected = np.zeros((size, size), dtype=bool)
    for y, x in pixels:
        expected[y, x] = True
    return expected.T if xy_axes else expected


def _written_masks(output_path):
    """The ``/masks`` dataset names in the written HDF5, and their contents."""
    with h5py.File(output_path, "r") as f:
        return sorted(f["masks"]), {mask: f[f"masks/{mask}"][()] for mask in f["masks"]}


# --------------------------------------------------------------------------------------
# 1. the TIFF detector model, which only ever reaches the scitiff DAQ metadata
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_detector_type"),
    [
        ("venus_tpx1", _METADATA_DETECTOR),
        ("venus_tpx3_histogram", _METADATA_DETECTOR),
        ("venus_tpx3_event", "TPX3"),
    ],
)
def test_tiff_detector_type_is_the_profiles_model(flood_inputs, tmp_path, name, expected_detector_type):
    """The ``detector_type`` in the written scitiff DAQ block, read back out of the file.

    Read back rather than intercepting the writer call: this is user-visible output, and the round trip
    covers the whole path from the profile to the bytes on disk. It is also the only route — the DAQ
    block is not in the HDF5 output the byte-comparison harness diffs, so nothing else reaches it.

    Two pipelines read the model from the sample's ``detector`` coordinate; the event pipeline
    hard-codes ``"TPX3"`` and therefore writes something DIFFERENT from the coordinate its own metadata
    carries — the synthetic input names the detector "MCP TPX3" in all three cases, which is what makes
    the override visible instead of coincidentally equal.
    """
    output_path = tmp_path / f"{name}.tiff"
    _run(name, flood_inputs, output_path)

    daq = load_scitiff(output_path)["daq"]
    assert daq.detector_type == expected_detector_type
    assert daq.facility == "SNS"
    assert daq.instrument == "VENUS"


@pytest.mark.parametrize("name", ["venus_tpx1", "venus_tpx3_histogram"])
def test_tiff_detector_type_falls_back_to_unknown_without_a_detector_coord(flood_inputs, tmp_path, name):
    """With no ``detector`` in the metadata, the two profiles that read it write "Unknown".

    The event pipeline is excluded on purpose: its model is a literal, so it never reaches this
    fallback — that asymmetry is what the test above pins.
    """
    inputs = dict(flood_inputs[name])
    for key, tag in (("sample_hdf5_paths", "sample"), ("ob_hdf5_paths", "ob")):
        inputs[key] = [_strip_detector_log(inputs[key][0], tmp_path / f"{name}_{tag}_no_detector.nxs.h5")]

    output_path = tmp_path / f"{name}_unknown.tiff"
    _PIPELINES[name](output_path=output_path, **inputs)

    assert load_scitiff(output_path)["daq"].detector_type == "Unknown"


# --------------------------------------------------------------------------------------
# 2. the profile table itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        pytest.param(
            _TPX1_PROFILE,
            {
                "label": "VENUS TPX1",
                "detect_hot": False,
                "remask_after_spatial_rebin_from": "ob",
                "hdf5_hot_pixel_mask": None,
                "tiff_detector_model": None,
            },
            id="venus_tpx1",
        ),
        pytest.param(
            _TPX3_HISTOGRAM_PROFILE,
            {
                "label": "VENUS TPX3 histogram",
                "detect_hot": True,
                "remask_after_spatial_rebin_from": "sample",
                "hdf5_hot_pixel_mask": "hot_pixels",
                "tiff_detector_model": None,
            },
            id="venus_tpx3_histogram",
        ),
        pytest.param(
            _TPX3_EVENT_PROFILE,
            {
                "label": "VENUS TPX3 event",
                "detect_hot": True,
                "remask_after_spatial_rebin_from": "ob",
                "hdf5_hot_pixel_mask": "hot_pixels",
                "tiff_detector_model": "TPX3",
            },
            id="venus_tpx3_event",
        ),
    ],
)
def test_profile_values_are_the_preserved_per_pipeline_behaviour(profile, expected):
    """Every field of every profile, as one table. A deliberate change detector.

    These are the three pipelines' PRE-EXISTING values, carried across the spine refactor unchanged,
    not decisions the refactor made. Two are surprising and still correct to keep:
    ``venus_tpx3_histogram`` re-detects post-spatial-rebin masks from the SAMPLE (the other two read
    the open beam, as does its own pre-rebin detection), and ``venus_tpx1`` passes the HDF5 writer no
    ``hot_pixel_mask`` name — a mask literally called ``hot_pixels`` would land at
    ``/masks/hot_pixels`` there and at ``/masks/hot`` in the other two. Changing an expected value here
    changes published output, so it needs a release note rather than a test update.

    Compared as a whole dict rather than field by field: a field added to ``TofPipelineProfile`` fails
    here until every pipeline's value for it is recorded.
    """
    assert dataclasses.asdict(profile) == expected


# --------------------------------------------------------------------------------------
# 3. the HDF5 mask layout each pipeline writes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_masks"),
    [
        ("venus_tpx1", ["dead"]),
        ("venus_tpx3_histogram", ["dead", "hot"]),
        ("venus_tpx3_event", ["dead", "hot"]),
    ],
)
def test_hdf5_mask_datasets_follow_the_profile(flood_inputs, tmp_path, name, expected_masks):
    """Which ``/masks/*`` datasets land in the output, and that they hold the planted bad pixels.

    ``venus_tpx1`` detects no hot pixels, so its output carries ``/masks/dead`` alone: a consumer that
    reads ``/masks/hot`` unconditionally works on the other two and raises on TPX1. That is on-disk
    layout, and pre-existing, so it is pinned.

    Each mask is compared against the exact pixels planted in the open beam rather than checked for
    non-emptiness — an all-False mask would satisfy "the dataset exists" while the detection did
    nothing, and a count would not notice a transposed mask.
    """
    output_path = tmp_path / f"{name}_masks.hdf5"
    _run(name, flood_inputs, output_path)

    xy_axes = _XY_AXES[name]
    expected = {
        "dead": _mask(_SIZE, [_DEAD_PIXEL], xy_axes=xy_axes),
        "hot": _mask(_SIZE, [_HOT_PIXEL], xy_axes=xy_axes),
    }

    written_names, written = _written_masks(output_path)
    assert written_names == expected_masks
    for mask_name in written_names:
        np.testing.assert_array_equal(written[mask_name], expected[mask_name])
        assert written[mask_name].any(), f"the {mask_name} mask is empty, so this input proves nothing"


# --------------------------------------------------------------------------------------
# 4. which stack the post-spatial-rebin masks are re-detected from
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_dead_superpixels"),
    [
        # The ob's dead block is at y 4:6, x 0:2 and the sample's at y 0:2, x 2:4, so a rebin by 2
        # collapses them to superpixels (2, 0) and (0, 1) respectively.
        #
        # (2, 0) appears under BOTH re-detection sources, and not by accident: the pre-rebin mask is
        # detected from the ob but attached to the SAMPLE, and scipp zeroes masked elements when it
        # sums, so the sample's superpixel over the ob's dead block sums to zero as well. The
        # sample's OWN dead block at (0, 1) is therefore the discriminating pixel — present only when
        # the re-detection reads the sample.
        ("venus_tpx3_histogram", [(0, 1), (2, 0)]),
        ("venus_tpx1", [(2, 0)]),
        ("venus_tpx3_event", [(2, 0)]),
    ],
)
def test_post_spatial_rebin_masks_come_from_the_profiles_source(
    split_dead_inputs, tmp_path, name, expected_dead_superpixels
):
    """After a spatial rebin, ``venus_tpx3_histogram`` re-detects from the SAMPLE and the other two
    from the open beam.

    Pre-existing behaviour, preserved, and it disagrees with the histogram pipeline's own pre-rebin
    detection, which reads the open beam like its siblings. It is observable only when the two stacks
    have bad pixels in different places, which is what this input arranges — so this is the test that
    catches a well-meaning "fix" unifying the three.
    """
    output_path = tmp_path / f"{name}_rebin_masks.hdf5"
    _run(name, split_dead_inputs, output_path, rebin_by_spatial=2)

    expected = _mask(_SIZE // 2, expected_dead_superpixels, xy_axes=_XY_AXES[name])
    _, written = _written_masks(output_path)
    np.testing.assert_array_equal(written["dead"], expected)
    assert written["dead"].any(), "no dead superpixel survived the rebin, so this input proves nothing"


# --------------------------------------------------------------------------------------
# 5. the export dispatch's fallthrough
# --------------------------------------------------------------------------------------


def test_unsupported_output_suffix_is_rejected(flood_inputs, tmp_path):
    """An unknown suffix raises instead of writing something unreadable under that name."""
    with pytest.raises(ValueError, match=re.escape("Unsupported output file format: .npy")):
        _run("venus_tpx3_histogram", flood_inputs, tmp_path / "out.npy")
