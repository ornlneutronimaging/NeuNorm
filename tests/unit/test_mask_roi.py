"""Unit tests for MaskROI (arbitrary-shape selection regions) and the unified region dispatch.

Covers construction/canonicalization from every accepted input form, pydantic model safety
(eq/hash/serialize on an array-bearing frozen model), the ``as_region_list`` grammar (including
the collision cases pinned by design: a bare 4-int sequence is always a rectangle, raw arrays and
paths are rejected), and the hardening of the legacy rect-only helpers.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from neunorm.data_models import MaskROI, as_region_list
from neunorm.data_models.roi import ROI, as_roi_bounds
from neunorm.processing.normalizer import as_roi_bounds_list


def _block_selection(ny=6, nx=8, y=slice(1, 3), x=slice(2, 5), dtype=bool):
    sel = np.zeros((ny, nx), dtype=dtype)
    sel[y, x] = 1
    return sel


class TestMaskROIConstruction:
    def test_bool_array(self):
        m = MaskROI(selection=_block_selection())
        assert m.n_selected == 6
        assert m.shape == (6, 8)
        assert m.selection.dtype == np.bool_

    def test_nonzero_thresholding_int_and_float(self):
        for dtype, value in [(np.int32, 7), (np.float64, 0.5)]:
            sel = np.zeros((6, 8), dtype=dtype)
            sel[1:3, 2:5] = value
            m = MaskROI(selection=sel)
            assert m.n_selected == 6, dtype

    def test_canonical_buffer_is_readonly_and_isolated(self):
        sel = _block_selection()
        m = MaskROI(selection=sel)
        assert not m.selection.flags.writeable
        with pytest.raises(ValueError):
            m.selection[0, 0] = True
        # mutating the caller's array must not change the region
        n_before = m.n_selected
        sel[:, :] = True
        assert m.n_selected == n_before

    def test_scipp_variable_yx_and_xy_are_equal(self):
        sel = _block_selection()
        m_np = MaskROI(selection=sel)
        m_yx = MaskROI(selection=sc.array(dims=["y", "x"], values=sel))
        m_xy = MaskROI(selection=sc.array(dims=["x", "y"], values=sel.T))
        assert m_np == m_yx == m_xy
        assert m_np.sha256() == m_xy.sha256()

    def test_scipp_variable_wrong_dims_raises(self):
        with pytest.raises(ValueError, match="dims"):
            MaskROI(selection=sc.array(dims=["row", "col"], values=_block_selection()))

    def test_numpy_dims_xy_transposes(self):
        sel = _block_selection()
        assert MaskROI(selection=sel.T, dims=("x", "y")) == MaskROI(selection=sel)

    def test_invalid_dims_kwarg_raises(self):
        with pytest.raises(ValueError, match="dims"):
            MaskROI(selection=_block_selection(), dims=("t", "x"))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            MaskROI(selection=np.ones(8, dtype=bool))
        with pytest.raises(ValueError, match="2D"):
            MaskROI(selection=np.ones((2, 3, 4), dtype=bool))

    def test_empty_selection_raises_structurally(self):
        with pytest.raises(ValueError, match="at least one pixel"):
            MaskROI(selection=np.zeros((6, 8), dtype=bool))

    def test_bounding_box_exclusive_stops(self):
        m = MaskROI(selection=_block_selection(y=slice(1, 3), x=slice(2, 5)))
        assert m.bounding_box() == (2, 1, 5, 3)

    def test_non_rectangular_selection(self):
        sel = np.zeros((8, 8), dtype=bool)
        rr, cc = np.ogrid[:8, :8]
        disk = (rr - 4) ** 2 + (cc - 4) ** 2 <= 4
        sel[disk] = True
        m = MaskROI(selection=sel)
        assert m.n_selected == int(disk.sum())


class TestMaskROIPydantic:
    def test_eq_and_hash(self):
        a = MaskROI(selection=_block_selection())
        b = MaskROI(selection=_block_selection().astype(np.int8) * 3)  # same region, different input form
        c = MaskROI(selection=_block_selection(x=slice(0, 2)))
        assert a == b and hash(a) == hash(b)
        assert a != c
        assert a != "not a region"

    def test_frozen(self):
        m = MaskROI(selection=_block_selection())
        with pytest.raises(Exception):  # pydantic frozen -> ValidationError
            m.source = "changed"

    def test_model_dump_json_is_summary(self):
        m = MaskROI(selection=_block_selection())
        dumped = m.model_dump_json()
        assert '"n_selected":6' in dumped.replace(" ", "")
        assert '"sha256"' in dumped

    def test_repr(self):
        r = repr(MaskROI(selection=_block_selection()))
        assert "ny=6" in r and "nx=8" in r and "n_selected=6" in r

    def test_provenance_summary_json_safe(self):
        import json

        s = MaskROI(selection=_block_selection()).provenance_summary()
        assert json.loads(json.dumps(s)) == s
        assert s["shape"] == [6, 8] and s["n_selected"] == 6 and s["source"] == "array"


class TestMaskROIFromFile:
    @pytest.mark.parametrize("suffix", [".png", ".tiff"])
    def test_grayscale_roundtrip(self, tmp_path, suffix):
        from PIL import Image

        sel = _block_selection(dtype=np.uint8) * 255
        path = tmp_path / f"mask{suffix}"
        Image.fromarray(sel).save(path)
        m = MaskROI.from_file(path)
        assert m == MaskROI(selection=sel)
        assert m.source == str(path)

    def test_rgb_any_channel_selects(self, tmp_path):
        from PIL import Image

        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[1:3, 2:5, 0] = 255  # red channel only
        path = tmp_path / "mask_rgb.png"
        Image.fromarray(rgb).save(path)
        assert MaskROI.from_file(path) == MaskROI(selection=_block_selection())


class TestMaskROIFromDataArrayMask:
    def _da(self):
        da = sc.DataArray(sc.ones(sizes={"y": 6, "x": 8}))
        da.masks["dead"] = sc.array(dims=["y", "x"], values=_block_selection())
        return da

    def test_invert_false_selects_flagged(self):
        m = MaskROI.from_dataarray_mask(self._da(), "dead", invert=False)
        assert m == MaskROI(selection=_block_selection())
        assert m.source == "dataarray_mask:dead"

    def test_invert_true_selects_kept(self):
        m = MaskROI.from_dataarray_mask(self._da(), "dead", invert=True)
        assert m == MaskROI(selection=~_block_selection())

    def test_invert_is_required_keyword(self):
        with pytest.raises(TypeError):
            MaskROI.from_dataarray_mask(self._da(), "dead", True)  # noqa: too many positional

    def test_missing_mask_raises(self):
        with pytest.raises(ValueError, match="no mask named"):
            MaskROI.from_dataarray_mask(self._da(), "hot", invert=False)

    def test_non_spatial_mask_raises(self):
        da = self._da()
        da.masks["frame"] = sc.array(dims=["y"], values=np.zeros(6, dtype=bool))
        with pytest.raises(ValueError, match="spatial"):
            MaskROI.from_dataarray_mask(da, "frame", invert=False)


class TestAsRegionList:
    def test_single_forms(self):
        m = MaskROI(selection=_block_selection())
        assert as_region_list(m) == [m]
        assert as_region_list(ROI(x0=1, y0=2, x1=3, y1=4)) == [(1, 2, 3, 4)]
        assert as_region_list((1, 2, 3, 4)) == [(1, 2, 3, 4)]
        assert as_region_list([0, 0, 1, 1]) == [(0, 0, 1, 1)]  # bare 4-int is ALWAYS a rectangle

    def test_mixed_pooled_list(self):
        m = MaskROI(selection=_block_selection())
        out = as_region_list([ROI(x0=0, y0=0, width=2, height=2), m, (5, 5, 7, 7)])
        assert out == [(0, 0, 2, 2), m, (5, 5, 7, 7)]

    def test_idempotent_on_own_output(self):
        m = MaskROI(selection=_block_selection())
        once = as_region_list([m, (0, 0, 2, 2)])
        assert as_region_list(once) == once

    def test_numpy_integer_bounds_coerced(self):
        out = as_region_list(np.array([1, 2, 3, 4]).tolist())
        assert all(type(v) is int for v in out[0])
        out2 = as_region_list([np.int64(1), np.int64(2), np.int64(3), np.int64(4)])
        assert all(type(v) is int for v in out2[0])

    def test_any_sequence_type_accepted(self):
        assert as_region_list(deque([1, 2, 3, 4])) == [(1, 2, 3, 4)]

    def test_bare_array_and_path_rejected(self):
        for bad in [np.zeros((6, 8), dtype=bool), "mask.tif", b"mask", Path("mask.tif"), 7]:
            with pytest.raises(ValueError, match="must be an ROI"):
                as_region_list(bad)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            as_region_list([])

    def test_malformed_int_sequence_raises(self):
        with pytest.raises(ValueError, match="4 integers"):
            as_region_list([1, 2, 3])
        with pytest.raises(ValueError, match="4 integers"):
            as_region_list([1, 2, 3, 4, 5])

    def test_arg_name_in_errors(self):
        with pytest.raises(ValueError, match="air_roi"):
            as_region_list("nope", arg_name="air_roi")


class TestLegacyHelpersHardened:
    def test_as_roi_bounds_rejects_mask_roi(self):
        with pytest.raises(ValueError, match="MaskROI"):
            as_roi_bounds(MaskROI(selection=_block_selection()))

    def test_as_roi_bounds_still_accepts_rects(self):
        assert as_roi_bounds(ROI(x0=1, y0=2, x1=3, y1=4)) == (1, 2, 3, 4)
        assert as_roi_bounds((1, 2, 3, 4)) == (1, 2, 3, 4)

    def test_as_roi_bounds_list_rejects_masks_bare_and_in_list(self):
        m = MaskROI(selection=_block_selection())
        with pytest.raises(ValueError, match="must be an ROI"):
            as_roi_bounds_list(m)  # not a Sequence -> legacy rejection path
        with pytest.raises(ValueError, match="MaskROI"):
            as_roi_bounds_list([m])  # element path -> hardened as_roi_bounds
