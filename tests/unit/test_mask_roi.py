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

    def test_non_dict_mapping_input_is_canonicalized(self):
        """Pydantic accepts any Mapping (e.g. UserDict), not just dict — it must canonicalize too."""
        from collections import UserDict

        sel = _block_selection().astype(np.int16) * 2  # non-bool; would miscount if not canonicalized
        m = MaskROI.model_validate(UserDict(selection=sel))
        assert m.selection.dtype == np.bool_  # coerced to bool
        assert not m.selection.flags.writeable  # owned read-only buffer
        assert m.n_selected == 6  # counts True pixels, not the summed int16 values
        # an empty selection supplied via a Mapping must still be rejected at construction
        with pytest.raises(Exception):
            MaskROI.model_validate(UserDict(selection=np.zeros((6, 8), dtype=bool)))


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


def _stack(seed=42, base=100, n=3, ny=6, nx=8, unit="counts"):
    rng = np.random.default_rng(seed)
    v = rng.poisson(base, size=(n, ny, nx)).astype("float64")
    return sc.DataArray(sc.array(dims=["N_image", "y", "x"], values=v, variances=v.copy(), unit=unit))


def _cross_selection(ny=6, nx=8):
    """Plus-shaped selection that provably differs from its bounding box (corners excluded)."""
    sel = np.zeros((ny, nx), dtype=bool)
    sel[2, 3:6] = True
    sel[1:4, 4] = True
    return sel


class TestPooledCoefficientMask:
    """The core #180 contract: a rectangle expressed as a MaskROI matches the sliced rect path."""

    def _rect_and_mask(self, ny=6, nx=8):
        sel = np.zeros((ny, nx), dtype=bool)
        sel[1:3, 2:5] = True
        return (2, 1, 5, 3), MaskROI(selection=sel)

    def test_normalize_transmission_rect_as_mask_matches(self):
        from neunorm.processing.normalizer import normalize_transmission

        s, o = _stack(1), _stack(2, base=200)
        rect, mask = self._rect_and_mask()
        t_rect = normalize_transmission(s, o, background_roi=rect)
        t_mask = normalize_transmission(s, o, background_roi=mask)
        np.testing.assert_allclose(t_mask.values, t_rect.values, rtol=1e-12)
        np.testing.assert_allclose(t_mask.variances, t_rect.variances, rtol=1e-12)

    def test_rect_as_mask_matches_with_dead_pixel_masks(self):
        from neunorm.processing.normalizer import normalize_transmission

        s, o = _stack(3), _stack(4, base=200)
        dead = np.zeros((6, 8), dtype=bool)
        dead[1, 2] = dead[2, 4] = True
        for da in (s, o):
            da.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
        rect, mask = self._rect_and_mask()
        t_rect = normalize_transmission(s, o, background_roi=rect)
        t_mask = normalize_transmission(s, o, background_roi=mask)
        np.testing.assert_allclose(t_mask.values, t_rect.values, rtol=1e-12)
        np.testing.assert_allclose(t_mask.variances, t_rect.variances, rtol=1e-12)

    def test_normalize_with_dark_rect_as_mask_matches(self):
        from neunorm.processing.normalizer import normalize_with_dark

        s, o = _stack(5), _stack(6, base=200)
        d = _stack(7, base=5)["N_image", 0].copy()
        rect, mask = self._rect_and_mask()
        t_rect = normalize_with_dark(s, o, d, background_roi=rect)
        t_mask = normalize_with_dark(s, o, d, background_roi=mask)
        np.testing.assert_allclose(t_mask.values, t_rect.values, rtol=1e-12)
        np.testing.assert_allclose(t_mask.variances, t_rect.variances, rtol=1e-12)

    def test_apply_background_roi_mixed_pooling_matches_two_rects(self):
        from neunorm.processing.normalizer import apply_background_roi

        s = _stack(8)
        rect2 = (0, 4, 2, 6)
        sel2 = np.zeros((6, 8), dtype=bool)
        sel2[4:6, 0:2] = True
        a_rects = apply_background_roi(s, [(2, 1, 5, 3), rect2])
        a_mixed = apply_background_roi(s, [(2, 1, 5, 3), MaskROI(selection=sel2)])
        np.testing.assert_allclose(a_mixed.values, a_rects.values, rtol=1e-12)
        np.testing.assert_allclose(a_mixed.variances, a_rects.variances, rtol=1e-12)

    def test_non_rectangular_selection_differs_from_bbox_and_matches_manual(self):
        from neunorm.processing.normalizer import apply_background_roi

        s = _stack(9)
        sel = _cross_selection()
        m = MaskROI(selection=sel)
        out = apply_background_roi(s, m)
        bbox_out = apply_background_roi(s, m.bounding_box())
        assert not np.allclose(out.values, bbox_out.values)  # corners excluded -> different mean
        # manual pooled mean over the selected pixels, per image
        coeff = s.values[:, sel].sum(axis=1) / sel.sum()
        expected = s.values / coeff[:, None, None]
        np.testing.assert_allclose(out.values, expected, rtol=1e-12)

    def test_mask_selection_pixels_all_dead_strict_raises_nonstrict_propagates(self):
        from neunorm.processing.normalizer import apply_background_roi

        s = _stack(10)
        sel = _cross_selection()
        s.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=sel.copy())  # kill exactly the selection
        with pytest.raises(ValueError, match="strictly positive"):
            apply_background_roi(s, MaskROI(selection=sel))
        out = apply_background_roi(s, MaskROI(selection=sel), strict=False)
        assert not np.isfinite(out.values).any()  # 0/0 propagates, legacy rect parity

    def test_nan_outside_selection_does_not_leak(self):
        """Pixels outside the selection may hold NaN/inf without affecting the coefficient."""
        from neunorm.processing.normalizer import apply_background_roi

        s = _stack(11)
        sel = _cross_selection()
        ref = apply_background_roi(s, MaskROI(selection=sel)).copy()
        vals = s.values.copy()
        vals[:, 0, 0] = np.nan  # outside the selection AND outside its bbox
        vals[:, 1, 5] = np.inf  # inside the bbox, outside the selection
        s2 = sc.DataArray(
            sc.array(dims=["N_image", "y", "x"], values=vals, variances=s.variances.copy(), unit="counts")
        )
        out = apply_background_roi(s2, MaskROI(selection=sel))
        finite = np.isfinite(out.values)
        np.testing.assert_allclose(out.values[finite], ref.values[finite], rtol=1e-12)
        # the coefficient itself was unaffected: all selected pixels stay finite
        assert np.isfinite(out.values[:, sel]).all()

    def test_shape_mismatch_raises_valueerror(self):
        from neunorm.processing.normalizer import normalize_transmission

        s, o = _stack(12), _stack(13, base=200)
        with pytest.raises(ValueError, match="does not match"):
            normalize_transmission(s, o, background_roi=MaskROI(selection=np.ones((4, 4), dtype=bool)))

    def test_user_mask_named_region_sel_not_clobbered(self):
        from neunorm.processing.normalizer import apply_background_roi

        s = _stack(14)
        # a user mask that would collide with the internal name: masks one selected pixel
        user = np.zeros((6, 8), dtype=bool)
        user[2, 4] = True
        s.masks["_region_sel"] = sc.array(dims=["y", "x"], values=user)
        sel = _cross_selection()
        out = apply_background_roi(s, MaskROI(selection=sel))
        # expected: pooled mean over selected & ~user (the user mask must stay in force)
        keep = sel & ~user
        coeff = s.values[:, keep].sum(axis=1) / keep.sum()
        np.testing.assert_allclose(out.values, s.values / coeff[:, None, None], rtol=1e-12)


class TestSharedDarkCovarianceMask:
    def test_covariance_matches_hand_computed_oracle(self):
        """Cov = sum Var(D) over (sel & unmasked_s & unmasked_o) / (n_s * n_o), per design."""
        from neunorm.processing.dark_corrector import subtract_dark
        from neunorm.processing.normalizer import _roi_dark_mean_covariance

        s, o = _stack(15), _stack(16, base=200)
        d = _stack(17, base=5)["N_image", 0].copy()
        dead_s = np.zeros((6, 8), dtype=bool)
        dead_s[2, 4] = True
        s.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead_s)
        sel = _cross_selection()
        s_dc, o_dc = subtract_dark(s, d), subtract_dark(o, d)
        cov = _roi_dark_mean_covariance(s_dc, o_dc, d, [MaskROI(selection=sel)])
        keep_s = sel & ~dead_s
        keep_o = sel
        n_s, n_o = keep_s.sum(), keep_o.sum()
        expected = d.variances[keep_s & keep_o].sum() / (n_s * n_o)
        np.testing.assert_allclose(float(sc.values(cov).min().value), expected, rtol=1e-12)
        np.testing.assert_allclose(float(sc.values(cov).max().value), expected, rtol=1e-12)

    def test_covariance_rect_as_mask_matches_rect(self):
        from neunorm.processing.dark_corrector import subtract_dark
        from neunorm.processing.normalizer import _roi_dark_mean_covariance

        s, o = _stack(18), _stack(19, base=200)
        d = _stack(20, base=5)["N_image", 0].copy()
        s_dc, o_dc = subtract_dark(s, d), subtract_dark(o, d)
        sel = np.zeros((6, 8), dtype=bool)
        sel[1:3, 2:5] = True
        cov_rect = _roi_dark_mean_covariance(s_dc, o_dc, d, [(2, 1, 5, 3)])
        cov_mask = _roi_dark_mean_covariance(s_dc, o_dc, d, [MaskROI(selection=sel)])
        # The rect path returns a constant per-spectral vector, the mask fast path a scalar — both
        # broadcast identically downstream (normalize_with_dark equivalence pins that end to end);
        # compare the broadcast values.
        rect_vals = np.atleast_1d(sc.values(cov_rect).values)
        mask_vals = np.broadcast_to(np.atleast_1d(sc.values(cov_mask).values), rect_vals.shape)
        np.testing.assert_allclose(mask_vals, rect_vals, rtol=1e-12)


class TestAirRegionUnified:
    def test_rect_parity_with_old_sequential_math_unmasked(self):
        """On unmasked data the pooled mean equals the old sc.mean-based path to allclose."""
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(21, base=50, unit="dimensionless")
        rect = (2, 1, 5, 3)
        out = apply_air_region_correction(t, rect)
        # old math, computed manually: T / mean(T[air]); var = corrected^2*(Var(T)/T^2 + Var(m)/m^2)
        x0, y0, x1, y1 = rect
        air = t.values[:, y0:y1, x0:x1]
        mean_air = air.mean(axis=(1, 2))
        var_mean = t.variances[:, y0:y1, x0:x1].sum(axis=(1, 2)) / air[0].size ** 2
        expected = t.values / mean_air[:, None, None]
        exp_var = expected**2 * (t.variances / t.values**2 + (var_mean / mean_air**2)[:, None, None])
        np.testing.assert_allclose(out.values, expected, rtol=1e-12)
        np.testing.assert_allclose(out.variances, exp_var, rtol=1e-12)

    def test_t_zero_pixel_variance_now_finite(self):
        """Old formula gave 0*inf = NaN variance at T==0 pixels; pooled path stays finite."""
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(22, base=50, unit="dimensionless")
        vals = t.values.copy()
        vals[:, 0, 0] = 0.0  # outside the air region
        t2 = sc.DataArray(
            sc.array(dims=["N_image", "y", "x"], values=vals, variances=t.variances.copy(), unit="dimensionless")
        )
        out = apply_air_region_correction(t2, (2, 1, 5, 3))
        assert np.isfinite(out.variances[:, 0, 0]).all()

    def test_mask_region_true_region_mean_under_dead_pixels(self):
        """With a dead pixel in the air region the pooled mean is the true region mean."""
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(23, base=50, unit="dimensionless")
        dead = np.zeros((6, 8), dtype=bool)
        dead[2, 4] = True
        t.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
        sel = _cross_selection()
        out = apply_air_region_correction(t, MaskROI(selection=sel))
        keep = sel & ~dead
        coeff = t.values[:, keep].sum(axis=1) / keep.sum()
        np.testing.assert_allclose(out.values, t.values / coeff[:, None, None], rtol=1e-12)

    def test_fully_masked_row_no_longer_nans_the_region(self):
        """A fully-dead row inside the air rect used to NaN the sequential mean-of-row-means."""
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(24, base=50, unit="dimensionless")
        dead = np.zeros((6, 8), dtype=bool)
        dead[1, 2:5] = True  # kills the entire first row of the (2,1,5,3) rect
        t.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
        out = apply_air_region_correction(t, (2, 1, 5, 3))
        assert np.isfinite(out.values).all()

    def test_nonpositive_air_mean_raises_with_air_roi_message(self):
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(25, base=50, unit="dimensionless")
        vals = t.values.copy()
        vals[:, 1:3, 2:5] = 0.0
        t2 = sc.DataArray(
            sc.array(dims=["N_image", "y", "x"], values=vals, variances=t.variances.copy(), unit="dimensionless")
        )
        with pytest.raises(ValueError, match="air_roi.*strictly positive"):
            apply_air_region_correction(t2, (2, 1, 5, 3))

    def test_pooled_multiple_air_regions(self):
        from neunorm.processing.air_region_corrector import apply_air_region_correction

        t = _stack(26, base=50, unit="dimensionless")
        out = apply_air_region_correction(t, [(2, 1, 5, 3), (0, 4, 2, 6)])
        sel = np.zeros((6, 8), dtype=bool)
        sel[1:3, 2:5] = True
        sel[4:6, 0:2] = True
        coeff = t.values[:, sel].sum(axis=1) / sel.sum()
        np.testing.assert_allclose(out.values, t.values / coeff[:, None, None], rtol=1e-12)


class TestApplyRoiCrop:
    def test_rect_crop_unchanged(self):
        from neunorm.processing.roi_clipper import apply_roi

        s = _stack(27)
        out = apply_roi(s, (2, 1, 5, 3))
        assert out.sizes["x"] == 3 and out.sizes["y"] == 2
        assert len(out.masks) == 0

    def test_crop_rejects_maskroi(self):
        """Crop is rectangle-only; a MaskROI must be routed to the region-statistics APIs (#180)."""
        from neunorm.processing.roi_clipper import apply_roi

        with pytest.raises(TypeError, match="does not accept a MaskROI"):
            apply_roi(_stack(28), MaskROI(selection=_cross_selection()))


class TestPooledOverlapDedup:
    """Overlapping pooled regions count each shared pixel once — correct mean AND variance (F3)."""

    def _frame(self):
        vals = np.arange(1, 49, dtype="float64").reshape(6, 8)
        return sc.DataArray(sc.array(dims=["y", "x"], values=vals.copy(), variances=vals.copy(), unit="counts"))

    def _union_oracle(self, regions, ny=6, nx=8):
        vals = np.arange(1, 49, dtype="float64").reshape(ny, nx)
        union = np.zeros((ny, nx), dtype=bool)
        for r in regions:
            if isinstance(r, MaskROI):
                union |= r.selection
            else:
                x0, y0, x1, y1 = r
                union[y0:y1, x0:x1] = True
        v = vals[union]
        n = v.size
        return v.sum() / n, v.sum() / n**2  # mean, Var(mean) with variances == values

    def test_overlapping_rects_dedup_to_union(self):
        from neunorm.processing.normalizer import _pooled_roi_coefficient

        rois = [(0, 0, 5, 4), (3, 2, 8, 6)]  # overlap on x 3-4, y 2-3
        coeff = _pooled_roi_coefficient(self._frame(), rois, "data", strict=True)
        mean, var = self._union_oracle(rois)
        np.testing.assert_allclose(coeff.value, mean, rtol=1e-12)
        np.testing.assert_allclose(coeff.variance, var, rtol=1e-12)

    def test_overlapping_rect_and_mask_dedup_to_union(self):
        from neunorm.processing.normalizer import _pooled_roi_coefficient

        sel = np.zeros((6, 8), dtype=bool)
        sel[1:5, 2:6] = True
        rois = [(0, 0, 4, 3), MaskROI(selection=sel)]  # rect and mask share pixels
        coeff = _pooled_roi_coefficient(self._frame(), rois, "data", strict=True)
        mean, var = self._union_oracle(rois)
        np.testing.assert_allclose(coeff.value, mean, rtol=1e-12)
        np.testing.assert_allclose(coeff.variance, var, rtol=1e-12)

    def test_duplicate_region_equals_single_not_halved(self):
        """Listing a region twice must equal listing it once — the old code halved the variance."""
        from neunorm.processing.normalizer import _pooled_roi_coefficient

        rect = (1, 1, 5, 4)
        single = _pooled_roi_coefficient(self._frame(), [rect], "data", strict=True)
        dup = _pooled_roi_coefficient(self._frame(), [rect, rect], "data", strict=True)
        np.testing.assert_allclose(dup.value, single.value, rtol=1e-12)
        np.testing.assert_allclose(dup.variance, single.variance, rtol=1e-12)

    def test_disjoint_pooled_keeps_fast_path(self):
        """Non-overlapping pooled regions are unchanged (values AND variances) — union oracle."""
        from neunorm.processing.normalizer import _pooled_roi_coefficient

        rois = [(0, 0, 2, 2), (5, 4, 8, 6)]
        coeff = _pooled_roi_coefficient(self._frame(), rois, "data", strict=True)
        mean, var = self._union_oracle(rois)
        np.testing.assert_allclose(coeff.value, mean, rtol=1e-12)
        np.testing.assert_allclose(coeff.variance, var, rtol=1e-12)

    def test_covariance_duplicate_region_equals_single(self):
        """Shared-dark covariance must also dedup: [R, R] == [R] (was double-counted)."""
        from neunorm.processing.dark_corrector import subtract_dark
        from neunorm.processing.normalizer import _roi_dark_mean_covariance

        s, o = _stack(40), _stack(41, base=200)
        d = _stack(42, base=5)["N_image", 0].copy()
        s_dc, o_dc = subtract_dark(s, d), subtract_dark(o, d)
        rect = (2, 1, 6, 4)
        cov1 = _roi_dark_mean_covariance(s_dc, o_dc, d, [rect])
        cov2 = _roi_dark_mean_covariance(s_dc, o_dc, d, [rect, rect])
        v1 = np.atleast_1d(sc.values(cov1).values)
        v2 = np.broadcast_to(np.atleast_1d(sc.values(cov2).values), v1.shape)
        np.testing.assert_allclose(v2, v1, rtol=1e-12)

    def test_disjoint_and_single_do_not_take_union_path(self, monkeypatch):
        """Pin the fast-path guard directly (a union-oracle match alone can't catch 'always union')."""
        import neunorm.processing.normalizer as nz

        def _boom(*_a, **_k):
            raise AssertionError("union recompute taken for a non-overlapping pool")

        monkeypatch.setattr(nz, "_pooled_union_selection", _boom)
        nz._pooled_roi_coefficient(self._frame(), [(1, 1, 4, 4)], "data", strict=True)  # single
        nz._pooled_roi_coefficient(self._frame(), [(0, 0, 2, 2), (5, 4, 8, 6)], "data", strict=True)  # disjoint
        monkeypatch.undo()
        # and the guard positively fires only on real overlap
        assert nz._pooled_regions_overlap([(0, 0, 5, 4), (3, 2, 8, 6)], 6, 8) is True
        assert nz._pooled_regions_overlap([(0, 0, 2, 2), (5, 4, 8, 6)], 6, 8) is False

    def test_covariance_partial_overlap_matches_union_oracle(self):
        """Partial (non-duplicate) overlap: covariance counts each shared dark pixel once (union)."""
        from neunorm.processing.dark_corrector import subtract_dark
        from neunorm.processing.normalizer import _roi_dark_mean_covariance

        s, o = _stack(50), _stack(51, base=200)
        d = _stack(52, base=5)["N_image", 0].copy()
        dead = np.zeros((6, 8), dtype=bool)
        dead[2, 3] = True  # a dead sample pixel inside the overlap
        s.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
        s_dc, o_dc = subtract_dark(s, d), subtract_dark(o, d)
        rois = [(0, 0, 5, 4), (3, 2, 8, 6)]  # partial overlap on x 3-4, y 2-3
        cov = _roi_dark_mean_covariance(s_dc, o_dc, d, rois)
        union = np.zeros((6, 8), dtype=bool)
        for x0, y0, x1, y1 in rois:
            union[y0:y1, x0:x1] = True
        keep_s, keep_o = union & ~dead, union  # A (sample), B (ob)
        n_s, n_o = keep_s.sum(), keep_o.sum()
        expected = d.variances[keep_s & keep_o].sum() / (n_s * n_o)
        got = np.atleast_1d(sc.values(cov).values)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
