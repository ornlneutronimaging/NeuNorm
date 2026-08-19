"""Uncertainty correctness for the ROI transmission spectrum (the "resonance mode" reduction).

Three things are pinned here. Two are confirmations; the first is a **finding**, and it is written
down as behaviour rather than repaired, because repairing it means changing the propagation and that
is the maintainer's call.

**1. The shared-dark covariance is NOT removed from a spectrum's variance.** When the same averaged
dark frame is subtracted from both stacks before the region is collapsed, the sample and open-beam
region means share it, so they are positively correlated::

    Cov(Sbar, Obar) = Var(Dbar over the region) = sum(Var(D) over the region) / n**2

and the honest variance of their ratio is

    Var(T) / T**2 = Var(Sbar)/Sbar**2 + Var(Obar)/Obar**2 - 2 Cov / (Sbar Obar)

``normalize_roi_spectrum`` divides through :func:`neunorm.processing.normalizer.normalize_transmission`,
which has no dark-covariance term, so what it emits is the first two terms alone — the **independent**
form. The spectrum therefore OVERSTATES its uncertainty. The tests below pin the size of that
overstatement (about 11% of the variance, 6% of the 1-sigma bar, for a warm dark) instead of asserting
a correctness that is not there, and one of them shows the pixel-level path
(:func:`neunorm.processing.normalizer.normalize_with_dark`) already removing exactly the term the
spectrum path keeps. If the correction is ever added here, these tests fail on purpose: the expected
value moves from the independent form to the covariance-corrected one, both of which each test
computes with plain numpy.

**2. The zero-open-beam policy**, on both settings. ``spectrum_roi_strict=True`` refuses a bin whose
open-beam region mean is not strictly positive and finite; ``spectrum_roi_strict=False`` lets it
propagate, which is the legacy 1.x behaviour. The numerator is deliberately unguarded on either
setting, because a fully absorbing bin is a measurement and not a fault.

**3. The exported uncertainty column is 1 sigma** of the returned spectrum, and the variance scales
with counting statistics as Poisson statistics require.
"""

import numpy as np
import pytest
import scipp as sc

from neunorm.exporters.ascii_writer import ASCII_SPECTRUM_HEADER, write_ascii_spectrum
from neunorm.processing.dark_corrector import subtract_dark
from neunorm.processing.normalizer import normalize_with_dark
from neunorm.processing.spectrum_reducer import normalize_roi_spectrum

#: x 1:5, y 1:4 with exclusive stops — a 4 x 3 = 12 pixel region, offset from the frame edge so a
#: reduction that quietly used the whole frame would be caught by the values, not just the shapes.
_ROI = (1, 1, 5, 4)
_ROI_PIXELS = 12


def _poisson(values, dims, unit="counts"):
    """A counts DataArray whose variance equals its value — the Poisson case the pipelines produce."""
    values = np.asarray(values, dtype=float)
    return sc.DataArray(sc.array(dims=list(dims), values=values.copy(), variances=values.copy(), unit=unit))


def _stack(values, *, mask=None):
    """A ``(tof, y, x)`` Poisson stack carrying N+1 tof bin edges, optionally a spatial mask."""
    data = _poisson(values, ["tof", "y", "x"])
    n_bins = np.shape(values)[0]
    data.coords["tof"] = sc.array(dims=["tof"], values=np.arange(n_bins + 1, dtype=float) * 1.0e-3, unit="s")
    if mask is not None:
        data.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=np.asarray(mask, dtype=bool))
    return data


def _dark_corrected(s_raw, o_raw, d_raw, *, sample_mask=None):
    """The two dark-subtracted stacks a pipeline would hand the reduction, plus the dark itself."""
    dark = _poisson(d_raw, ["y", "x"])
    sample = subtract_dark(_stack(s_raw, mask=sample_mask), dark)
    ob = subtract_dark(_stack(o_raw), dark)
    return sample, ob, dark


def _analytic(s_raw, o_raw, d_raw, *, keep=None):
    """Both variance forms for the ratio of two dark-correlated region means, by plain numpy.

    Returns ``(T, var_independent, var_covariance_corrected, cov_fraction_of_var)`` over ``_ROI``,
    where ``keep`` is the boolean ``(y, x)`` selection of region pixels that survive masking.
    ``var_independent`` is what treating the two means as uncorrelated gives; the corrected form
    subtracts ``2 Cov / (Sbar Obar)`` from the relative variance.
    """
    x0, y0, x1, y1 = _ROI
    s_box = np.asarray(s_raw, dtype=float)[:, y0:y1, x0:x1]
    o_box = np.asarray(o_raw, dtype=float)[:, y0:y1, x0:x1]
    d_box = np.broadcast_to(np.asarray(d_raw, dtype=float)[y0:y1, x0:x1], s_box.shape)
    # subtract_dark clips negatives to zero; an oracle built on the unclipped difference is only the
    # same arithmetic while the dark is strictly smaller than both stacks.
    assert np.all(s_box > d_box) and np.all(o_box > d_box)
    if keep is None:
        keep = np.ones(s_box.shape[1:], dtype=bool)
    n = int(np.count_nonzero(keep))
    sbar = (s_box - d_box)[:, keep].sum(axis=1) / n
    obar = (o_box - d_box)[:, keep].sum(axis=1) / n
    # Var(S - D) = Var(S) + Var(D) per pixel, and Var(mean) = sum(Var) / n**2.
    var_sbar = (s_box + d_box)[:, keep].sum(axis=1) / n**2
    var_obar = (o_box + d_box)[:, keep].sum(axis=1) / n**2
    cov = d_box[:, keep].sum(axis=1) / n**2
    transmission = sbar / obar
    rel_independent = var_sbar / sbar**2 + var_obar / obar**2
    cov_term = 2.0 * cov / (sbar * obar)
    return (
        transmission,
        transmission**2 * rel_independent,
        transmission**2 * (rel_independent - cov_term),
        cov_term / rel_independent,
    )


class TestSharedDarkCovariance:
    """What the spectrum's variance is when sample and open beam share a dark frame."""

    def test_variance_is_the_independent_form_not_the_covariance_corrected_one(self):
        """The emitted variance matches the UNCORRELATED analytic form, to the last bit.

        Both references are computed from the raw counts with numpy. The code lands exactly on the
        independent one, which is the finding: the shared dark makes Sbar and Obar correlated, and
        that correlation is not subtracted, so the reported variance is too large. Should the
        correction ever be added, this test fails and the expected value is the second reference the
        assertions below already carry.
        """
        n_bins, ny, nx = 3, 6, 6
        s_raw = np.full((n_bins, ny, nx), 500.0)
        o_raw = np.full((n_bins, ny, nx), 1000.0)
        d_raw = np.full((ny, nx), 100.0)
        sample, ob, _ = _dark_corrected(s_raw, o_raw, d_raw)

        spectrum = normalize_roi_spectrum(sample, ob, _ROI)

        t, var_independent, var_corrected, _ = _analytic(s_raw, o_raw, d_raw)
        # Uniform counts, so the whole reference is hand-checkable: Sbar = 400, Obar = 900,
        # Var(Sbar) = (500 + 100)/12 = 50, Var(Obar) = (1000 + 100)/12 = 1100/12, Cov = 100/12.
        np.testing.assert_allclose(t, 400.0 / 900.0, rtol=1e-14)
        np.testing.assert_allclose(var_independent, 8.408271096885638e-05, rtol=1e-12)
        np.testing.assert_allclose(var_corrected, 7.493776355230402e-05, rtol=1e-12)

        np.testing.assert_allclose(spectrum.values, t, rtol=1e-13)
        np.testing.assert_allclose(spectrum.variances, var_independent, rtol=1e-13)
        # The gap is real, not a rounding difference: the two references differ by far more than the
        # tolerance the assertion above allows.
        assert np.all(var_corrected < var_independent * 0.95)

    def test_covariance_term_is_over_ten_percent_of_the_variance_for_a_warm_dark(self):
        """The uncorrected covariance is not a rounding-level omission — it is worth about 6% of sigma.

        500-count sample, 1000-count open beam, a 100-count dark (a long CCD exposure, or a warm
        detector) over a 12-pixel region. Quantified so the finding is actionable rather than
        theoretical: the reported variance is ~10.9% high and the reported 1-sigma bar ~5.9% high, in
        the same direction for every bin, which is a systematic overstatement of the error bars a
        resonance fit is weighted by.
        """
        n_bins, ny, nx = 2, 6, 6
        s_raw = np.full((n_bins, ny, nx), 500.0)
        o_raw = np.full((n_bins, ny, nx), 1000.0)
        d_raw = np.full((ny, nx), 100.0)
        sample, ob, _ = _dark_corrected(s_raw, o_raw, d_raw)

        spectrum = normalize_roi_spectrum(sample, ob, _ROI)

        _, var_independent, var_corrected, cov_fraction = _analytic(s_raw, o_raw, d_raw)
        np.testing.assert_allclose(spectrum.variances, var_independent, rtol=1e-13)

        variance_overstatement = 100.0 * (spectrum.variances / var_corrected - 1.0)
        sigma_overstatement = 100.0 * (np.sqrt(spectrum.variances / var_corrected) - 1.0)
        np.testing.assert_allclose(100.0 * cov_fraction, 10.876132930513595, rtol=1e-10)
        np.testing.assert_allclose(variance_overstatement, 12.203389830508481, rtol=1e-8)
        np.testing.assert_allclose(sigma_overstatement, 5.926101519176319, rtol=1e-8)
        # Guard the claim in this file's header rather than only the exact numbers above: a dark this
        # warm costs more than 5% of the variance whatever the surrounding values are.
        assert np.all(cov_fraction > 0.05)

    def test_one_pixel_region_overstates_by_exactly_normalize_with_dark_s_correction(self):
        """Over a 1x1 region the two code paths are directly comparable, and they disagree.

        A one-pixel region mean IS the pixel, so ``normalize_roi_spectrum`` on dark-subtracted stacks
        and ``normalize_with_dark`` on the raw ones compute the same transmission from the same
        numbers. Their variances differ by exactly ``2 * (sample - dark) * Var(dark) / (ob - dark)**3``
        — the shared-dark term ``normalize_with_dark`` documents and removes. That identity is what
        makes this a gap in the spectrum path specifically, and not a disagreement about the physics:
        the correction is already implemented one module away.
        """
        n_bins = 2
        s_raw = np.full((n_bins, 1, 1), 500.0)
        o_raw = np.full((n_bins, 1, 1), 1000.0)
        d_raw = np.full((1, 1), 100.0)
        dark = _poisson(d_raw, ["y", "x"])

        pixel = normalize_with_dark(_stack(s_raw), _stack(o_raw), dark)
        spectrum = normalize_roi_spectrum(
            subtract_dark(_stack(s_raw), dark), subtract_dark(_stack(o_raw), dark), (0, 0, 1, 1)
        )

        s, o, var_d = 500.0 - 100.0, 1000.0 - 100.0, 100.0
        correction = 2.0 * s * var_d / o**3
        np.testing.assert_allclose(correction, 1.0973936899862826e-04, rtol=1e-12)

        np.testing.assert_allclose(spectrum.values, pixel.values.ravel(), rtol=1e-13)
        np.testing.assert_allclose(spectrum.variances - pixel.variances.ravel(), correction, rtol=1e-9)
        # And the pixel path's own variance is the covariance-corrected form, so the direction is not
        # ambiguous: the spectrum is the one that is too large.
        assert np.all(pixel.variances.ravel() < spectrum.variances)

    def test_masked_nonuniform_region_still_lands_on_the_independent_form(self):
        """The overstatement is not an artefact of uniform counts or of an unmasked region.

        Varied counts, a varied dark, and two dead pixels masked on the sample only. Mask
        symmetrization gives both sides the same exclusions, so the analytic reference divides both
        means by the same 10 surviving pixels; without symmetrization the open beam would average over
        12 and the values below would not match either.
        """
        n_bins, ny, nx = 3, 5, 5
        rng = np.random.default_rng(11)
        s_raw = rng.integers(300, 600, size=(n_bins, ny, nx)).astype(float)
        o_raw = rng.integers(900, 1200, size=(n_bins, ny, nx)).astype(float)
        d_raw = rng.integers(60, 140, size=(ny, nx)).astype(float)
        dead = np.zeros((ny, nx), dtype=bool)
        dead[1, 2] = True  # inside _ROI
        dead[3, 3] = True  # inside _ROI
        sample, ob, _ = _dark_corrected(s_raw, o_raw, d_raw, sample_mask=dead)

        spectrum = normalize_roi_spectrum(sample, ob, _ROI)

        x0, y0, x1, y1 = _ROI
        keep = ~dead[y0:y1, x0:x1]
        assert int(np.count_nonzero(keep)) == _ROI_PIXELS - 2
        t, var_independent, var_corrected, cov_fraction = _analytic(s_raw, o_raw, d_raw, keep=keep)

        np.testing.assert_allclose(spectrum.values, t, rtol=1e-13)
        np.testing.assert_allclose(spectrum.variances, var_independent, rtol=1e-13)
        assert np.all(cov_fraction > 0.05)
        assert np.all(var_corrected < var_independent)


class TestZeroOpenBeamPolicy:
    """A bin whose open-beam region mean is zero, on the strict and the legacy setting."""

    @staticmethod
    def _pair_with_a_dead_bin(n_bins=3, ny=6, nx=6, sample_counts=400.0):
        """Poisson stacks whose open beam recorded nothing at all in bin 1."""
        s_raw = np.full((n_bins, ny, nx), sample_counts)
        o_raw = np.full((n_bins, ny, nx), 800.0)
        o_raw[1] = 0.0
        return _stack(s_raw), _stack(o_raw)

    def test_strict_raises_and_the_message_names_the_open_beam_and_the_argument(self):
        """Strict is the default, and its message has to say which side and which argument failed.

        The denominator's region mean is guarded because a zero there is a fault, not a measurement.
        The text names the offending side as ``ob`` and the argument as ``spectrum_roi``, so a user who
        passed several regions knows what to look at; the min it reports is the offending value.
        """
        sample, ob = self._pair_with_a_dead_bin()

        with pytest.raises(ValueError, match=r"spectrum_roi ob pooled mean must be strictly positive and finite"):
            normalize_roi_spectrum(sample, ob, _ROI)

        with pytest.raises(ValueError) as excinfo:
            normalize_roi_spectrum(sample, ob, _ROI, spectrum_roi_strict=True)
        message = str(excinfo.value)
        assert "ob" in message
        assert "spectrum_roi" in message
        assert "min=0.0" in message

    def test_legacy_propagates_an_infinite_value_and_a_nan_variance(self):
        """``spectrum_roi_strict=False`` reproduces 1.x: the bad bin passes through, unflagged.

        The surviving bins are unaffected and keep their hand-computed values, so opting out is not
        a global loss of precision — it is one poisoned point in an otherwise usable spectrum. The
        variance of that point comes out **NaN**, not inf: Var(Sbar)/0 is inf while
        Sbar**2 * 0 / 0 is NaN, and inf + NaN is NaN. So the legacy setting emits a point whose value
        says "infinite transmission" and whose error bar says nothing at all.
        """
        sample, ob = self._pair_with_a_dead_bin()

        spectrum = normalize_roi_spectrum(sample, ob, _ROI, spectrum_roi_strict=False)

        assert np.isinf(spectrum.values[1])
        assert np.isnan(spectrum.variances[1])
        # good bins: T = 400/800, Var(T)/T**2 = (400/12)/400**2 + (800/12)/800**2
        rel_var = (400.0 / 12.0) / 400.0**2 + (800.0 / 12.0) / 800.0**2
        good = np.array([0, 2])
        np.testing.assert_allclose(spectrum.values[good], 0.5, rtol=1e-14)
        np.testing.assert_allclose(spectrum.variances[good], 0.25 * rel_var, rtol=1e-13)

    def test_legacy_zero_on_both_sides_gives_a_nan_value(self):
        """Zero over zero is NaN rather than inf — the two failure modes are distinguishable.

        Worth pinning separately because a downstream reader that filters on ``isinf`` alone would
        keep this point.
        """
        sample, ob = self._pair_with_a_dead_bin()
        sample.values[1] = 0.0
        sample.variances[1] = 0.0

        spectrum = normalize_roi_spectrum(sample, ob, _ROI, spectrum_roi_strict=False)

        assert np.isnan(spectrum.values[1])
        assert np.isnan(spectrum.variances[1])

    def test_a_zero_sample_region_mean_is_a_measurement_even_under_strict(self):
        """The numerator is never guarded: a fully absorbing bin must give T = 0, not an exception.

        The sample is dark-subtracted down to zero counts over the region (a black resonance), while
        the open beam is healthy. Strict still returns, the transmission is exactly 0, and the
        uncertainty is the one the dark left behind: Var(T) = Var(Sbar) / Obar**2 with
        Var(Sbar) = sum(Var(D) over the region) / n**2, since the clipped zero still carries the dark's
        variance. A guard applied to both sides would turn this measurement into a crash.
        """
        n_bins, ny, nx = 2, 6, 6
        d_raw = np.full((ny, nx), 25.0)
        sample, ob, _ = _dark_corrected(np.zeros((n_bins, ny, nx)) + 25.0, np.full((n_bins, ny, nx), 1000.0), d_raw)

        spectrum = normalize_roi_spectrum(sample, ob, _ROI, spectrum_roi_strict=True)

        obar = 1000.0 - 25.0
        var_sbar = 2.0 * 25.0 * _ROI_PIXELS / _ROI_PIXELS**2  # Var(S) + Var(D) = 25 + 25 per pixel
        np.testing.assert_allclose(spectrum.values, 0.0, atol=0.0)
        np.testing.assert_allclose(spectrum.variances, var_sbar / obar**2, rtol=1e-13)

    def test_a_partly_dead_open_beam_region_does_not_raise_and_unmasked_zeros_still_count(self):
        """Zero pixels are data; only masked pixels leave the denominator.

        Two open-beam pixels inside the region read zero and are NOT masked, and one further pixel is
        masked on the sample. The guard looks at the region MEAN, which is still positive, so strict
        passes. The pooled mean is ``sum / count(unmasked)``: the masked pixel leaves both the sum and
        the count, the two dead-but-unmasked pixels leave only the sum. Dropping the zeros from the
        count instead would read 800 where the truth is 693.33 and bias every transmission low.
        """
        n_bins, ny, nx = 3, 4, 4
        s_raw = np.full((n_bins, ny, nx), 400.0)
        o_raw = np.full((n_bins, ny, nx), 800.0)
        o_raw[:, 0, 0] = 0.0
        o_raw[:, 1, 2] = 0.0
        dead = np.zeros((ny, nx), dtype=bool)
        dead[3, 3] = True
        roi = (0, 0, 4, 4)

        spectrum = normalize_roi_spectrum(_stack(s_raw, mask=dead), _stack(o_raw), roi, spectrum_roi_strict=True)

        n = ny * nx - 1  # 15 unmasked pixels, 13 of them with counts
        obar = 13 * 800.0 / n
        sbar = 400.0
        np.testing.assert_allclose(obar, 693.3333333333334, rtol=1e-13)
        np.testing.assert_allclose(spectrum.values, sbar / obar, rtol=1e-13)
        var_sbar = n * 400.0 / n**2
        var_obar = 13 * 800.0 / n**2
        rel_var = var_sbar / sbar**2 + var_obar / obar**2
        np.testing.assert_allclose(spectrum.variances, (sbar / obar) ** 2 * rel_var, rtol=1e-13)


class TestExportedUncertaintyColumn:
    """What the third ASCII column contains, given the spectrum it was written from."""

    @staticmethod
    def _spectrum():
        """A healthy four-bin dark-corrected spectrum, uncertainty around 1e-2."""
        n_bins, ny, nx = 4, 6, 6
        rng = np.random.default_rng(3)
        s_raw = rng.integers(300, 600, size=(n_bins, ny, nx)).astype(float)
        o_raw = rng.integers(900, 1200, size=(n_bins, ny, nx)).astype(float)
        d_raw = rng.integers(60, 140, size=(ny, nx)).astype(float)
        sample, ob, _ = _dark_corrected(s_raw, o_raw, d_raw)
        return normalize_roi_spectrum(sample, ob, _ROI)

    def test_third_column_is_one_sigma_of_the_returned_spectrum(self, tmp_path):
        """The file carries sqrt(variance), not the variance — a squared column would read as tiny.

        Compared against ``np.sqrt(spectrum.variances)`` computed here, at the six decimals the writer
        formats, so both the sqrt and the column order are pinned. The header is the one plain line
        the ecosystem's readers expect, which is what lets ``skiprows=1`` find the first data row.
        """
        spectrum = self._spectrum()
        path = tmp_path / "spectrum.txt"

        write_ascii_spectrum(path, spectrum)

        lines = path.read_text().splitlines()
        assert lines[0] == ASCII_SPECTRUM_HEADER == "bin_index,transmission,uncertainty"
        table = np.loadtxt(path, skiprows=1, delimiter=",", ndmin=2)
        assert table.shape == (spectrum.sizes["tof"], 3)
        np.testing.assert_array_equal(table[:, 0].astype(int), np.arange(spectrum.sizes["tof"]))
        # %.6f rounds to the nearest 1e-6, so the residual is bounded absolutely, not relatively.
        np.testing.assert_allclose(table[:, 1], spectrum.values, atol=5.1e-7, rtol=0.0)
        one_sigma = np.sqrt(spectrum.variances)
        np.testing.assert_allclose(table[:, 2], one_sigma, atol=5.1e-7, rtol=0.0)
        # A sanity floor: sigma here is ~1e-2, so writing the variance instead would be ~1e-4 and the
        # comparison above would fail by two orders of magnitude rather than by rounding.
        assert np.all(one_sigma > 1e-3)

    def test_a_legacy_nonfinite_variance_reaches_the_file_as_a_literal_nan(self, tmp_path):
        """``spectrum_roi_strict=False`` writes ``nan`` into the uncertainty column, unflagged.

        The consequence of the legacy policy at the point where it becomes someone else's problem: the
        writer neither rejects nor annotates the point, and the row parses back as NaN. Pinned so the
        cost of opting out is visible in the artefact and not only in the array.
        """
        n_bins, ny, nx = 3, 6, 6
        s_raw = np.full((n_bins, ny, nx), 400.0)
        o_raw = np.full((n_bins, ny, nx), 800.0)
        o_raw[1] = 0.0
        spectrum = normalize_roi_spectrum(_stack(s_raw), _stack(o_raw), _ROI, spectrum_roi_strict=False)
        path = tmp_path / "legacy.txt"

        write_ascii_spectrum(path, spectrum)

        rows = path.read_text().splitlines()[1:]
        assert rows[1] == "1,inf,nan"
        table = np.loadtxt(path, skiprows=1, delimiter=",", ndmin=2)
        assert np.isinf(table[1, 1])
        assert np.isnan(table[1, 2])


class TestCountingStatisticsScaling:
    """The variance has to behave like counting statistics, not merely be present."""

    def test_quadrupling_the_counts_on_both_sides_halves_the_relative_uncertainty(self):
        """Four times the counts, half the relative error bar — the sqrt(N) law, end to end.

        For Poisson pixels ``Var(Sbar) = Sbar / n``, so the relative variance of the ratio is
        ``(1/Sbar + 1/Obar) / n`` and scaling both sides by 4 divides it by 4 exactly. Run without a
        proton charge, so the 0.5% systematic floor is absent and the scaling is exact rather than
        approximate; with proton charge the floor would dominate at high counts, which is a different
        invariant.
        """
        n_bins, ny, nx = 2, 6, 6
        base_sample, base_ob = 400.0, 800.0
        one_x = normalize_roi_spectrum(
            _stack(np.full((n_bins, ny, nx), base_sample)),
            _stack(np.full((n_bins, ny, nx), base_ob)),
            _ROI,
        )
        four_x = normalize_roi_spectrum(
            _stack(np.full((n_bins, ny, nx), 4.0 * base_sample)),
            _stack(np.full((n_bins, ny, nx), 4.0 * base_ob)),
            _ROI,
        )

        expected_rel_var = (1.0 / base_sample + 1.0 / base_ob) / _ROI_PIXELS
        rel_one_x = np.sqrt(one_x.variances) / one_x.values
        rel_four_x = np.sqrt(four_x.variances) / four_x.values

        # the transmission itself is unchanged by the scaling
        np.testing.assert_allclose(one_x.values, base_sample / base_ob, rtol=1e-14)
        np.testing.assert_allclose(four_x.values, base_sample / base_ob, rtol=1e-14)
        np.testing.assert_allclose(rel_one_x, np.sqrt(expected_rel_var), rtol=1e-13)
        np.testing.assert_allclose(rel_one_x, np.sqrt(0.00031250), rtol=1e-9)
        np.testing.assert_allclose(rel_four_x, rel_one_x / 2.0, rtol=1e-12)
