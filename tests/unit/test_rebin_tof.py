"""
Unit tests for the HistogramRebinner Module
"""

import numpy as np
import pytest
import scipp as sc


def test_rebin_tof_by_bins():
    """Test the rebin_tof function when rebinning by number of bins"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # Add some unaligned coordinates that also depend on TOF to test that they are rebinned correctly
    data.coords["p_charge"] = sc.array(dims=["tof"], values=np.linspace(2, 13, num=12), unit="C")
    data.coords.set_aligned("p_charge", False)
    data.coords["run_number"] = sc.scalar(42)
    data.coords.set_aligned("run_number", False)

    # rebin by a width of 1, which should return data that is unchanged
    result = rebin_tof(data, width=1)
    assert sc.identical(result, data)  # Should return data that is identical to the original data
    assert sc.identical(result.coords["p_charge"], data.coords["p_charge"])  # Unaligned coord should be unchanged
    assert sc.identical(
        result.coords["run_number"], data.coords["run_number"]
    )  # Unaligned scalar coord should be unchanged

    # rebin by a width of 2, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=2)
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly
    # check that unaligned coord is rebinned correctly by summing every 2 adjacent values
    assert sc.identical(result.coords["p_charge"], sc.array(dims=["tof"], values=np.linspace(5, 25, num=6), unit="C"))
    assert sc.identical(
        result.coords["run_number"], data.coords["run_number"]
    )  # Unaligned scalar coord should be unchanged

    # rebin by a width of 3, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=3)
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly
    # check that unaligned coord is rebinned correctly by summing every 3 adjacent values
    assert sc.identical(result.coords["p_charge"], sc.array(dims=["tof"], values=np.linspace(9, 36, num=4), unit="C"))
    assert sc.identical(
        result.coords["run_number"], data.coords["run_number"]
    )  # Unaligned scalar coord should be unchanged

    # rebin by a width of 5, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=5)
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 30000])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 100, which should combine all TOF bins into one
    result = rebin_tof(data, width=100)
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 30000])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_by_time():
    """Test the rebin_tof function when rebinning by time"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by a width of 2500, which should be unchanged since each original bin is 2500 us wide
    result = rebin_tof(data, width=2500, unit="time")
    assert sc.identical(result, data)  # Should be unchanged

    # rebin by a width of 5000, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=5000, unit="time")
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a width of 7500, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=7500, unit="time")
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a width of 12500, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=12500, unit="time")
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 30000])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 250000, which should combine all TOF bins into one
    result = rebin_tof(data, width=250000, unit="time")
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 30000])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_by_logarithmic_time():
    """Test the rebin_tof function when rebinning by logarithmic time"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((11, 5, 5), 10.0)  # 11 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 2500, 30000, num=12, unit="us"),  # N+1 edges for N bins. Bin widths = 2500
        },
    )
    data.variances = values

    # rebin by a logarithmic width of 2
    result = rebin_tof(data, width=2, unit="time", logarithmic=True)
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins based on logarithmic spacing

    np.testing.assert_allclose(result.coords["tof"].values, [2500, 7500, 22500, 30000])  # New edges

    # Each rebinned bin should have sum of original bins based on logarithmic spacing
    new_value = [20, 60, 30]
    expected_values = np.tile(new_value, (5, 5, 1)).T
    np.testing.assert_allclose(result.values, expected_values)
    np.testing.assert_allclose(result.variances, expected_values)  # Variance should also sum correctly


def test_rebin_tof_by_wavelength():
    """Test the rebin_tof function when rebinning by wavelength"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    tof_bin_width = 2500  # Each original TOF bin is 2500 us wide
    wavelength_bin_width = (
        sc.scalar(tof_bin_width, unit="us") * sc.constants.h / (sc.constants.m_n * sc.scalar(25.0, unit="m"))
    )  # Convert to wavelength bin width using l_source_to_detector = 25 m
    one_wavelength_bin = sc.to_unit(wavelength_bin_width, "Angstrom").value

    # rebin based on original bin width in wavelength, which should be unchanged
    result = rebin_tof(data, width=one_wavelength_bin, unit="wavelength")
    assert sc.identical(result, data)  # Should be unchanged

    # rebin by a width of 2 bins, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 2, unit="wavelength")
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a width of 7500, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 3, unit="wavelength")
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a width of 12500, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=one_wavelength_bin * 5, unit="wavelength")
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 30000])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 250000, which should combine all TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 625, unit="wavelength")
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 30000])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_by_logarithmic_wavelength():
    """Test the rebin_tof function when rebinning by logarithmic wavelength

    By setting offset to 0, the logarithmic rebinning by wavelength should match the logarithmic rebinning by time,
    since wavelength is directly proportional to time for a given source-to-detector distance."""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((11, 5, 5), 10.0)  # 11 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 2500, 30000, num=12, unit="us"),  # N+1 edges for N bins. Bin widths = 2500
        },
    )
    data.variances = values

    # rebin by a logarithmic width of 2
    # This should match the logarithmic rebinning by tof_width=2 when offset equals 0
    result = rebin_tof(data, width=2, unit="wavelength", logarithmic=True, detector_time_offset=0)
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins based on logarithmic spacing

    np.testing.assert_allclose(result.coords["tof"].values, [2500, 7500, 22500, 30000])  # New edges

    # Each rebinned bin should have sum of original bins based on logarithmic spacing
    new_value = [20, 60, 30]
    expected_values = np.tile(new_value, (5, 5, 1)).T
    np.testing.assert_allclose(result.values, expected_values)
    np.testing.assert_allclose(result.variances, expected_values)


def test_rebin_tof_by_manual_edges():
    """Test the rebin_tof function when rebinning by manually specified edges"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by manually specifying edges in bin indices
    new_bin_edges = sc.array(dims=["tof"], values=[1, 2, 5, 12], unit="dimensionless", dtype="int64")
    result = rebin_tof(data, width=new_bin_edges, unit="manual")
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins based on specified edges
    np.testing.assert_allclose(
        result.coords["tof"].values, [2500, 5000, 12500, 30000]
    )  # New edges converted from bin indices
    new_values = [10, 30, 70]
    expected_values = np.tile(new_values, (5, 5, 1)).T
    np.testing.assert_allclose(
        result.data.values, expected_values
    )  # Each rebinned bin should have sum of original bins based on specified edges
    np.testing.assert_allclose(result.variances, expected_values)  # Variance should also sum correctly

    # rebin by manually specifying edges in tof
    new_tof_edges = sc.array(dims=["tof"], values=[10000, 20000, 25000], unit="us")
    result = rebin_tof(data, width=new_tof_edges, unit="manual")
    assert result.shape == (2, 5, 5)  # Should have 2 TOF bins based on specified edges
    np.testing.assert_allclose(result.coords["tof"].values, [10000, 20000, 25000])  # New edges
    new_values = [40, 20]
    expected_values = np.tile(new_values, (5, 5, 1)).T
    np.testing.assert_allclose(
        result.data.values, expected_values
    )  # Each rebinned bin should have sum of original bins based on specified edges
    np.testing.assert_allclose(result.variances, expected_values)  # Variance should also sum correctly

    # rebin by manually specifying edges in wavelength
    new_wavelength_edges = sc.array(dims=["tof"], values=[1, 2, 4], unit="Angstrom")
    result = rebin_tof(data, width=new_wavelength_edges, unit="manual")
    assert result.shape == (2, 5, 5)  # Should have 2 TOF bins based on specified edges
    np.testing.assert_allclose(result.coords["tof"].values, [0, 7500, 20000])  # New edges converted to TOF
    new_values = [30, 50]
    expected_values = np.tile(new_values, (5, 5, 1)).T
    np.testing.assert_allclose(
        result.data.values, expected_values
    )  # Each rebinned bin should have sum of original bins based on specified edges
    np.testing.assert_allclose(result.variances, expected_values)


def test_rebin_tof_invalid():
    """Test that rebin_tof raises an error for invalid rebinning widths"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 1e5, 1e7, num=13, unit="ns"),
        },
    )
    data.variances = values

    with pytest.raises(ValueError):
        rebin_tof(data, width=0)  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=-1)  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=-1, unit="time")  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=-1, unit="wavelength")  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, tof_dim="invalid_dim")  # Invalid TOF dimension

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, unit="invalid_unit")  # Invalid unit for width

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, unit="bins", logarithmic=True)  # Logarithmic binning not supported for 'bins' unit

    with pytest.raises(ValueError):
        rebin_tof(data, width=2.5, unit="bins")  # Non-integer width not allowed for 'bins' unit

    with pytest.raises(ValueError, match="When unit is 'manual', width must be provided as a sc.Variable"):
        rebin_tof(data, width=42, unit="manual")

    with pytest.raises(ValueError, match="Manual TOF edges must have at least two values."):
        rebin_tof(data, width=sc.array(dims=["tof"], values=[0], unit="us"), unit="manual")

    with pytest.raises(
        ValueError,
        match="When width is a dimensionless sc.Variable, it must have an integer dtype representing bin indices.",
    ):
        rebin_tof(data, width=sc.array(dims=["tof"], values=[0.5, 1.5], unit="dimensionless"), unit="manual")

    with pytest.raises(ValueError, match="Bin indices in width are out of bounds for the TOF dimension."):
        rebin_tof(
            data, width=sc.array(dims=["tof"], values=[-1, 1, 2], unit="dimensionless", dtype="int64"), unit="manual"
        )

    with pytest.raises(
        ValueError, match="Width provided as a sc.Variable could not be converted to the unit of the TOF coordinates."
    ):
        rebin_tof(data, width=sc.array(dims=["tof"], values=[1, 2, 3], unit="K"), unit="manual")

    with pytest.raises(ValueError, match="When width is provided as a sc.Variable, unit must be set to 'manual'"):
        rebin_tof(data, width=sc.array(dims=["tof"], values=[1, 2, 3], unit="us"))


def test_rebin_with_snapped_boundaries():
    """Test the rebin_with_snapped_boundaries function"""
    from neunorm.tof.histogram_rebinner import rebin_with_snapped_boundaries

    old_edges = sc.array(dims=["tof"], values=[0, 10, 20, 30, 40], unit="us")

    new_edges = rebin_with_snapped_boundaries(old_edges, sc.array(dims=["tof"], values=[15, 35], unit="us"))
    np.testing.assert_allclose(new_edges.values, [10, 30])

    new_edges = rebin_with_snapped_boundaries(old_edges, sc.array(dims=["tof"], values=[-10, 100], unit="us"))
    np.testing.assert_allclose(new_edges.values, [0, 40])

    with pytest.raises(ValueError, match="Requested TOF binning would require splitting existing bins"):
        rebin_with_snapped_boundaries(old_edges, sc.array(dims=["tof"], values=[0, 1, 2, 3], unit="us"))
