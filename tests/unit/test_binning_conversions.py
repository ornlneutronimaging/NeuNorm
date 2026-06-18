"""
Unit tests for TOF/Energy/Wavelength conversion functions.

Tests physics-based conversions using scipy.constants.
"""

import numpy as np
import scipp as sc
import scipy.constants as scipy_const


def test_binning_module_imports():
    """Test that binning module can be imported"""


def test_tof_to_energy_conversion():
    """Test TOF to energy conversion with known values"""
    from neunorm.tof.binning import tof_to_energy

    # Test case: L = 25 m, t = 1 ms → E ≈ 5.227 eV
    tof = sc.scalar(1e-3, unit="s")  # 1 millisecond
    flight_path = sc.scalar(25.0, unit="m")

    energy = tof_to_energy(tof, flight_path)

    # Verify result
    assert energy.unit == "eV"

    # Calculate expected value using scipy.constants
    velocity = 25.0 / 1e-3  # m/s
    expected_j = 0.5 * scipy_const.m_n * velocity**2
    expected_ev = expected_j / scipy_const.e

    # Should match to machine precision
    assert abs(energy.value - expected_ev) / expected_ev < 1e-10


def test_tof_to_energy_inverse_relationship():
    """Verify higher TOF → lower energy"""
    from neunorm.tof.binning import tof_to_energy

    flight_path = sc.scalar(25.0, unit="m")

    tof_short = sc.scalar(1e-4, unit="s")  # 100 μs
    tof_long = sc.scalar(1e-3, unit="s")  # 1 ms

    energy_short = tof_to_energy(tof_short, flight_path)
    energy_long = tof_to_energy(tof_long, flight_path)

    # Shorter TOF → higher energy
    assert energy_short.value > energy_long.value

    # E ∝ 1/t² relationship
    ratio = energy_short.value / energy_long.value
    expected_ratio = (tof_long.value / tof_short.value) ** 2
    assert abs(ratio - expected_ratio) / expected_ratio < 1e-10


def test_tof_to_wavelength_conversion():
    """Test TOF to wavelength conversion with known values"""
    from neunorm.tof.binning import tof_to_wavelength

    # Test case: L = 25 m, t = 1 ms → λ ≈ 1.58 Å
    tof = sc.scalar(1e-3, unit="s")
    flight_path = sc.scalar(25.0, unit="m")

    wavelength = tof_to_wavelength(tof, flight_path)

    assert wavelength.unit == "angstrom"

    # Calculate expected using scipy.constants
    # λ = h * t / (m_n * L)
    h_over_mn = scipy_const.h / scipy_const.m_n
    expected_m = h_over_mn * 1e-3 / 25.0
    expected_angstrom = expected_m * 1e10

    assert abs(wavelength.value - expected_angstrom) / expected_angstrom < 1e-10


def test_tof_to_wavelength_linear_relationship():
    """Verify TOF ∝ wavelength (linear)"""
    from neunorm.tof.binning import tof_to_wavelength

    flight_path = sc.scalar(25.0, unit="m")

    tof1 = sc.scalar(1e-3, unit="s")
    tof2 = sc.scalar(2e-3, unit="s")  # 2× longer

    wl1 = tof_to_wavelength(tof1, flight_path)
    wl2 = tof_to_wavelength(tof2, flight_path)

    # 2× TOF → 2× wavelength (linear relationship)
    ratio = wl2.value / wl1.value
    assert abs(ratio - 2.0) < 1e-10


def test_wavelength_to_energy_de_broglie():
    """Test wavelength-energy conversion via de Broglie relation"""
    from neunorm.tof.binning import wavelength_to_energy

    # Test case: λ = 1.8 Å (thermal neutron) → E ≈ 0.025 eV
    wavelength = sc.scalar(1.8, unit="angstrom")

    energy = wavelength_to_energy(wavelength)

    assert energy.unit == "eV"

    # E = h² / (2 * m_n * λ²)
    wl_m = 1.8e-10
    expected_j = scipy_const.h**2 / (2 * scipy_const.m_n * wl_m**2)
    expected_ev = expected_j / scipy_const.e

    assert abs(energy.value - expected_ev) / expected_ev < 1e-10


def test_energy_to_wavelength_de_broglie():
    """Test energy-wavelength conversion via de Broglie relation"""
    from neunorm.tof.binning import energy_to_wavelength

    # Test case: E = 0.025 eV (thermal) → λ ≈ 1.8 Å
    energy = sc.scalar(0.025, unit="eV")

    wavelength = energy_to_wavelength(energy)

    assert wavelength.unit == "angstrom"

    # λ = h / sqrt(2 * m_n * E)
    energy_j = 0.025 * scipy_const.e
    expected_m = scipy_const.h / np.sqrt(2 * scipy_const.m_n * energy_j)
    expected_angstrom = expected_m * 1e10

    assert abs(wavelength.value - expected_angstrom) / expected_angstrom < 1e-10


def test_wavelength_energy_roundtrip():
    """Test roundtrip conversion: wavelength → energy → wavelength"""
    from neunorm.tof.binning import energy_to_wavelength, wavelength_to_energy

    original_wl = sc.scalar(1.8, unit="angstrom")

    # Forward: wavelength → energy
    energy = wavelength_to_energy(original_wl)

    # Backward: energy → wavelength
    recovered_wl = energy_to_wavelength(energy)

    # Should recover original value to machine precision
    assert abs(recovered_wl.value - original_wl.value) / original_wl.value < 1e-10


def test_create_tof_bins_energy_mode():
    """Test creating TOF bins from energy specification"""
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    config = BinningConfig(bins=100, bin_space="energy", energy_range=(1.0, 100.0), use_log_bin=True)
    flight_path = sc.scalar(25.0, unit="m")

    tof_bins = create_tof_bins(config, flight_path)

    # Should return scipp Variable with TOF dimension
    assert isinstance(tof_bins, sc.Variable)
    assert "tof" in tof_bins.dims
    assert tof_bins.unit == "ns"

    # Should have bins+1 edges
    assert len(tof_bins) == 101

    # TOF bins should be in ASCENDING order (sorted for histogramming)
    # Energy ascending (1→100 eV) → TOF descending → reversed to ascending
    assert tof_bins.values[0] < tof_bins.values[-1]

    # First bin (low TOF) corresponds to high energy (100 eV)
    # Last bin (high TOF) corresponds to low energy (1 eV)


def test_create_tof_bins_wavelength_mode():
    """Test creating TOF bins from wavelength specification"""
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    config = BinningConfig(bins=100, bin_space="wavelength", wavelength_range=(0.5, 3.0), use_log_bin=False)
    flight_path = sc.scalar(25.0, unit="m")

    tof_bins = create_tof_bins(config, flight_path)

    assert isinstance(tof_bins, sc.Variable)
    assert "tof" in tof_bins.dims
    assert tof_bins.unit == "ns"
    assert len(tof_bins) == 101

    # TOF bins should be in ascending order (low λ = low TOF)
    assert tof_bins.values[0] < tof_bins.values[-1]


def test_create_tof_bins_tof_mode_linear():
    """Test creating TOF bins directly (linear spacing)"""
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    config = BinningConfig(
        bins=1000,
        bin_space="tof",
        tof_range=(1e5, 1e7),  # 100 μs to 10 ms
        use_log_bin=False,
    )

    tof_bins = create_tof_bins(config)

    assert tof_bins.unit == "ns"
    assert len(tof_bins) == 1001

    # Linear spacing: bin widths should be constant
    bin_widths = np.diff(tof_bins.values)
    assert np.allclose(bin_widths, bin_widths[0], rtol=1e-10)


def test_create_tof_bins_tof_mode_logarithmic():
    """Test creating TOF bins directly (logarithmic spacing)"""
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    config = BinningConfig(bins=1000, bin_space="tof", tof_range=(1e5, 1e7), use_log_bin=True)

    tof_bins = create_tof_bins(config)

    assert len(tof_bins) == 1001

    # Logarithmic spacing: ratios should be constant
    bin_ratios = tof_bins.values[1:] / tof_bins.values[:-1]
    assert np.allclose(bin_ratios, bin_ratios[0], rtol=1e-6)


def test_get_energy_histogram_from_tof():
    """Test converting TOF histogram to energy histogram"""
    from neunorm.tof.binning import get_energy_histogram

    # Create mock TOF histogram
    tof_edges = sc.linspace("tof", 1e5, 1e7, num=101, unit="ns")
    data_values = np.random.rand(100, 10, 10)  # (tof, x, y)

    hist_tof = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=data_values, unit="counts"), coords={"tof": tof_edges}
    )

    flight_path = sc.scalar(25.0, unit="m")
    hist_energy = get_energy_histogram(hist_tof, flight_path)

    # Should have energy dimension instead of tof
    assert "energy" in hist_energy.dims
    assert "tof" not in hist_energy.dims
    assert "energy" in hist_energy.coords

    # Energy coordinate should be in eV
    assert hist_energy.coords["energy"].unit == "eV"

    # Data should be reversed (high TOF → low energy)
    # First TOF bin (low TOF, high E) should become last energy bin
    assert hist_energy.values[-1, 0, 0] == hist_tof.values[0, 0, 0]


def test_get_wavelength_histogram_from_tof():
    """Test converting TOF histogram to wavelength histogram"""
    from neunorm.tof.binning import get_wavelength_histogram

    # Create mock TOF histogram
    tof_edges = sc.linspace("tof", 1e5, 1e6, num=51, unit="ns")
    data_values = np.random.rand(50, 10, 10)

    hist_tof = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=data_values, unit="counts"), coords={"tof": tof_edges}
    )

    flight_path = sc.scalar(25.0, unit="m")
    hist_wavelength = get_wavelength_histogram(hist_tof, flight_path)

    # Should have wavelength dimension
    assert "wavelength" in hist_wavelength.dims
    assert "tof" not in hist_wavelength.dims
    assert "wavelength" in hist_wavelength.coords

    # Wavelength coordinate should be in angstrom
    assert hist_wavelength.coords["wavelength"].unit == "angstrom"

    # Data should NOT be reversed (low TOF → low λ, both ascending)
    # First TOF bin should remain first wavelength bin
    assert hist_wavelength.values[0, 0, 0] == hist_tof.values[0, 0, 0]


def test_histogram_with_variance_preserved_in_conversion():
    """Test that variance is preserved during TOF→energy conversion"""
    from neunorm.tof.binning import get_energy_histogram

    # Create histogram with variance (Poisson)
    # NOTE: Scipp requires float data for variances
    tof_edges = sc.linspace("tof", 1e5, 1e7, num=51, unit="ns")
    data_values = np.array([100.0, 200.0, 150.0, 300.0] + [100.0] * 46, dtype=float)
    variance_values = data_values.copy()  # Poisson: var = N

    hist_tof = sc.DataArray(
        data=sc.array(dims=["tof"], values=data_values, unit="counts", dtype="float64"), coords={"tof": tof_edges}
    )
    hist_tof.variances = variance_values

    flight_path = sc.scalar(25.0, unit="m")
    hist_energy = get_energy_histogram(hist_tof, flight_path)

    # Variance should be preserved (and reversed along with data)
    assert hist_energy.variances is not None
    assert hist_energy.variances.shape == hist_energy.values.shape

    # First TOF variance (low TOF, high E) → last energy variance
    assert hist_energy.variances[-1] == hist_tof.variances[0]


def test_conversion_with_different_flight_paths():
    """Test that flight path affects conversions correctly"""
    from neunorm.tof.binning import tof_to_energy

    tof = sc.scalar(1e-3, unit="s")

    # Shorter flight path → same TOF gives lower energy
    l_short = sc.scalar(10.0, unit="m")
    l_long = sc.scalar(50.0, unit="m")

    e_short = tof_to_energy(tof, l_short)
    e_long = tof_to_energy(tof, l_long)

    # E ∝ L² for fixed TOF
    assert e_long.value > e_short.value
    ratio = e_long.value / e_short.value
    expected_ratio = (50.0 / 10.0) ** 2  # 25
    assert abs(ratio - expected_ratio) / expected_ratio < 1e-10


def test_tof_to_energy_handles_arrays():
    """Test conversion works with arrays, not just scalars"""
    from neunorm.tof.binning import tof_to_energy

    tof_array = sc.array(dims=["tof"], values=[1e-4, 5e-4, 1e-3, 5e-3], unit="s")
    flight_path = sc.scalar(25.0, unit="m")

    energy_array = tof_to_energy(tof_array, flight_path)

    assert energy_array.dims == ("tof",)  # Dimension preserved
    assert len(energy_array) == 4
    assert energy_array.unit == "eV"

    # Verify each element
    for i in range(4):
        tof_val = tof_array.values[i]
        velocity = 25.0 / tof_val
        expected_ev = (0.5 * scipy_const.m_n * velocity**2) / scipy_const.e
        assert abs(energy_array.values[i] - expected_ev) / expected_ev < 1e-10


def test_wavelength_energy_de_broglie_constant():
    """Verify de Broglie constant matches scipy calculation"""
    from neunorm.tof.binning import wavelength_to_energy
    from neunorm.utils.constants import DE_BROGLIE_EV_ANGSQ

    # Test with several values
    wavelengths = [0.5, 1.0, 1.8, 3.0]  # Angstrom

    for wl_val in wavelengths:
        wl = sc.scalar(wl_val, unit="angstrom")
        energy = wavelength_to_energy(wl)

        # Should match: E = DE_BROGLIE_EV_ANGSQ / λ²
        expected_ev = DE_BROGLIE_EV_ANGSQ / (wl_val**2)
        assert abs(energy.value - expected_ev) / expected_ev < 1e-6


# --- issue #141: energy/wavelength binning must apply detector_time_offset ---


def test_convert_energy_to_tof_is_inverse_of_tof_to_energy():
    """convert_energy_to_tof is the exact inverse of convert_tof_to_energy, offset included.

    This is what makes the energy bin edges consistent with the coordinate labeling (#141).
    """
    from neunorm.tof.coordinate_converter import (
        convert_energy_to_tof,
        convert_tof_to_energy,
        convert_tof_to_wavelength,
        convert_wavelength_to_tof,
    )

    flight_path = sc.scalar(25.0, unit="m")
    offset = sc.scalar(5.0, unit="us")

    energy = sc.array(dims=["energy"], values=[1.0, 10.0, 100.0], unit="eV")
    energy_rt = convert_tof_to_energy(convert_energy_to_tof(energy, flight_path, offset), flight_path, offset)
    np.testing.assert_allclose(sc.to_unit(energy_rt, "eV").values, [1.0, 10.0, 100.0], rtol=1e-9)

    wl = sc.array(dims=["wavelength"], values=[0.5, 1.8, 5.0], unit="angstrom")
    wl_rt = convert_tof_to_wavelength(convert_wavelength_to_tof(wl, flight_path, offset), flight_path, offset)
    np.testing.assert_allclose(sc.to_unit(wl_rt, "angstrom").values, [0.5, 1.8, 5.0], rtol=1e-9)


def test_create_tof_bins_offset_shifts_edges_into_raw_space():
    """A non-zero detector_time_offset shifts the energy/wavelength TOF edges by exactly -offset.

    Regression for #141: the bin edges must live in raw detector-TOF space so they match the
    raw event TOF histogrammed into them; offset=0 reproduces the old physical-TOF edges.
    """
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    flight_path = sc.scalar(25.0, unit="m")
    offset = sc.scalar(5.0, unit="us")  # 5000 ns

    e_cfg = BinningConfig(bins=100, bin_space="energy", energy_range=(1.0, 100.0), use_log_bin=True)
    e0 = sc.to_unit(create_tof_bins(e_cfg, flight_path), "ns").values
    e_off = sc.to_unit(create_tof_bins(e_cfg, flight_path, offset), "ns").values
    np.testing.assert_allclose(e_off, e0 - 5000.0, rtol=1e-9)

    w_cfg = BinningConfig(bins=100, bin_space="wavelength", wavelength_range=(0.5, 5.0), use_log_bin=False)
    w0 = sc.to_unit(create_tof_bins(w_cfg, flight_path), "ns").values
    w_off = sc.to_unit(create_tof_bins(w_cfg, flight_path, offset), "ns").values
    np.testing.assert_allclose(w_off, w0 - 5000.0, rtol=1e-9)


def test_event_energy_binning_places_events_by_raw_tof_with_offset():
    """Events histogrammed in energy space land in the bin bracketing their RAW TOF (#141).

    With the offset applied, the energy bin edges are in raw detector-TOF space, so a peak at
    a known raw TOF falls in the bin whose raw-TOF edges bracket it — consistent with the
    later coordinate labeling. Without the offset (the old behavior) the edges are shifted, so
    the peak lands in a different bin.
    """
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    flight_path = sc.scalar(25.0, unit="m")
    offset = sc.scalar(20.0, unit="us")  # >> bin width here, so the offset clearly moves the peak bin
    raw_tof_ns = 500_000.0  # within the TOF span of the (1, 100) eV range at L=25 m
    n = 1000
    events = EventData(
        tof=np.full(n, raw_tof_ns, dtype=np.int64),
        x=np.full(n, 5, dtype=np.int32),
        y=np.full(n, 5, dtype=np.int32),
        file_path="synthetic.h5",
        total_events=n,
    )
    cfg = BinningConfig(bins=300, bin_space="energy", energy_range=(1.0, 100.0), use_log_bin=True)

    hist = convert_events_to_histogram(
        events, cfg, flight_path, x_bins=10, y_bins=10, compute_variance=False, detector_time_offset=offset
    )
    counts = hist.sum(("x", "y")).data.values
    peak = int(np.argmax(counts))
    assert counts[peak] == n  # all events fell in a single bin
    edges = sc.to_unit(hist.coords["tof"], "ns").values
    # The populated bin's raw-TOF edges bracket the raw event TOF.
    assert edges[peak] <= raw_tof_ns <= edges[peak + 1]

    # Old behavior (offset=0): edges are shifted, so the same events land in a different bin.
    hist0 = convert_events_to_histogram(events, cfg, flight_path, x_bins=10, y_bins=10, compute_variance=False)
    peak0 = int(np.argmax(hist0.sum(("x", "y")).data.values))
    assert peak0 != peak


def test_get_energy_and_wavelength_histogram_apply_offset():
    """get_energy_histogram / get_wavelength_histogram label TOF with detector_time_offset (#141).

    The public histogram-relabeling helpers must use the same offset-aware converter as the bin
    construction, so a histogram built with an offset is labeled consistently. Without the offset
    the labels differ.
    """
    from neunorm.tof.binning import get_energy_histogram, get_wavelength_histogram
    from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength

    flight_path = sc.scalar(25.0, unit="m")
    offset = sc.scalar(20.0, unit="us")
    tof_edges = sc.linspace("tof", 1.0e5, 1.0e6, num=6, unit="ns")
    hist = sc.DataArray(
        data=sc.ones(dims=["tof", "x", "y"], shape=[5, 2, 2], unit="counts"),
        coords={"tof": tof_edges, "x": sc.arange("x", 2), "y": sc.arange("y", 2)},
    )

    # Energy: edges match the offset-aware converter (in eV), reversed (high TOF = low energy).
    he = get_energy_histogram(hist, flight_path, offset)
    expected_e = sc.to_unit(convert_tof_to_energy(tof_edges, flight_path, offset), "eV").values[::-1]
    np.testing.assert_allclose(he.coords["energy"].values, expected_e, rtol=1e-9)
    # Independent (non-circular) physics anchor for one edge: E = 0.5*m_n*(L/(t+offset))^2.
    # The lowest TOF edge (1e5 ns) maps to the highest energy -> last edge after the reversal.
    t_plus_offset_s = (1.0e5 + 20_000.0) * 1e-9  # tof 1e5 ns + 20 us offset, in seconds
    e_hand_ev = 0.5 * scipy_const.m_n * (25.0 / t_plus_offset_s) ** 2 / scipy_const.e
    np.testing.assert_allclose(he.coords["energy"].values[-1], e_hand_ev, rtol=1e-5)
    he0 = get_energy_histogram(hist, flight_path)  # offset=0 -> different labels
    assert not np.allclose(he.coords["energy"].values, he0.coords["energy"].values)

    # Wavelength: edges match the offset-aware converter (in angstrom), not reversed.
    hw = get_wavelength_histogram(hist, flight_path, offset)
    expected_w = sc.to_unit(convert_tof_to_wavelength(tof_edges, flight_path, offset), "angstrom").values
    np.testing.assert_allclose(hw.coords["wavelength"].values, expected_w, rtol=1e-9)
    # Independent physics anchor: lambda = h*(t+offset)/(m_n*L); lowest TOF -> first (shortest) edge.
    w_hand_angstrom = scipy_const.h * t_plus_offset_s / (scipy_const.m_n * 25.0) * 1e10
    np.testing.assert_allclose(hw.coords["wavelength"].values[0], w_hand_angstrom, rtol=1e-5)
    hw0 = get_wavelength_histogram(hist, flight_path)  # offset=0 -> different labels
    assert not np.allclose(hw.coords["wavelength"].values, hw0.coords["wavelength"].values)
