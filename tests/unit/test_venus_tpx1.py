import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from PIL import Image
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.pipelines.venus_tpx1 import _tof_bin_edges_from_left_edges, run_venus_tpx1_pipeline


def _write_spectra(path, tof_values):
    """Write a whitespace-delimited ``*_Spectra.txt`` sidecar (one TOF value per row)."""
    with open(path, "w") as f:
        for v in tof_values:
            f.write(f"{v} 0\n")


def _write_nexus(path, proton_charge, das_image_path, include_time_offset=True):
    """Write a minimal VENUS NeXus file. ``das_image_path`` is the RAW dir recorded in the DAS log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset("proton_charge", data=[proton_charge])
        entry.create_dataset("duration", data=[60.0])
        daslogs = entry.create_group("DASlogs")
        daslogs.create_group("BL10:Exp:IM:ImageFilePath").create_dataset("value", data=[[das_image_path]])
        if include_time_offset:
            daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        daslogs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[b"MCP TPX1"]])


def test_tof_bin_edges_from_left_edges_non_uniform():
    """The appended closing edge extrapolates the LAST bin's width, verified on non-uniform edges.

    Left edges [0.1, 0.2, 0.4, 0.7] have widths [0.1, 0.2, 0.3], so the closing edge is
    0.7 + 0.3 = 1.0 — which distinguishes the ``last + (last - prev)`` rule from a first-width
    or average-width formula (uniform fixtures could not).
    """
    left = sc.array(dims=["N_image"], values=[0.1, 0.2, 0.4, 0.7], unit="s")
    edges = _tof_bin_edges_from_left_edges(left)
    assert edges.dims == ("tof",)
    assert edges.unit == sc.Unit("s")
    np.testing.assert_allclose(edges.values, [0.1, 0.2, 0.4, 0.7, 1.0])


class TestVenusTPX1Pipeline:
    """Tests for the VENUS TPX1 pipeline."""

    @classmethod
    def setup_class(cls):
        """Create tiff files (with co-located spectra) for testing once for all tests in this class.

        Mirrors real VENUS TPX1 topology: the images live in the auto-reduction tree with their
        ``*_Spectra.txt`` sidecar CO-LOCATED next to them, while the NeXus DAS log records a
        DIFFERENT raw-acquisition directory (a decoy here) whose spectra has a different frame
        count. The pipeline must read the co-located sidecar, not the DAS-log one (GitHub #187).
        """
        cls.sample_tiff_paths = []
        cls.ob_tiff_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # Images + co-located spectra live in the auto-reduction tree (where the pipeline reads).
        sample_image_dir = tmp_dir / "autoreduce" / "sample"
        ob_image_dir = tmp_dir / "autoreduce" / "ob"
        sample_image_dir.mkdir(parents=True, exist_ok=True)
        ob_image_dir.mkdir(parents=True, exist_ok=True)

        # create 5 sample tiffs with values 81-85
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            filename = sample_image_dir / f"sample_{i:03}.tiff"
            Image.fromarray(data).save(filename)
            cls.sample_tiff_paths.append(filename)

        # create 5 OB tiffs with values 99, 100, 101, 102, 103
        for i in range(5):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            filename = ob_image_dir / f"ob_{i:03}.tiff"
            Image.fromarray(data).save(filename)
            cls.ob_tiff_paths.append(filename)

        # Co-located spectra: 5 rows = one LEFT bin edge per image (real autoreduce topology). The
        # pipeline appends the closing edge -> N+1 bin edges [0.1..0.6], matching the
        # wavelength/energy/rebin assertions below.
        _write_spectra(sample_image_dir / "sample_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])
        _write_spectra(ob_image_dir / "ob_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])

        # DECOY raw dir recorded in the DAS log, with a DIFFERENT frame count (8). If the fix
        # regresses to DAS-log resolution, assigning a length-8 coord onto the length-5 image stack
        # raises scipp DimensionError and every pipeline test fails — the #187 regression guard.
        sample_raw_dir = tmp_dir / "raw" / "sample"
        ob_raw_dir = tmp_dir / "raw" / "ob"
        sample_raw_dir.mkdir(parents=True, exist_ok=True)
        ob_raw_dir.mkdir(parents=True, exist_ok=True)
        _write_spectra(sample_raw_dir / "raw_Spectra.txt", [round(0.05 * (i + 1), 2) for i in range(8)])
        _write_spectra(ob_raw_dir / "raw_Spectra.txt", [round(0.05 * (i + 1), 2) for i in range(8)])

        # NeXus files; DAS log points at the raw decoy (relative to the NeXus grandparent = tmp_dir).
        cls.sample_nexus_path = tmp_dir / "nexus" / "test_sample_metadata.nxs.h5"
        _write_nexus(cls.sample_nexus_path, 12345, b"raw/sample")
        cls.ob_nexus_path = tmp_dir / "nexus" / "test_ob_metadata.nxs.h5"
        _write_nexus(cls.ob_nexus_path, 12345 * 2, b"raw/ob")

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_venus_tpx1_pipeline_hdf5(self):
        """
        Test the VENUS TPX1 pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = S / AVERAGE(OB) * (proton_charge_ob / proton_charge_sample)
                # but will be two times because of the difference proton charge between sample and OB
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / (99 + i) * 2)
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.245, rtol=0.1)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
                assert "tof" in hf
                np.testing.assert_allclose(hf["tof"], np.arange(0.1, 0.7, 0.1))
                assert "wavelength" in hf
                np.testing.assert_allclose(hf["wavelength"], np.array([16.6, 32.4, 48.3, 64.1, 79.9, 95.7]), atol=0.1)
                assert "energy" in hf
                np.testing.assert_allclose(
                    hf["energy"], np.array([0.296, 0.078, 0.035, 0.020, 0.013, 0.008]), atol=0.001
                )
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_tiff_paths" in hf
                assert "metadata/ob_tiff_paths" in hf
                assert "metadata/sample_hdf5_paths" in hf
                assert "metadata/ob_hdf5_paths" in hf
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test
                assert "proton_charge" in hf
                np.testing.assert_equal(hf["proton_charge"][()], 12345)
                assert "detector" in hf
                np.testing.assert_equal(hf["detector"].asstr()[()], "MCP TPX1")

    def test_venus_tpx1_pipeline_tiff(self):
        """
        Test the VENUS TPX1 pipeline end-to-end with TIFF output and ROI.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

            # Load back the file to verify contents
            dg = load_scitiff(output_path)

        assert isinstance(dg, sc.DataGroup)
        assert "image" in dg
        assert "daq" in dg
        assert "extra" in dg

        # Check image data and metadata
        image = dg["image"]
        assert image.dtype == sc.DType.float32
        assert image.dims == ("t", "y", "x")
        assert image.values.shape == (5, 20, 20)
        # check that transmission values are correct based on the formula
        # T = (S) / AVERAGE(OB) * (proton_charge_ob / proton_charge_sample)
        for i in range(5):
            np.testing.assert_allclose(image.values[i], (81 + i) / (99 + i) * 2)
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.06, rtol=0.1)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "SNS"
        assert daq.instrument == "VENUS"
        assert daq.detector_type == "MCP TPX1"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert json.loads(extra["sample_tiff_paths"]) == [[str(p) for p in self.sample_tiff_paths]]
        assert json.loads(extra["ob_tiff_paths"]) == [[str(p) for p in self.ob_tiff_paths]]
        assert json.loads(extra["sample_hdf5_paths"]) == [str(self.sample_nexus_path)]
        assert json.loads(extra["ob_hdf5_paths"]) == [str(self.ob_nexus_path)]

        assert "processing_timestamp" in extra
        np.testing.assert_equal(json.loads(extra["roi_applied"]), (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_venus_tpx1_pipeline_spatial_rebin(self):
        """
        Test the rebin_by_spatial function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_spatial=8,
            )

            assert output_path.exists()

        assert transmission.shape == (5, 4, 4)  # original was (5, 32, 32) so 8x8 rebin should give (5, 4, 4)

        # values should be the same but variances should be reduced because of the rebinning
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / (99 + i) * 2)

        np.testing.assert_allclose(transmission.variances, 0.001, rtol=0.1)

    def test_venus_tpx1_pipeline_tof_rebin(self):
        """
        Test the rebin_by_tof function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=2,
            )

            assert output_path.exists()

        assert transmission.shape == (
            3,
            32,
            32,
        )  # original was (5, 32, 32) so rebin by factor of 2 should give (3, 32, 32)

        np.testing.assert_allclose(transmission.values[0], (81 + 82) / (99 + 100) * 2)
        np.testing.assert_allclose(transmission.values[1], (83 + 84) / (101 + 102) * 2)
        np.testing.assert_allclose(transmission.values[2], (85) / 103 * 2)

        np.testing.assert_allclose(transmission.variances[0], 0.03, rtol=0.1)
        np.testing.assert_allclose(transmission.variances[1], 0.03, rtol=0.1)
        np.testing.assert_allclose(transmission.variances[2], 0.06, rtol=0.1)

        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.3, 0.5, 0.6])
        np.testing.assert_allclose(transmission.coords["wavelength"].values, [16.6, 48.3, 79.9, 95.7], atol=0.1)
        np.testing.assert_allclose(transmission.coords["energy"].values, [0.296, 0.035, 0.013, 0.008], atol=0.001)

    def test_venus_tpx1_pipeline_tof_rebin_auto(self):
        """
        Test the rebin_by_tof function with analyze statistics to get recommended rebinning factor.
        Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=True,
            )

            assert output_path.exists()

        assert transmission.shape == (
            5,
            32,
            32,
        )  # should be unchanged because the recommended rebinning factor based on the test TOF data is 1 (no rebinning)

        # values and variances should be the same
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / (99 + i) * 2)

        np.testing.assert_allclose(transmission.variances, 0.06, rtol=0.1)

        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_venus_tpx1_pipeline_air_region_correction(self):
        """
        Test the air region correction function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                air_roi=(0, 0, 10, 10),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 32, 32)
        # Since all the data are the same for a single tof the air correction should just normalize 1.
        np.testing.assert_allclose(transmission.values, 1)
        np.testing.assert_allclose(transmission.variances, 0.0227, rtol=0.1)

    def test_venus_tpx1_pipeline_air_roi_provenance_recorded(self):
        """A TPX pipeline records air_roi in HDF5 output-file provenance (#180 F7)."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                air_roi=(0, 0, 10, 10),
            )
            with h5py.File(output_path, "r") as hf:
                assert "metadata/air_roi" in hf
                ds = hf["metadata/air_roi"]
                assert ds.attrs.get("encoding") != "json"  # native int array, not the JSON backstop
                np.testing.assert_array_equal(ds[()], [0, 0, 10, 10])

    def test_venus_tpx1_pipeline_mask_air_roi_over_bin_edge_tof(self):
        """A non-rectangular MaskROI air_roi flows through the TPX1 pipeline on top of #187's N+1
        bin-edge tof coord (the #187 x MaskROI seam).

        Regression guard for the exact interaction introduced when this feature was rebased onto the
        #187 fix: a MaskROI air region reduces over (y, x) only, so the bin-edge tof coord (length
        N+1 on a length-N tof dim) must survive air correction untouched, the air-region mean must be
        driven to 1.0 per tof bin, the propagated variance must stay finite, and the mask must be
        recorded in provenance as its JSON summary (not a native-int rectangle).
        """
        from neunorm.data_models.roi import MaskROI

        # L-shaped (non-rectangular) selection over the (y, x) frame -> exercises the mask reduction
        # path (as_region_list), not the rectangle as_roi_bounds path.
        sel = np.zeros((32, 32), dtype=bool)
        sel[0:10, 0:5] = True
        sel[0:5, 0:10] = True
        mask = MaskROI(selection=sel)

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                air_roi=mask,
            )
            assert output_path.exists()

            # #187: tof is a proper bin-edge axis (N+1 edges) and survives MaskROI air correction.
            assert transmission.shape == (5, 32, 32)
            assert transmission.coords["tof"].sizes["tof"] == transmission.sizes["tof"] + 1
            np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

            # Spatially-uniform input -> air correction drives the frame to 1.0 per tof bin; the
            # mask-aware pooled mean keeps the propagated variance finite (no 0*inf = NaN).
            np.testing.assert_allclose(transmission.values, 1.0, rtol=1e-5)
            assert np.all(np.isfinite(transmission.variances))
            assert np.all(transmission.variances >= 0.0)

            # MaskROI air_roi provenance is the JSON mask summary, not a native-int rect array.
            with h5py.File(output_path, "r") as hf:
                assert "metadata/air_roi" in hf
                assert json.loads(hf["metadata/air_roi"].asstr()[()]) == {"mask": mask.provenance_summary()}

    def test_venus_tpx1_pipeline_invalid_paths(self):
        """Check error when the length of tiff and hdf5 paths do not match."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(
                ValueError,
                match=r"Number of sample HDF5 paths \(0\) does not match number of sample TIFF path groups \(1\).",
            ):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[],  # empty list should trigger error
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

            with pytest.raises(
                ValueError, match=r"Number of OB HDF5 paths \(0\) does not match number of OB TIFF path groups \(1\)."
            ):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[],  # empty list should trigger error
                    output_path=output_path,
                )

    def test_venus_tpx1_pipeline_empty_tiff_group(self):
        """An empty inner TIFF path group is rejected with a descriptive error, not a bare IndexError."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(ValueError, match=r"sample TIFF path group must contain at least one"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[[]],  # empty run group
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

            with pytest.raises(ValueError, match=r"OB TIFF path group must contain at least one"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[[]],  # empty run group
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

    def test_venus_tpx1_pipeline_invalid_rebin_by_tof(self):
        """Check error for invalid rebin_by_tof values."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(ValueError, match=r"bool, an int factor, or a list"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                    rebin_by_tof="invalid",  # invalid value should trigger error
                )

    def test_venus_tpx1_pipeline_rebin_by_tof_list_mean(self):
        """A bin-list rebin_by_tof mean-reduces the 5-frame TOF stack and carries spectra_tof out."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=[[0, 2], [2, 5]],  # 5 frames -> 2 mean bins
            )
            assert transmission.shape == (2, 32, 32)
            assert transmission.coords["tof"].sizes["tof"] == 3  # bin-edge axis for 2 bins
            assert "spectra_tof" in transmission.coords
            # mean-reduced: bin0 = mean(81,82)/mean(99,100)*2, bin1 = mean(83,84,85)/mean(101,102,103)*2
            np.testing.assert_allclose(transmission.values[0], 81.5 / 99.5 * 2, rtol=1e-5)
            np.testing.assert_allclose(transmission.values[1], 84.0 / 102.0 * 2, rtol=1e-5)
            with h5py.File(output_path, "r") as hf:
                assert "spectra_tof" in hf  # per-bin mean-time exported

    def test_venus_tpx1_pipeline_rebin_by_tof_list_gap(self):
        """A gap in the bin list flags the dropped frame as missing data through to the output."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=[[0, 2], [3, 5]],  # frame 2 dropped -> gap bin
            )
            assert transmission.shape == (3, 32, 32)  # real, gap, real
            assert "dropped_frames" in transmission.masks
            np.testing.assert_array_equal(transmission.masks["dropped_frames"].values, [False, True, False])
            # the missing-data flag + per-bin mean-time must persist to the HDF5 output
            with h5py.File(output_path, "r") as hf:
                assert "masks/dropped_frames" in hf
                np.testing.assert_array_equal(hf["masks/dropped_frames"][()], [False, True, False])
                assert "spectra_tof" in hf

    def test_venus_tpx1_pipeline_rebin_by_tof_list_gap_tiff(self):
        """A gapped bin-list rebin must also export to TIFF: the 1-D dropped_frames mask has to
        broadcast to the full (t, y, x) stack for scitiff (regression for the mask-combining path)."""
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)
            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=[[0, 2], [3, 5]],  # gap -> 1-D dropped_frames mask
            )
            assert output_path.exists()  # scitiff write succeeded with the broadcast mask

    def test_venus_tpx1_pipeline_empty_rebin_list_raises(self):
        """An explicit but empty bin list is invalid input, not a silent no-op rebin."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            with pytest.raises(ValueError, match="at least one"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                    rebin_by_tof=[],  # empty list -> invalid, must raise (not skipped)
                )

    def test_venus_tpx1_pipeline_rebin_by_tof_list_gap_with_air_roi(self):
        """A gapped bin-list rebin combined with air_roi must not crash: the NaN gap bin is excluded
        from the air-correction strict-finiteness guard (regression for the gap + air_roi P0)."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=[[0, 2], [3, 5]],  # frame 2 dropped -> gap bin (NaN)
                air_roi=(0, 0, 10, 10),
            )
            assert transmission.shape == (3, 32, 32)
            np.testing.assert_array_equal(transmission.masks["dropped_frames"].values, [False, True, False])
            # real bins air-corrected to ~1.0 (spatially uniform); the gap bin stays NaN
            np.testing.assert_allclose(transmission.values[0], 1.0, rtol=1e-5)
            np.testing.assert_allclose(transmission.values[2], 1.0, rtol=1e-5)
            assert np.isnan(transmission.values[1]).all()

    def test_venus_tpx1_pipeline_invalid_output_format(self):
        """Check error for unsupported output file format."""
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(ValueError, match=r"Unsupported output file format: .bmp"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

    def test_venus_tpx1_pipeline_missing_detector_time_offset(self):
        """Check warning when detector_time_offset is missing."""

        path = Path(self._tmpdir.name) / "nexus" / "test_missing_detector_time_offset.nxs.h5"
        _write_nexus(path, 12345, b"raw/sample", include_time_offset=False)

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=2,
            )

            assert output_path.exists()

        assert "tof" in transmission.coords
        assert "wavelength" not in transmission.coords
        assert "energy" not in transmission.coords

    def test_venus_tpx1_reads_spectra_from_image_dir_not_daslog(self):
        """#187 regression: the TOF axis must come from the spectra co-located with the images,
        not the raw dir recorded in the DAS log.

        The shared fixture's DAS log points at a decoy raw dir whose spectra has 8 rows; the
        co-located sidecar has 5 rows (left edges 0.1..0.5), which the pipeline turns into 6 bin
        edges (0.1..0.6). Reading the decoy would raise scipp DimensionError (a mismatched-length
        coord onto the length-5 stack). A successful run whose tof equals the co-located edges
        proves the fix.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
            )

        # tof from the co-located sidecar (5 left edges -> 6 bin edges), NOT the decoy raw dir (8 rows)
        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_venus_tpx1_real_data_topology_default_path(self):
        """Real autoreduce topology: the co-located spectra has exactly one row per image (N LEFT
        bin edges) — the shape of actual VENUS TPX1 data that #187 crashed on. The pipeline appends
        the closing edge, so the default (non-rebin) pipeline runs and produces an N+1 bin-edge tof
        coord from that sidecar.
        """
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            sample_dir = tmp / "autoreduce" / "sample"
            ob_dir = tmp / "autoreduce" / "ob"
            sample_dir.mkdir(parents=True)
            ob_dir.mkdir(parents=True)

            sample_tiffs, ob_tiffs = [], []
            for i in range(5):
                sp = sample_dir / f"s_{i:03}.tiff"
                Image.fromarray(np.full((8, 8), 81 + i, dtype=np.float32)).save(sp)
                sample_tiffs.append(sp)
                op = ob_dir / f"o_{i:03}.tiff"
                Image.fromarray(np.full((8, 8), 99 + i, dtype=np.float32)).save(op)
                ob_tiffs.append(op)

            # N (=5) co-located spectra rows — one LEFT bin edge per image (real-data shape)
            _write_spectra(sample_dir / "s_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])
            _write_spectra(ob_dir / "o_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])
            # decoy raw dir with a different frame count, recorded in the DAS log
            (tmp / "raw" / "sample").mkdir(parents=True)
            (tmp / "raw" / "ob").mkdir(parents=True)
            _write_spectra(tmp / "raw" / "sample" / "r_Spectra.txt", [round(0.05 * (i + 1), 2) for i in range(7)])
            _write_spectra(tmp / "raw" / "ob" / "r_Spectra.txt", [round(0.05 * (i + 1), 2) for i in range(7)])
            _write_nexus(tmp / "nexus" / "s.nxs.h5", 12345, b"raw/sample")
            _write_nexus(tmp / "nexus" / "o.nxs.h5", 12345 * 2, b"raw/ob")

            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
                output_path = Path(f.name)

                transmission = run_venus_tpx1_pipeline(
                    sample_tiff_paths=[sample_tiffs],
                    ob_tiff_paths=[ob_tiffs],
                    sample_hdf5_paths=[tmp / "nexus" / "s.nxs.h5"],
                    ob_hdf5_paths=[tmp / "nexus" / "o.nxs.h5"],
                    output_path=output_path,
                )

                assert output_path.exists()

        # 5 images = 5 bins; the 5 left edges + appended closing edge => 6 bin edges [0.1..0.6]
        assert transmission.shape == (5, 8, 8)
        assert transmission.coords["tof"].sizes["tof"] == 6
        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / (99 + i) * 2)
