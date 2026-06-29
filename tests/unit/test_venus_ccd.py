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
from neunorm.loaders.stack_loader import load_stack
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.run_combiner import combine_runs


class TestVenusCCDPipeline:
    """Tests for the VENUS CCD pipeline."""

    @classmethod
    def setup_class(cls):
        """Create tiff files for testing once for all tests in this class."""
        cls.sample_paths = []
        cls.sample_paths_bad_pixels = []
        cls.sample_paths_air = []
        cls.ob_paths = []
        cls.dark_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # create 5 sample tiffs with values 81-85 and metadata.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "IntegratedPCharge:0.1"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ManufacturerStr:ANDOR"
            filename = tmp_dir / f"sample_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.sample_paths.append(filename)

        # create 5 sample tiffs with values 81-85 and metadata. These have a dead pixel and a gamma spike.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            # add dead pixel at (22, 8)
            data[22, 8] = 0
            # add pixel for gamma spike at (7, 19)
            data[7, 19] = 1000
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "IntegratedPCharge:0.1"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ManufacturerStr:ANDOR"
            filename = tmp_dir / f"sample_bad_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.sample_paths_bad_pixels.append(filename)

        # create 3 OB tiffs with values 99, 100, 101 and metadata
        for i in range(3):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "IntegratedPCharge:0.2"
            exif[65025] = "ManufacturerStr:ANDOR"
            filename = tmp_dir / f"ob_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.ob_paths.append(filename)

        # create 2 dark tiffs with values 4 and 6 and metadata
        for i in range(2):
            data = np.full((32, 32), 4 + 2 * i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "IntegratedPCharge:0.1"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ManufacturerStr:ANDOR"
            filename = tmp_dir / f"dark_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.dark_paths.append(filename)

        # create a sample for testing air region correction
        data = np.full((32, 32), 80, dtype=np.float32)
        # set higher values in region
        data[6:11, 6:11] = 100.0
        img = Image.fromarray(data)
        exif = img.getexif()
        exif[65027] = "IntegratedPCharge:0.1"
        filename = tmp_dir / "sample_air.tiff"
        img.save(filename, exif=exif)
        cls.sample_paths_air.append(filename)

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_venus_ccd_pipeline_hdf5(self):
        """
        Test the VENUS CCD pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=output_path,
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
                # but will be two times because of the difference proton charge between sample and OB
                for i in range(5):
                    # rtol consistent with float32 compute precision (issue #147)
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i - 5) / (100 - 5) * 2, rtol=1e-5)
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.217, rtol=0.1)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                # Check metadata (nested per-run paths stored as round-trippable JSON, issue #140)
                assert "metadata/sample_paths" in hf
                assert json.loads(hf["metadata/sample_paths"].asstr()[()]) == [[str(p) for p in self.sample_paths]]
                assert "metadata/ob_paths" in hf
                assert json.loads(hf["metadata/ob_paths"].asstr()[()]) == [[str(p) for p in self.ob_paths]]
                assert "metadata/dark_paths" in hf
                assert json.loads(hf["metadata/dark_paths"].asstr()[()]) == [[str(p) for p in self.dark_paths]]
                assert "metadata/dark_correction_applied" in hf
                np.testing.assert_equal(hf["metadata/dark_correction_applied"][()], True)
                assert "metadata/gamma_filter_applied" in hf
                np.testing.assert_equal(hf["metadata/gamma_filter_applied"][()], True)
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test
                assert "IntegratedPCharge" in hf
                np.testing.assert_equal(hf["IntegratedPCharge"][:], [0.1] * 5)
                assert "ManufacturerStr" in hf
                np.testing.assert_equal(hf["ManufacturerStr"].asstr()[()], "ANDOR")

    def test_venus_ccd_pipeline_roi_and_tiff(self):
        """
        Test the VENUS CCD pipeline with ROI applied and TIFF output, then verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
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
        assert image.dims == ("z", "y", "x")
        assert image.values.shape == (5, 20, 20)
        # check that transmission values are correct based on the formula
        # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
        # but will be two times because of the difference proton charge between sample and OB
        for i in range(5):
            # rtol consistent with float32 compute precision (issue #147)
            np.testing.assert_allclose(image.values[i], (81 + i - 5) / (100 - 5) * 2, rtol=1e-5)
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.047, rtol=0.1)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "SNS"
        assert daq.instrument == "VENUS"
        assert daq.detector_type == "ANDOR"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert len(extra["sample_paths"].values) == 5
        assert len(extra["ob_paths"].values) == 3
        assert len(extra["dark_paths"].values) == 2
        assert extra["gamma_filter_applied"]

        assert "processing_timestamp" in extra
        np.testing.assert_equal(extra["roi_applied"].value, (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_venus_ccd_pipeline_dark_gamma_spike_pixels(self):
        """Test that the dark and gamma spike pixels are correctly handled.

        For this test just check the returned DataArray instead of looking at the output file.
        """

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths_bad_pixels],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        for i in range(5):
            expected_value = np.full((20, 20), (81 + i - 5) / (100 - 5) * 2)
            expected_value[22 - 5, 8 - 5] = 0  # dead pixel should be zero after dark subtraction and normalization
            # gamma spike should be replaced with the same values from that image
            np.testing.assert_allclose(transmission.values[i], expected_value, rtol=1e-5)

        # check that the variances are higher for the gamma spike pixel and near zero for the dead pixel
        approximate_variances = np.full((5, 20, 20), 0.047)
        approximate_variances[:, 22 - 5, 8 - 5] = 0  # dead pixel should have almost zero variance
        approximate_variances[:, 7 - 5, 19 - 5] = 0.4557  # gamma spike pixel should have higher variance
        np.testing.assert_allclose(transmission.variances, approximate_variances, atol=0.004)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_dead_pixel_mask[22 - 5, 8 - 5] = True
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)

    def test_venus_ccd_pipeline_combine_runs(self):
        """Test that multiple runs are combined correctly.
        Include the same sample paths 3 times, OB twice and dark twice to check that the combination is working.
        Data should double along with selected metadata.

        Just check the returned DataArray instead of looking at the output file."""

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_ccd_pipeline(
                sample_paths=[
                    self.sample_paths,
                    self.sample_paths,
                    self.sample_paths,
                ],  # include the same sample paths three times to test combination
                ob_paths=[self.ob_paths, self.ob_paths],  # include the same OB paths twice to test combination
                dark_paths=[self.dark_paths, self.dark_paths],  # include the same dark paths twice to test combination
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        # values should be the same as the original sample values since we are combining three runs but also normalizing
        # by the number of runs, so it should not change the final transmission values, just reduce the variance.
        for i in range(5):
            expected_value = np.full((20, 20), ((81 + i) - 5) / (100 - 5) * 2)
            np.testing.assert_allclose(transmission.values[i], expected_value, rtol=1e-5)

        # Variances are reasonable and reduced by combining runs. Pinned to the corrected
        # values (~0.0168–0.0180 across the 5 images) after fixing the shared-dark variance
        # double-count (issue #142); previously ~0.018 with the over-counted Var(dark).
        np.testing.assert_allclose(transmission.variances, 0.0174, atol=0.0006)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)

        # check that the metadata keys that should be summed are unchanged
        assert "IntegratedPCharge" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["IntegratedPCharge"].values, [0.1] * 5
        )  # should be unchanged since we are normalizing by the number of runs
        assert "ManufacturerStr" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["ManufacturerStr"].values, "ANDOR"
        )  # should be unchanged since it's the same for both runs

    def test_venus_ccd_pipeline_air_roi(self):
        """Test that air_roi is handled correctly."""

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_ccd_pipeline(
                sample_paths=[
                    self.sample_paths_air,
                ],
                ob_paths=[self.ob_paths[1:2]],  # include only the OB with value 100
                dark_paths=[self.dark_paths],
                output_path=output_path,
                gamma_filter=False,
                air_roi=(6, 6, 11, 11),
            )

            assert output_path.exists()

        assert transmission.shape == (1, 32, 32)
        # air region should equal 1.0 (rtol consistent with float32 compute precision, issue #147)
        np.testing.assert_allclose(transmission.values[:, 6:11, 6:11], 1, rtol=1e-5)
        # check all values
        expected_value = np.full((1, 32, 32), (80 - 5) / (100 - 5) * 2)
        expected_value[:, 6:11, 6:11] = (100 - 5) / (100 - 5) * 2  # air region which equals 2
        expected_value /= 2  # air region should be normalized to 1.0
        np.testing.assert_allclose(transmission.values, expected_value, rtol=1e-5)

        # check that the variances exist and are reasonable
        expected_variances = np.full((1, 32, 32), 0.0168)
        expected_variances[:, 6:11, 6:11] = 0.0237
        np.testing.assert_allclose(transmission.variances, expected_variances, atol=0.001)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((32, 32), dtype=bool)
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)

        # check that the metadata keys that should be summed are unchanged
        assert "IntegratedPCharge" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["IntegratedPCharge"].values, [0.1]
        )  # should be unchanged since we are normalizing by the number of runs

    def test_venus_ccd_pipeline_no_dark(self):
        """Dark current is optional (issue #146): omitting dark_paths skips dark correction.

        Without dark subtraction the transmission is T = (S / pc_s) / (OB / pc_ob),
        i.e. (81 + i) / 100 * 2 for these fixtures (vs (81 + i - 5) / (100 - 5) * 2 with dark).
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
            )

            assert output_path.exists()
            assert transmission.shape == (5, 32, 32)

            with h5py.File(output_path, "r") as hf:
                assert hf["transmission"].shape == (5, 32, 32)
                # No dark subtraction: T = (S / pc_s) / (OB / pc_ob) = (81 + i) / 100 * 2
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / 100 * 2, rtol=1e-5)
                # Uncertainty is present, positive and finite
                assert "uncertainty" in hf
                assert np.all(np.isfinite(hf["uncertainty"][:]))
                assert np.all(hf["uncertainty"][:] > 0)
                # Provenance: dark correction not applied, dark_paths omitted entirely
                assert "metadata/dark_correction_applied" in hf
                np.testing.assert_equal(hf["metadata/dark_correction_applied"][()], False)
                assert "metadata/dark_paths" not in hf

    def test_venus_ccd_pipeline_empty_dark_paths(self):
        """An empty dark_paths list is treated the same as omitting dark (issue #146)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            transmission = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[],
                output_path=Path(f.name),
            )
        assert transmission.shape == (5, 32, 32)
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / 100 * 2, rtol=1e-5)

    def test_venus_ccd_pipeline_requires_output_path(self):
        """output_path is required even though it carries a default for signature compatibility."""
        with pytest.raises(ValueError, match="output_path is required"):
            run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
            )

    def test_venus_ccd_pipeline_no_dark_uncertainty(self):
        """No-dark UQ equals the dark-free propagation, with no dark-frame variance term (issue #146).

        Two independent checks:
        1. The pipeline's no-dark output (values AND variances) matches a direct
           composition of the same library functions with dark subtraction omitted,
           proving no spurious variance term is added or dropped.
        2. Removing the dark removes its variance contribution, so for these uniform
           fixtures the no-dark uncertainty is smaller than the with-dark uncertainty
           everywhere. (Fixture-specific, not a universal law — dark subtraction also
           shrinks the numerator/denominator; the pinned oracle above is what guards
           the general Var(T) property.)
        """
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            no_dark = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=Path(f.name),
                gamma_filter=False,
            )

        # Independent dark-free propagation using the same composable functions the pipeline uses.
        sample = combine_runs(
            [load_stack(self.sample_paths)],
            metadata_keys_to_sum=("IntegratedPCharge",),
            metadata_check_match=["ManufacturerStr"],
            normalize_by_runs=True,
        )
        ob = combine_runs(
            [load_stack(self.ob_paths)],
            metadata_keys_to_sum=("IntegratedPCharge",),
            metadata_check_match=["ManufacturerStr"],
            normalize_by_runs=True,
        )
        ob = prepare_reference(ob, dim="N_image")
        # Mirror the pipeline's float32 handling (issue #147): cast the proton-charge
        # coords to float32 and the result to float32, so this structural check stays
        # bit-tight against the float32 pipeline output (the independent first-principles
        # guard is the pinned-constant oracle below, at rtol=1e-3).
        expected = normalize_transmission(
            sample=sample,
            ob=ob,
            proton_charge_sample=sample.coords["IntegratedPCharge"].astype("float32"),
            proton_charge_ob=ob.coords["IntegratedPCharge"].astype("float32"),
        ).astype("float32")
        assert no_dark.dtype == sc.DType.float32
        np.testing.assert_allclose(no_dark.values, expected.values)
        np.testing.assert_allclose(no_dark.variances, expected.variances)

        # Independent (non-circular) oracle: pin the no-dark variance to literal
        # values, NOT recomposed from the same helpers. A regression inside the
        # shared variance propagation would shift `expected` in lock-step but not
        # these constants, so this is what actually guards the Var(T) property.
        expected_var_per_image = np.array([0.041279, 0.0419, 0.042523, 0.043149, 0.043778])
        expected_var = np.broadcast_to(expected_var_per_image[:, None, None], no_dark.variances.shape)
        np.testing.assert_allclose(no_dark.variances, expected_var, rtol=1e-3)

        # With dark, the dark-frame variance is folded into both numerator and
        # denominator, raising the uncertainty for these fixtures.
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            with_dark = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=Path(f.name),
                gamma_filter=False,
            )
        # The with-dark, proton-charge path is the one that would silently re-promote
        # to float64 if the pc coord were not cast to float32 (issue #147).
        assert with_dark.dtype == sc.DType.float32
        assert np.all(no_dark.variances < with_dark.variances)

    def test_venus_ccd_pipeline_background_roi(self):
        """background_roi flux proxy replaces proton-charge normalization end-to-end (issue #159)."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            t_pc = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths], ob_paths=[self.ob_paths], output_path=output_path, gamma_filter=False
            )
            t_bg = run_venus_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
                gamma_filter=False,
                background_roi=(0, 0, 8, 8),
            )
        assert str(t_bg.unit) == "dimensionless"
        assert t_bg.shape == (5, 32, 32)
        # uniform images -> background-ROI normalization cancels to T = 1 (proton charge skipped)
        np.testing.assert_allclose(t_bg.values, 1.0, rtol=1e-5)
        # and it differs from the proton-charge normalization
        assert not np.allclose(t_bg.values, t_pc.values)
