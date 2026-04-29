import glob
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import scipp as sc
from loguru import logger


def load_metadata(  # noqa: C901
    file_path: Union[str, Path], read_shutter_counts: bool = False, read_spectra_tof: bool = False
) -> dict[str, sc.Variable]:
    """Load metadata from NeXus file.

    Parameters
    ----------
    file_path : str or Path
        Path to NeXus HDF5 file containing metadata
    read_shutter_counts : bool
        Whether to read shutter counts from the image directory specified in the metadata (default: False)
    read_spectra_tof : bool
        Whether to read spectra TOF from the image directory specified in the metadata (default: False)
    Returns
    -------
    dict
        Metadata values for proton charge, duration, image file path, and optionally shutter counts.
        All values are returned as scipp Variables.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Metadata source file not found: {file_path}")

    logger.info(f"Loading metadata from {file_path}")

    metadata: dict[str, sc.Variable] = {}

    with h5py.File(file_path, "r") as f:
        if "entry" not in f:
            raise KeyError("Expected 'entry' group not found in HDF5 file")

        if "proton_charge" not in f["entry"]:
            raise KeyError("Expected 'proton_charge' dataset not found in HDF5 file under 'entry'")
        metadata["proton_charge"] = sc.scalar(float(f["entry"]["proton_charge"][0]), unit="pC")

        if "duration" not in f["entry"]:
            raise KeyError("Expected 'duration' dataset not found in HDF5 file under 'entry'")
        metadata["duration"] = sc.scalar(float(f["entry"]["duration"][0]), unit="s")

        image_path = ""
        if "DASlogs" in f["entry"] and "BL10:Exp:IM:ImageFilePath" in f["entry"]["DASlogs"]:
            image_file_path = f["entry"]["DASlogs"]["BL10:Exp:IM:ImageFilePath"]["value"][-1][0].decode("utf-8").strip()
            metadata["image_file_path"] = sc.scalar(image_file_path)
            # The image path is relative to the parent directory of the HDF5 file, so we need to resolve it
            image_path = file_path.parent.parent.joinpath(image_file_path).resolve()

        if read_shutter_counts:
            metadata["shutter_counts"] = load_shutter_counts(image_path)

        if read_spectra_tof:
            metadata["spectra_tof"] = load_spectra_tof(image_path)

        if "DASlogs" in f["entry"]:
            if "BL10:Exp:Det" in f["entry"]["DASlogs"]:
                metadata["detector"] = sc.scalar(
                    f["entry"]["DASlogs"]["BL10:Exp:Det"]["value_strings"][-1][0].decode("utf-8").strip()
                )
            if "BL10:Det:TH:DSPT1:TIDelay" in f["entry"]["DASlogs"]:
                metadata["detector_time_offset"] = sc.scalar(
                    float(f["entry"]["DASlogs"]["BL10:Det:TH:DSPT1:TIDelay"]["average_value"][0]), unit="us"
                )

            # The TOF binning can be determined by these logs, it provides start, bin size, and number of bins.
            if (
                "BL10:Det:T1:TSStart_RBV" in f["entry"]["DASlogs"]
                and "BL10:Det:T1:TSBinSize_RBV" in f["entry"]["DASlogs"]
                and "BL10:Det:T1:TSSize_RBV" in f["entry"]["DASlogs"]
            ):
                metadata["tof_binning"] = {}
                metadata["tof_binning"]["start"] = sc.scalar(
                    float(f["entry"]["DASlogs"]["BL10:Det:T1:TSStart_RBV"]["value"][0]), unit="us"
                )

                metadata["tof_binning"]["bin_size"] = sc.scalar(
                    float(f["entry"]["DASlogs"]["BL10:Det:T1:TSBinSize_RBV"]["value"][0]), unit="us"
                )

                metadata["tof_binning"]["num_bins"] = sc.scalar(
                    int(f["entry"]["DASlogs"]["BL10:Det:T1:TSSize_RBV"]["value"][0])
                )

    logger.debug(f"Loaded metadata: {metadata}")

    return metadata


def load_shutter_counts(image_path: Union[str, Path]) -> sc.Variable:
    """Load shutter counts from a text file.

    Parameters
    ----------
    image_path : str or Path
        Path to the directory containing the image files, where we expect to find a shutter count file
        named *_ShutterCount.txt

    Returns
    -------
    sc.Variable
        Variable containing shutter counts loaded from the file, up until the first count of 0 is encountered.
    """

    image_path = Path(image_path)

    if image_path.is_dir():
        # Look for shutter count files in the image directory. It is expected to end in _ShutterCount.txt
        shutter_files = glob.glob(str(image_path / "*_ShutterCount.txt"))

        if len(shutter_files) == 0:
            logger.warning("Shutter count file not found!")
        else:
            if len(shutter_files) > 1:
                logger.warning(
                    f"Multiple shutter count files found in {image_path}. "
                    f"Expected only one. Found: {shutter_files}. Using the first one."
                )
            # There should only be one shutter count file
            shutter_count_file = shutter_files[0]
            logger.info(f"Loading shutter counts from {shutter_count_file}")

            # stop loading shutter counts if we encounter a count of 0
            list_shutter_counts = []
            with open(shutter_count_file) as txt_fh:
                lines = txt_fh.readlines()
                for _line in lines:
                    _, _value = _line.split()
                    if _value == "0":
                        break
                    list_shutter_counts.append(float(_value))

            return sc.array(dims=["N_image"], values=list_shutter_counts)
    else:
        logger.warning(f"Image directory in metadata not found: {image_path}. Shutter counts will not be loaded.")
    return sc.array(dims=["N_image"], values=np.array([], dtype=float))


def load_spectra_tof(image_path: Union[str, Path]) -> sc.Variable:  # noqa: C901
    """Load TOF values from spectra text file.

    Parameters
    ----------
    image_path : str or Path
        Path to the directory containing the image files, where we expect to find a spectra file
        named *_Spectra.txt

    Returns
    -------
    sc.Variable
        Variable containing TOF values loaded from the file, same number as images in the stack.
    """

    image_path = Path(image_path)

    if image_path.is_dir():
        # Look for spectra files in the image directory. It is expected to end in _Spectra.txt
        spectra_files = glob.glob(str(image_path / "*_Spectra.txt"))
        if len(spectra_files) == 0:
            raise FileNotFoundError(
                f"Spectra TOF file not found in {image_path}. Expected a file ending with '_Spectra.txt'."
            )

        if len(spectra_files) > 1:
            logger.warning(
                f"Multiple spectra files found in {image_path}. "
                f"Expected only one. Found: {spectra_files}. Using the first one."
            )
        # There should only be one spectra file
        spectra_file = spectra_files[0]
        logger.info(f"Loading spectra from {spectra_file}")
        try:
            data = np.loadtxt(spectra_file, skiprows=1, delimiter=",")
        except ValueError:
            data = np.loadtxt(spectra_file)
        return sc.array(
            dims=["N_image"], values=data[:, 0], unit="s"
        )  # return just the TOF values, which should be in the first column

    raise FileNotFoundError(
        f"Image directory in metadata not found: {image_path}. Spectra TOF values will not be loaded."
    )
