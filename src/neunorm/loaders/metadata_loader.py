"""Metadata loaders for NeuNorm.

Reads acquisition metadata (proton charge, duration, detector, TOF binning) from
VENUS NeXus files, plus optional shutter-count and spectra-TOF sidecar text files,
and locates the NeXus file of a VENUS image from the run number in its name.
"""

import glob
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import scipp as sc
from loguru import logger


def load_metadata(  # noqa: C901
    file_path: Union[str, Path],
    read_shutter_counts: bool = False,
    read_spectra_tof: bool = False,
    *,
    image_dir: Optional[Union[str, Path]] = None,
) -> dict[str, sc.Variable]:
    """Load metadata from NeXus file.

    Parameters
    ----------
    file_path : str or Path
        Path to NeXus HDF5 file containing metadata
    read_shutter_counts : bool
        Whether to read shutter counts from the image directory (default: False)
    read_spectra_tof : bool
        Whether to read spectra TOF from the image directory (default: False)
    image_dir : str or Path, optional
        Directory the image stack was actually loaded from. When given, the ``*_Spectra.txt`` and
        ``*_ShutterCount.txt`` sidecar files are read from here instead of the raw-acquisition
        directory recorded in the NeXus DAS log ``BL10:Exp:IM:ImageFilePath``. This is required for
        VENUS TPX1, whose images come from the auto-reduction tree (different frame count than raw);
        reading the sidecars from the raw tree mismatches the image stack (GitHub #187). When given,
        the returned ``image_file_path`` reflects this directory rather than the DAS-log value, so the
        recorded provenance matches where the data was actually read.
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

        # The DAS log records the RAW acquisition directory, which is not necessarily where the
        # caller's images came from. When the caller supplies the actual image directory, read the
        # sidecar files (spectra TOF, shutter counts) from there so they match the image stack (#187),
        # and record that directory as image_file_path so the stored provenance points at the real
        # data source instead of the known-mismatched raw DAS-log path.
        if image_dir is not None:
            # Resolve to an absolute path (like the DAS-log branch above) so a relative image_dir
            # does not depend on the current working directory. Note the two readers diverge on a
            # directory that does not exist: load_spectra_tof raises FileNotFoundError, while
            # load_shutter_counts only warns and returns an empty array.
            image_path = Path(image_dir).resolve()
            metadata["image_file_path"] = sc.scalar(str(image_path))

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
            if "BL10:Det:T1:TSStart_RBV" in f["entry"]["DASlogs"]:
                metadata["tof_start"] = sc.scalar(
                    float(f["entry"]["DASlogs"]["BL10:Det:T1:TSStart_RBV"]["value"][0]), unit="us"
                )

            if "BL10:Det:T1:TSBinSize_RBV" in f["entry"]["DASlogs"]:
                metadata["tof_bin_size"] = sc.scalar(
                    float(f["entry"]["DASlogs"]["BL10:Det:T1:TSBinSize_RBV"]["value"][0]), unit="us"
                )

            if "BL10:Det:T1:TSSize_RBV" in f["entry"]["DASlogs"]:
                metadata["tof_num_bins"] = sc.scalar(int(f["entry"]["DASlogs"]["BL10:Det:T1:TSSize_RBV"]["value"][0]))

    logger.debug(f"Loaded metadata: {metadata}")

    return metadata


#: The TIFF tag the VENUS DAQ writes on each iKon-XL frame: the proton charge delivered during the
#: exposure, in coulombs. The CCD pipeline reads it as a dimensionless coordinate of that name; the
#: QHY411 frames do not carry it, which is what :func:`find_nexus_file` is for.
PROTON_CHARGE_TAG = "IntegratedPCharge"


def run_number_in_name(name: Union[str, Path]) -> Optional[int]:
    """The run number in a VENUS image file name: the digits of its ``_Run_<n>`` token.

    The token ends at the next underscore, at the extension or at the end of the name, so
    ``20260928_Run_31567_sample_0.tiff`` gives 31567 and a name without the token gives ``None``.
    A ``_Run_`` not followed by digits is skipped and the search goes on.
    """
    stem = Path(name).name
    marker = "_Run_"
    start = 0
    while (pos := stem.find(marker, start)) != -1:
        rest = stem[pos + len(marker) :]
        end = 0
        while end < len(rest) and rest[end].isdigit():
            end += 1
        if end and (end == len(rest) or rest[end] in "_."):
            return int(rest[:end])
        start = pos + len(marker)
    return None


def find_nexus_file(image_path: Union[str, Path], *, nexus_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """The NeXus file of the run an image belongs to, or ``None`` when it cannot be located.

    The run number is the ``_Run_<n>`` token of the image name and the file is ``*_<n>.nxs.h5`` in
    ``nexus_dir``. Without ``nexus_dir`` the ``nexus`` folder of the experiment is used: the
    ``IPTS-<m>`` directory the image sits under, so an image anywhere below
    ``/SNS/VENUS/IPTS-38722/`` is matched with ``/SNS/VENUS/IPTS-38722/nexus/VENUS_<n>.nxs.h5``.

    Parameters
    ----------
    image_path : str or Path
        An image file written by the VENUS DAQ (``<date>_Run_<n>_<title>_<i>.tiff``).
    nexus_dir : str or Path, optional
        The folder holding the NeXus files, when the images do not live in an ``IPTS-<m>`` tree
        (copies, a local checkout).

    Returns
    -------
    Path or None
        The NeXus file, or ``None`` when the name has no run number, no ``IPTS-<m>`` directory is
        above the image and ``nexus_dir`` was not given, or no ``*_<n>.nxs.h5`` exists there.
    """
    image_path = Path(image_path)
    run = run_number_in_name(image_path.name)
    if run is None:
        return None
    if nexus_dir is None:
        ipts = next((p for p in image_path.resolve().parents if p.name.startswith("IPTS-")), None)
        if ipts is None:
            return None
        nexus_dir = ipts / "nexus"
    nexus_dir = Path(nexus_dir)
    if not nexus_dir.is_dir():
        return None
    matches = sorted(nexus_dir.glob(f"*_{run}.nxs.h5"))
    return matches[0] if matches else None


def load_proton_charge_coulombs(nexus_path: Union[str, Path]) -> float:
    """The proton charge of a run from its NeXus file, in coulombs.

    ``entry/proton_charge`` is stored in picocoulombs; the value is converted to coulombs, the unit
    of the :data:`PROTON_CHARGE_TAG` TIFF tag, so the two sources are interchangeable. Note that the
    NeXus value covers the whole run while the tag covers the exposure alone: on a 300 s iKon-XL
    run the two differ by about 0.4 %.
    """
    return float(load_metadata(nexus_path)["proton_charge"].to(unit="C").value)


def load_shutter_counts(image_path: Union[str, Path]) -> sc.Variable:
    """Load shutter counts from a text file.

    Parameters
    ----------
    image_path : str or Path
        Path to the directory containing the image files, where we expect to find a shutter count file
        named ``*_ShutterCount.txt``

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
        named ``*_Spectra.txt``

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
