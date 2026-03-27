import glob
from pathlib import Path
from typing import Union

import h5py
from loguru import logger


def load_metadata(file_path: Union[str, Path], read_shutter_counts: bool = False) -> dict:  # noqa: C901
    """Load metadata from NeXus file.

    Parameters
    ----------
    file_path : str or Path
        Path to NeXus HDF5 file containing metadata
    read_shutter_counts : bool
        Whether to read shutter counts from the image directory specified in the metadata (default: False)
    Returns
    -------
    dict        Metadata values for proton charge, duration, image file path, and optionally shutter counts
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Metadata source file not found: {file_path}")

    logger.info(f"Loading metadata from {file_path}")

    metadata = {}

    if read_shutter_counts:
        metadata["shutter_counts"] = None  # Initialize with None in case we fail to load them later

    with h5py.File(file_path, "r") as f:
        if "entry" not in f:
            raise KeyError("Expected 'entry' group not found in HDF5 file")

        if "proton_charge" not in f["entry"]:
            raise KeyError("Expected 'proton_charge' dataset not found in HDF5 file under 'entry'")
        metadata["proton_charge"] = float(f["entry"]["proton_charge"][0])

        if "duration" not in f["entry"]:
            raise KeyError("Expected 'duration' dataset not found in HDF5 file under 'entry'")
        metadata["duration"] = float(f["entry"]["duration"][0])

        if "DASlogs" not in f["entry"] or "BL10:Exp:IM:ImageFilePath" not in f["entry"]["DASlogs"]:
            logger.warning("Unable to find 'BL10:Exp:IM:ImageFilePath' dataset in HDF5 file under 'entry/DASlogs'")
            metadata["image_file_path"] = None
        else:
            # Always read the last value from /entry/DASlogs/BL10:Exp:IM:ImageFilePath/value
            metadata["image_file_path"] = (
                f["entry"]["DASlogs"]["BL10:Exp:IM:ImageFilePath"]["value"][-1][0].decode("utf-8").strip()
            )

            if read_shutter_counts:
                # The image path is relative to the parent directory of the HDF5 file, so we need to resolve it
                image_path = file_path.parent.parent.joinpath(metadata["image_file_path"]).resolve()
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
                        list_shutter_counts = []

                        # stop loading shutter counts if we encounter a count of 0
                        with open(shutter_count_file) as txt_fh:
                            lines = txt_fh.readlines()
                            for _line in lines:
                                _, _value = _line.split()
                                if _value == "0":
                                    break
                                list_shutter_counts.append(float(_value))

                        metadata["shutter_counts"] = list_shutter_counts
                else:
                    logger.warning(
                        f"Image directory in metadata not found: {image_path}. Shutter counts will not be loaded."
                    )

        if "DASlogs" in f["entry"] and "BL10:Exp:Det" in f["entry"]["DASlogs"]:
            metadata["detector"] = f["entry"]["DASlogs"]["BL10:Exp:Det"]["value_strings"][-1][0].decode("utf-8").strip()

    logger.debug(f"Loaded metadata: {metadata}")

    return metadata
