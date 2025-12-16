"""
Conversion utilities for converting `.npz` files into `MonteCarloResult` dataclass.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Iterable

from src.strw_amuse.utils.config import MonteCarloResult

logger = logging.getLogger(__name__)


def to_MonteCarloResults(file_path: Path | str) -> MonteCarloResult:
    """
    MC results converter
    - Load mc results from `file_path`
    - Convert to `MonteCarloResult` dataclass

    Args:
        file_path (Path | str): File path of `.npz` file.

    Returns:
        MonteCarloResult: MonteCarloResult dataclass for analysis.
    """

    data = np.load(file_path, allow_pickle=True)

    result = MonteCarloResult(
        samples=data["samples"],
        param_names=list(data["param_names"]),
        distances=data["distances"],
        weights=data["weights"],
        all_star_outcomes=data["all_star_outcomes"],
        all_star_weights=data["all_star_weights"],
        sample_ids=data["sample_ids"],
        weighted_counts=data["weighted_counts"],
        probabilities=data["probabilities"],
        unique_outcomes=data["unique_outcomes"],
    )
    return result


def to_one_npz(
    dir_path: str | Path,
    file_path: str | Path,
    pattern: str = "mc_result_*.npz",
) -> None:
    """
    Combine all `.npz` files under mc run results dir into a single `.npz` file.
    - Assume all `.npz` share the same keys and arrays compatible for concatenation along axis=0.

    Args:
        dir_path (str | Path): Dir path to mc run saved results.
        file_path (str | Path): Dir path to save combined results.
        pattern (str, optional): File name pattern to look for. Defaults to "mc_result_*.npz".

    Returns:
        None: No returns expected.
    """
    dir_path = Path(dir_path)
    file_path = Path(file_path)

    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {dir_path}")

    # quick fix for metadata concat
    metadata_keys = {'param_names', 'unique_outcomes'}

    # get keys and basic shapes from first .npz
    first = np.load(files[0])
    keys: Iterable[str] = first.files

    # load and stack per key
    combined = {}
    for key in keys:
        arrays = [np.load(f)[key] for f in files]
        if key in metadata_keys:
            # use metadata from 1st file
            combined[key] = arrays[0]
            logger.debug(f"Metadata {key}: using first file (shape: {arrays[0].shape})")
        else:
            # concat along axis=0
            combined[key] = np.concatenate(arrays, axis=0)
            logger.debug(
                f"Data {key}: concatenated {len(files)} files (shape: {combined[key].shape})"
            )

    np.savez_compressed(file_path, **combined)
    logger.info("MC run results combined to %s.", file_path)

    return None
