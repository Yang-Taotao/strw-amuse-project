"""
Logger utilities.
- Use `setup_logging` to configure a root logger with a console
- Optional per-rank file handler (MPI-aware if `mpi4py` is present).
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from .config import OUTPUT_DIR_LOGS

# Import `mpi4py` if possible. ImportError is expected
# If mpi4py is not installed; use `MPI=None` to run without mpi support.
try:
    from mpi4py import MPI  # type: ignore
except ImportError:  # pragma: no cover - mpi4py may not be installed in tests
    MPI = None


def _get_mpi_rank() -> None | int:
    """
    Return the MPI rank if available, otherwise `None`.

    The helper checks whether `mpi4py` was imported successfully and
    attempts to access `COMM_WORLD.Get_rank()`. If the MPI runtime or
    `COMM_WORLD` is not available the function returns `None`.
    """
    # local check
    if MPI is None:
        return None
    # Access COMM_WORLD and catch attribute errors
    # See if the MPI object is present but has not `COMM_WORLD` for some reason.
    try:
        comm = getattr(MPI, "COMM_WORLD", None)
        if comm is None:
            return None
        return int(comm.Get_rank())
    except AttributeError:
        return None


def setup_logging(
    rank=None, log_dir=OUTPUT_DIR_LOGS, level=logging.INFO, name="src.strw_amuse"
) -> logging.Logger:
    """
    Configure logging for the package.

    Parameters
    ----------
    rank: int or None
        If None, an MPI rank will be attempted to be detected via `mpi4py`.
    log_dir: str
        Directory where per-rank log files will be written if rank is not None.
    level: logging level
    name: logger name (default `src.strw_amuse` to match package module names)
    """
    if rank is None:
        rank = _get_mpi_rank()

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if rank is not None:
        logfile = os.path.join(log_dir, f"{name}_rank{rank}.log")
    else:
        logfile = os.path.join(log_dir, f"{name}.log")

    fh = RotatingFileHandler(logfile, maxBytes=10_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
