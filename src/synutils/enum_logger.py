import logging
import sys
from pathlib import Path


def get_logger(
    name=None, level=logging.INFO, filename: Path | None = None
) -> logging.Logger:
    """Returns a logger that is configured as:
    - by default INFO level or higher messages are logged out in STDOUT.
    - format includes file name, line number, etc.

    Args:
        name (str, optional): Name of the logger. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
        filename (Path, optional): File to log to. Defaults to None, by default logs to STDOUT.

    usage:
        LOGGER = get_logger()
        LOGGER.info("some information")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        # Remove existing handlers so that capsys can capture
        # the output from patched sys.stdout
        for handler in logger.handlers:
            logger.removeHandler(handler)

    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    # Send everything to stdout
    handler_out = logging.StreamHandler(sys.stdout)
    handler_out.setFormatter(log_formatter)
    logger.addHandler(handler_out)

    # log to filename
    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger
