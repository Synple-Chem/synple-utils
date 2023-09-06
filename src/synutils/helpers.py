import csv
import time
from functools import wraps
from pathlib import Path

from synutils.enum_logger import get_logger

LOGGER = get_logger()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        LOGGER.debug(
            f"Function {func.__name__} {kwargs.keys()} Took {total_time:.4f} seconds = {total_time/60:.1f} mins"
        )
        return result

    return timeit_wrapper


def get_csv_line_num(fpath: Path):
    """Get the number of lines in a file

    Args:
        fpath (Path): path to the file
    """
    with open(fpath, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line_number, _ in enumerate(csv_reader, start=1):
            pass
    return line_number
