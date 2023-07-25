import time
from functools import wraps

from synutils.enum_logger import get_logger

LOGGER = get_logger()


def timeit(func):
    # TODO: change print to logger
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        LOGGER.info(
            f"Function {func.__name__} {kwargs.keys()} Took {total_time:.4f} seconds = {total_time/60:.1f} mins"
        )
        return result

    return timeit_wrapper
