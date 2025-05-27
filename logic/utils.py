import time
from typing import Callable, Any

def time_it(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """
    Measures the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to time.
        *args (Any): Positional arguments for the function.
        **kwargs (Any): Keyword arguments for the function.

    Returns:
        tuple[float, Any]: A tuple containing the elapsed time in seconds 
                           and the result of the function call.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return elapsed_time, result