import time
import argparse
import logging
from typing import Callable, Any

def setup_logging(args: argparse.Namespace) -> None:
    """
    Configure logging based on verbosity arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    elif args.quiet:
        logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

def time_it(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """
    Measures the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to time.
        *args (Any): Positional arguments for the function.
        **kwargs (Any): Keyword arguments for the function.

    Returns:
        tuple[float, Any]: A tuple containing the elapsed time in ms 
                           and the result of the function call.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = (time.time() - start_time) * 1000
    return elapsed_time, result