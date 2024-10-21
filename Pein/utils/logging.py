import logging
import sys
from logging.handlers import QueueHandler
from multiprocessing import Queue


def worker_logging_configurer(queue: Queue):
    """Configure logging for worker processes."""
    # if not isinstance(queue, Queue):
    #     raise ValueError("queue must be a multiprocessing.Queue instance")
    handler = QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def setup_logger(
    log_file: str,
    name: str,
    level: str,
    log_to_console: bool,
    overwrite: bool,
):
    """
    Set up a logger with flexible options.

    Args:
        log_file (str): Path to the log file.
        name (str): Name of the logger.
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_to_console (bool): Whether to log to console in addition to file.
        overwrite (bool): Whether to overwrite existing log file or append to it.

    Raises:
        ValueError: If any required parameter is missing or invalid.
    """
    if not log_file:
        raise ValueError("log_file must be provided")
    if not name:
        raise ValueError("name must be provided")

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    file_mode = "w" if overwrite else "a"
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d-%H-%M-%S",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_listener(queue: Queue, log_file: str):
    """
    Listener process for handling log records from the queue.
    """
    logger = setup_logger(
        log_file,
        name="global_logger",
        level="INFO",
        log_to_console=True,
        overwrite=True,
    )
    while True:
        try:
            record = queue.get()
            if record is None:  # Sentinel to quit
                break
            logger.handle(record)
        except Exception:
            import traceback

            print("Logging problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
