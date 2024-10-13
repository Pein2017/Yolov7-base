import logging
import sys
from logging.handlers import QueueHandler, RotatingFileHandler
from multiprocessing import Queue


def worker_logging_configurer(queue: Queue):
    """Configure logging for worker processes."""
    handler = QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def setup_global_logger(name: str, log_file: str, level=logging.INFO):
    """Set up a global logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_file_logger(log_file: str = "logs/training.log"):
    """Set up a file logger with rotating file handler."""
    logger = logging.getLogger("file_logger")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def setup_logger(
    log_file, name=__name__, level: str = "info", log_to_console=True, overwrite=True
):
    """
    Set up a logger to log messages to a file and optionally to the console.

    Parameters:
    - log_file (str): The path to the log file.
    - name (str): The name of the logger (default is the module name).
    - level (str): The logging level as a string (default is 'info'). Options: 'debug', 'info', 'warning', 'error', 'critical'.
    - log_to_console (bool): Whether to also log to the console (default is True).
    - overwrite (bool): Whether to overwrite the log file (default is True).

    Returns:
    - logger: A configured logger instance.
    """
    # Convert string level to logging level
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if level.lower() not in level_mapping:
        raise ValueError(
            f"Invalid log level: {level}. Choose from 'debug', 'info', 'warning', 'error', 'critical'."
        )

    log_level = level_mapping[level.lower()]

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Set the file mode to 'w' for overwrite, 'a' for append
    file_mode = "w" if overwrite else "a"

    # Create a file handler to write to a log file
    file_handler = logging.FileHandler(log_file, mode=file_mode)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger if it hasn't been added already
    if not logger.handlers:
        logger.addHandler(file_handler)

    # Optionally log to the console
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
