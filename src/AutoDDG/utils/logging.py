import logging
import os

import torch
from colorama import Fore, Style, init

# Initialize colorama for cross-platform support
init(autoreset=True)


class JobFilter(logging.Filter):
    """
    A filter that dynamically adds the job ID from the job object
    passed to each logger.
    """

    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def filter(self, record):
        record.job_id = self.job_id or "none"
        return True


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log levels and Job IDs.
    """

    def format(self, record):
        log_colors = {
            "DEBUG": Fore.CYAN + Style.BRIGHT,
            "INFO": Fore.GREEN + Style.BRIGHT,
            "WARNING": Fore.YELLOW + Style.BRIGHT,
            "ERROR": Fore.RED + Style.BRIGHT,
            "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
        }

        # Get the color for the log level or use default
        color = log_colors.get(record.levelname, Fore.WHITE + Style.BRIGHT)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"

        # Format the Job ID in bold cyan
        if hasattr(record, "job_id"):
            record.job_id = f"{Fore.CYAN}{Style.BRIGHT}{record.job_id}{Style.RESET_ALL}"
        else:
            record.job_id = "none"

        return super().format(record)


def setup_logger(name, job=None):
    # Create a logger with the module's path as its name
    logger = logging.getLogger(name)

    # Grab LOG_LEVEL from environment, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Disable propagation to avoid duplicate logs
    logger.propagate = False

    # Only add handler if it doesn't exist yet
    if not logger.handlers:
        handler = logging.StreamHandler()

        # Set up the formatter with or without job context
        if job:
            logger.addFilter(JobFilter(job))
            formatter = ColoredFormatter(
                "%(asctime)s [%(name)s] %(job_id)s %(levelname)s: %(message)s"
            )
        else:
            formatter = ColoredFormatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_cuda_memory(
    logger,
    s: str = "",
):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"CUDA Memory [{s}] - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
        )
