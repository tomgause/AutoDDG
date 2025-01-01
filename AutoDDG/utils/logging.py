import logging
from pathlib import Path


def setup_logging(
    logger_name=__name__,
    log_level_console=logging.DEBUG,
    log_level_file=logging.DEBUG,
    log_file_path="processing.log",
):
    """
    Configure logging for the application.

    Args:
        log_level_console (int): Logging level for the console handler.
        log_level_file (int): Logging level for the file handler.
        log_file_path (str or Path): Path to the log file.
        logger_name (str): Name of the logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter

    # Prevent adding multiple handlers if the logger is already configured
    if not logger.handlers:
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_console)

        # Ensure log_file_path is a Path object
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level_file)

        # Create formatter and add it to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
