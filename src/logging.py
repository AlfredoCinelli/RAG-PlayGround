"""
Module containig logging configuration and setup.

The main functions defined in the module are:

* setup_logger: Sets up logging configuration.
"""

# Import packages and modules

import logging
import os

from src.constants import LOG_FILE_DIR, LOG_FILE_NAME


def setup_logger(
    log_file_name: str = LOG_FILE_NAME,
    log_file_dir: str = LOG_FILE_DIR,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Function to set up logging configuration.

    :param log_file_name: name of the logging file, defaults to LOG_FILE_NAME
    :type log_file_name: str, optional
    :param log_file_dir: name of the logs directory, defaults to LOG_FILE_DIR
    :type log_file_dir: str, optional
    :param log_level: minimum level of log to be tracked, defaults to logging.INFO
    :type log_level: int, optional (one of the logging levels)
    :return: logger object (it can be invoked as logger.level("Message"))
    :rtype: logging.Logger
    """
    # Check directory existence
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)
    # Get log file full path
    log_full_path = os.path.join(log_file_dir, log_file_name)

    # Setup logger

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] [%(module)s]: %(message)s",
        level=log_level,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_full_path)],
    )

    return logging.getLogger(__name__)


logger = setup_logger()
