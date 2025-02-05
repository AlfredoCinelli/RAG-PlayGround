"""
Module containig logging configuration and setup.

The main class is Logger that behaves as Singleto class
by inheriting from the SingletonMeta to enforce just one
single instance of the logger exists.
"""

# Import packages and modules

import logging
import os
import typing

from typing_extension import Self

from src.constants import LOG_FILE_DIR, LOG_FILE_NAME


class SingletonMeta(type):
    """Metaclass for implementing the Singleton pattern."""

    _instances = {}

    @typing.no_type_check
    def __call__(
        cls: object,
        *args,
        **kwargs,
    ) -> object:
        """
        Ensure only one instance of the class is created.

        :param *args: Variable length argument list
        :param **kwargs: Arbitrary keyword arguments
        :return: single instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):
    """Singleton Logger class that ensures only one logger instance exists."""

    def __init__(
        self: Self,
        log_file_name: str = LOG_FILE_NAME,
        log_file_dir: str = LOG_FILE_DIR,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Initialize the singleton logger.

        :param log_file_name: name of the log file
        :type log_file_name: str
        :param log_file_dir: directory for log file
        :type log_file_dir: str
        :param log_level: minimum level of log to be tracked
        :type log_level: int
        """
        self._logger: logging.Logger | None = None
        self.setup_logger(log_file_name, log_file_dir, log_level)

    def setup_logger(
        self: Self,
        log_file_name: str = LOG_FILE_NAME,
        log_file_dir: str = LOG_FILE_DIR,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Set up logging configuration.

        :param log_file_name: name of the log file
        :type log_file_name: str
        :param log_file_dir: directory for log file
        :type log_file_dir: str
        :param log_level: minimum level of log to be tracked
        :type log_level: int

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

        self._logger = logging.getLogger(__name__)

    def get_logger(self: Self) -> logging.Logger:
        """
        Get the logger instance.

        :return: configured logger instance
        :rtype: logging.Logger
        """
        if self._logger is None:
            raise RuntimeError("Logger not initialized")
        return self._logger


# Create a single instance that can be imported and used throughout the application
logger = Logger().get_logger()
