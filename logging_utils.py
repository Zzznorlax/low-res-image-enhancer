import logging
import sys

from logging import Logger

LOGGER_DEFAULT_NAME = "default_logger"
LOGGERS = {}

DEFAULT_MESSAGE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str = LOGGER_DEFAULT_NAME, level: int = logging.DEBUG):

    logger = LOGGERS.get(name, None)

    if not isinstance(logger, Logger):
        logger = Logger(name=name)
        logger.setLevel(level=level)

        LOGGERS[name] = logger

    return logger


def add_stdout_handler(logger: Logger, level: int = logging.DEBUG, message_format: str = DEFAULT_MESSAGE_FORMAT):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(message_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
