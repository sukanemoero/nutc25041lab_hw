from loguru import logger as base_logger
from os import environ as env
from sys import stderr

_Logger = base_logger.__class__

logger: _Logger = base_logger.bind()
logger.remove()
logger.add(stderr, level=env.get("LOG_LEVEL", "INFO").upper())


def change_logger(my_logger):
    global logger
    logger = my_logger
