import logging

logger = None
def setup_logger():
    global logger
    if logger is not None:
        return logger

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
