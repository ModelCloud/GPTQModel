import sys

from gptqmodel.utils.logger import setup_logger

log = setup_logger()

# Check if GIL (global interpreter lock) is enabled.
# Starting from python 3.13 it is possible to disable GIL
def has_gil():
    if hasattr(sys, '_is_gil_enabled'):
        return sys._is_gil_enabled()

    return True

def log_gil_required(feature: str):
    log.warn.once(f"Feature `{feature}` requires python GIL. Feature is currently skipped/disabled.")
