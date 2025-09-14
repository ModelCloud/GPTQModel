import platform
import sys

from gptqmodel.utils.logger import setup_logger
from packaging.version import Version

log = setup_logger()

# Check if GIL (global interpreter lock) is controllable in this Python build.
# Starting from python 3.13 it is possible to disable GIL
def has_gil_control():
    return hasattr(sys, '_is_gil_enabled')

# Check if GIL (global interpreter lock) is enabled.
# Starting from python 3.13 it is possible to disable GIL
def has_gil_disabled():
    return has_gil_control() and not sys._is_gil_enabled()

# Check For Python > 3.13.3
def gte_python_3_13_3():
    return Version(platform.python_version()) >= Version("3.13.3")

# torch compile requires GIL=1 or python 3.13.3t with GIL=0
def log_gil_requirements_for(feature: str):
    log.warn.once(f"Feature `{feature}` requires python GIL or Python >= 3.13.3T (T for Threading-Free edition of Python) plus Torch 2.8. Feature is currently skipped/disabled.")
