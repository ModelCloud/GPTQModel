# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import platform
import sys
import sysconfig

from packaging.version import Version

from gptqmodel.utils.logger import setup_logger


log = setup_logger()

# Check if this Python build supports free-threading / GIL control.
# Starting from python 3.13 it is possible to disable GIL at build time.
def is_free_threading_build():
    """Return True when Python was built with free-threading support."""
    py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    try:
        return int(py_gil_disabled or 0) == 1
    except (TypeError, ValueError):
        return False


# Check if GIL (global interpreter lock) is controllable in this Python build.
# Starting from python 3.13 it is possible to disable GIL.
def has_gil_control():
    return is_free_threading_build()

# Check if GIL (global interpreter lock) is enabled at runtime.
# Starting from python 3.13 it is possible to disable GIL.
def has_gil_disabled():
    gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return has_gil_control() and callable(gil_enabled) and not gil_enabled()

# Check For Python >= 3.13.3
def gte_python_3_13_3():
    return Version(platform.python_version()) >= Version("3.13.3")

# Check For Python >= 3.14
def gte_python_3_14():
    return Version(platform.python_version()) >= Version("3.14")

# torch compile requires GIL=1 or python 3.13.3t with GIL=0
def log_gil_requirements_for(feature: str):
    log.warn.once(f"Feature `{feature}` requires Python < 3.14 and Python GIL enabled and Python >= 3.13.3T (T for Threading-Free edition of Python) plus Torch 2.8. Feature is currently skipped/disabled.")
