# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import platform
import sys

from packaging.version import Version

from gptqmodel.utils.logger import setup_logger


log = setup_logger()

# Check if GIL (global interpreter lock) is controllable in this Python build.
# Starting from python 3.13 it is possible to disable GIL
def has_gil_control():
    return hasattr(sys, '_is_gil_enabled')

# Check if GIL (global interpreter lock) is enabled.
# Starting from python 3.13 it is possible to disable GIL
def has_gil_disabled():
    return has_gil_control() and not sys._is_gil_enabled()

# Check For Python >= 3.13.3
def gte_python_3_13_3():
    return Version(platform.python_version()) >= Version("3.13.3")

# Check For Python >= 3.14
def gte_python_3_14():
    return Version(platform.python_version()) >= Version("3.14")

# torch compile requires GIL=1 or python 3.13.3t with GIL=0
def log_gil_requirements_for(feature: str):
    log.warn.once(f"Feature `{feature}` requires Python < 3.14 and Python GIL enabled and Python >= 3.13.3T (T for Threading-Free edition of Python) plus Torch 2.8. Feature is currently skipped/disabled.")
