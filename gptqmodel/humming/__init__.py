# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

try:
    from ._version import version as __version__
except ImportError:  # not built/installed (e.g. running from a source tree without a build)
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("humming-kernels")
    except Exception:
        __version__ = "0.0.0+unknown"

from . import ops
from . import dtypes
