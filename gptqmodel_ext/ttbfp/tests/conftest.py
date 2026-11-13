# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Test bootstrap for the ttbfp extension.

``make test`` builds ``libttbfp.so`` and points ``TTBFP_LIB`` at it.  A
root-level ``pytest`` run would otherwise collect these tests without the
library built and fail at import, so skip this directory unless the library is
present.  Repository CI runs ``pytest tests/<script>.py`` and never reaches it.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LIB = Path(os.environ.get("TTBFP_LIB") or (_ROOT / "libttbfp.so"))
if _LIB.is_file():
    sys.path.insert(0, str(_ROOT / "python"))
else:
    collect_ignore_glob = ["test_*.py"]
