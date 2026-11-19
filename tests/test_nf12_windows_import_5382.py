# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The portable NF12 path must import and probe without ``fcntl`` (#5382).

``gptqmodel_nf12._jit`` imported ``fcntl`` at module scope, so on Windows both
``import gptqmodel.utils.nf12`` and the ``available()`` capability probe raised
``ModuleNotFoundError`` instead of reporting the native kernels as unavailable.
The JIT builder is only reachable on hosts that can build it, so its platform
imports must not gate the portable path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

_PROGRAM = """
import sys

# ``None`` in sys.modules makes ``import fcntl`` raise ImportError, which is
# exactly how a Windows host fails.
sys.modules["fcntl"] = None

import torch
torch.cuda.is_available = lambda: False

# Importing the portable shim (and, transitively, the whole gptqmodel package)
# must not pull in the CUDA JIT builder.
import gptqmodel.utils.nf12 as shim
assert "gptqmodel_nf12._jit" not in sys.modules, "the shim imported the JIT builder eagerly"

# The capability probe reports "unavailable" instead of raising.
assert shim.available() is False, shim.last_error()
assert "CUDA" in (shim.last_error() or "")

# The builder module itself is importable without fcntl.
import gptqmodel_nf12._jit as jit
assert jit.fcntl is None, jit.fcntl

# ... and it refuses to start a native build on Windows, where it could only
# produce an ELF shared object.
posix_platform = sys.platform
sys.platform = "win32"
assert jit.source_build_available() is False
sys.platform = posix_platform

import gptqmodel_nf12
assert isinstance(gptqmodel_nf12.available(), bool)

print("ok")
"""


def test_portable_nf12_import_and_probe_without_fcntl():
    """Fresh interpreter with ``fcntl`` unavailable, as on Windows."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env.setdefault("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROGRAM)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "ok" in result.stdout
