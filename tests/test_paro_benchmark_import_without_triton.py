# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The Paro benchmark helpers must import on hosts without Triton.

``gptqmodel.utils.paro_benchmark`` also owns CPU-only helpers
(``make_paro_config``, calibration tokenization). It used to import
``ParoTritonLinear`` at module scope, so on a CPU-only host every importer,
including the Paro unit tests, crashed with ``No module named 'triton'``.
"""

import subprocess
import sys


def test_paro_benchmark_imports_when_triton_is_unavailable():
    # ``sys.modules[name] = None`` makes ``import name`` raise ImportError,
    # which reproduces a Triton-less host even where Triton is installed.
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "from gptqmodel.utils.paro_benchmark import make_paro_config, _tokenize_calibration_sample\n"
        "assert callable(make_paro_config) and callable(_tokenize_calibration_sample)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-4000:]
