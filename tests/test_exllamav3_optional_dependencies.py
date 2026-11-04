# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
from pathlib import Path


def test_exllamav3_quantizer_import_without_pydantic():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        import gptqmodel

        class NoPydantic(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pydantic" or fullname.startswith("pydantic."):
                    raise ModuleNotFoundError("pydantic intentionally unavailable", name="pydantic")

        for name in list(sys.modules):
            if name == "pydantic" or name.startswith("pydantic."):
                del sys.modules[name]
        sys.meta_path.insert(0, NoPydantic())

        from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
        from gptqmodel.exllamav3.util.memory import free_mem

        assert callable(quantize_tiles)
        free_mem()
        assert "pydantic" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
