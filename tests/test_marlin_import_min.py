import pytest
import torch

from gptqmodel import extension
from gptqmodel.nn_modules.qlinear import marlin as marlin_mod


def test_marlin_runtime_chain_test():
    """
    extension.load -> marlin_runtime_available -> (if error: marlin_runtime_error)
    """

    loaded = extension.load(name="marlin_fp16", use_cache=True)
    assert loaded.get("marlin_fp16") is True

    ok = marlin_mod.marlin_runtime_available(torch.float16)
    if not ok:
        pytest.fail("Marlin torch.ops kernels are not properly installed. Error: " + marlin_mod.marlin_runtime_error(torch.float16))
