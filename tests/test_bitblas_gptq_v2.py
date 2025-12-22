import os

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.bitblas import (
    BITBLAS_AVAILABLE,
    import_bitblas,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BitBLAS")
@pytest.mark.skipif(not BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_forward_pass():
    import_bitblas()

    device_index = int(os.environ.get("BITBLAS_TEST_DEVICE", 0))
    torch.cuda.set_device(device_index)

    # Load a dummy model (1.0 GB) to test if there are errors while converting to bitblas
    # Take a few minutes for compiling (1st run) and repacking (each time)
    GPTQModel.load("XXXXyu/Qwen3-1.7B-w2g64-gptq_v2", trust_remote_code=True, backend=BACKEND('bitblas'))
