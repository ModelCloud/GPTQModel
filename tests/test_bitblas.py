import os

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitblas import (
    BITBLAS_AVAILABLE,
    BitblasQuantLinear,
    import_bitblas,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BitBLAS")
@pytest.mark.skipif(not BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_forward_pass():
    import_bitblas()

    device_index = int(os.environ.get("BITBLAS_TEST_DEVICE", 0))
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)

    layer = BitblasQuantLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=32,
        out_features=32,
        bias=False,
    ).to(device)

    with torch.no_grad():
        layer.qweight.zero_()
        layer.scales.zero_()
        if layer.quant_config.with_zeros:
            layer.qzeros.zero_()

    x = torch.randn(2, 32, device=device, dtype=layer.TORCH_DTYPE)
    y = layer(x)

    assert y.shape == (2, 32)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-4, rtol=1e-4)
