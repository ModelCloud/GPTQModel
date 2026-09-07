# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitblas import BitblasBaseQuantLinear


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_zero_buffer_uses_actual_storage_word_width(bits):
    module = object.__new__(BitblasBaseQuantLinear)
    torch.nn.Module.__init__(module)
    module.bits = bits
    module.group_size = 32
    module.TORCH_DTYPE = torch.float16
    module.quant_config = SimpleNamespace(
        with_zeros=True,
        torch_storage_dtype=torch.int8,
        pack_factor=32 // bits,
    )
    module.bitblas_matmul = SimpleNamespace(
        retrieve_weight_shape=lambda: (64, 64 * bits // 8)
    )
    module._initialize_buffers(64, 64, False)
    assert module.qzeros.shape == (2, 64 * bits // 8)
    assert module.qzeros.dtype == torch.int8
    assert module.zeros is module.qzeros
