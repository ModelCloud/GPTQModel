# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The GPTQ_TORCH eval decode must stay finite for finite inputs (#5606).

The packed fixture decodes to ``(15 - 0) * 8192 = 122880`` on every element.
That weight is finite in FP32 and exactly representable in BF16, but not in
FP16. The eval fast path used to multiply in the stored FP16 scale dtype, so
the linear returned NaN (``0 * inf``) and the embedding returned inf. The
train-mode decode and both embedding lookup routes must agree with an
independently computed BF16 oracle.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT


BITS = 4
GROUP_SIZE = 32
FEATURES = 32
SCALE = 8192.0
DECODED = ((1 << BITS) - 1) * SCALE  # 122880
X_VALUE = 0.25


def _fixture(cls, *, compile: bool = False):
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=FEATURES,
        out_features=FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        compile=compile,
    )
    module.qweight.fill_(-1)  # every nibble is 15
    module.qzeros.zero_()     # zero point 0
    module.scales.fill_(SCALE)
    module.post_init()
    return module


def _linear_oracle(dtype: torch.dtype) -> torch.Tensor:
    x = torch.zeros(1, FEATURES, dtype=torch.float64)
    x[:, 0] = X_VALUE
    weight = torch.full((FEATURES, FEATURES), DECODED, dtype=torch.float64)
    return (x @ weight).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_torch_linear_eval_decode_matches_the_wide_oracle(dtype):
    module = _fixture(TorchLinear)
    x = torch.zeros(1, FEATURES, dtype=dtype)
    x[:, 0] = X_VALUE
    oracle = _linear_oracle(dtype)

    module.eval()
    weights = module.dequantize_weight()
    assert torch.isfinite(weights).all(), "eval dequantize returned a non-finite weight"
    torch.testing.assert_close(
        weights.double(),
        torch.full_like(weights.double(), DECODED),
        rtol=0,
        atol=0,
    )
    with torch.inference_mode():
        eval_out = module(x)

    module.train()
    train_out = module(x)

    for name, out in (("eval", eval_out), ("train", train_out)):
        assert torch.isfinite(out).all(), f"{name} forward returned {out.flatten()[:4]}"
        torch.testing.assert_close(out, oracle, rtol=0, atol=0)


def test_torch_quant_embeddings_eval_decode_matches_both_lookup_routes():
    module = _fixture(TorchQuantEmbeddings)
    module.set_embedding_output_dtype(torch.bfloat16)
    module.eval()

    input_ids = torch.tensor([0, 7])
    oracle = torch.full((2, FEATURES), DECODED, dtype=torch.bfloat16)

    # Small table: the whole-table F.embedding path.
    whole_table = module(input_ids)
    assert torch.isfinite(whole_table).all(), f"whole-table lookup returned {whole_table.flatten()[:4]}"
    torch.testing.assert_close(whole_table, oracle, rtol=0, atol=0)

    # Force the large-table row gather, which multiplies the gathered scales.
    module.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 0
    row_gather = module(input_ids)
    assert torch.isfinite(row_gather).all(), f"row lookup returned {row_gather.flatten()[:4]}"
    torch.testing.assert_close(row_gather, oracle, rtol=0, atol=0)
