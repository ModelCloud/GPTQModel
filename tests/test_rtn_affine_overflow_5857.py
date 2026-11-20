# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""BF16 RTN must not overflow a finite affine span before fitting the scale.

The public ``GPTQModel.quantize()`` path with BF16 asymmetric RTN exported as
AWQ GEMM used to form ``xmax - xmin`` in the weight's compute dtype (FP32 for
RTN).  A finite group with endpoints ``+/-2**127`` has a representable
``span / 15``, but the FP32 subtraction overflows to ``inf`` first.  The
resulting infinite scale is stored in the checkpoint and every packed code
derived from it is invalid.
"""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq
from gptqmodel.quantization.config import FORMAT, RTNConfig
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.utils.importer import get_auto_backend_kernel_mapping


TARGET = "model.layers.0.self_attn.q_proj"
BITS = 4
GROUP_SIZE = 16
WIDTH = 32


def _save_tiny_llama(directory, row):
    """Save a one-layer BF16 Llama whose q_proj row 0 first group is ``row``."""

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=WIDTH,
        intermediate_size=2 * WIDTH,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"

    tokenizer = Tokenizer(models.WordLevel({f"w{i}": i for i in range(config.vocab_size)}, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="w0", pad_token="w0").save_pretrained(directory)

    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    weight = model.get_submodule(TARGET).weight
    with torch.no_grad():
        weight[0, :GROUP_SIZE] = row
    model.save_pretrained(directory)
    return weight.detach().clone()


def _quantize(directory):
    qcfg = RTNConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=False,
        format=FORMAT.GEMM,
        device="cpu",
        offload_to_disk=False,
    )
    model = GPTQModel.load(str(directory), qcfg, dtype=torch.bfloat16, backend=BACKEND.AWQ_TORCH)
    model.quantize(backend=BACKEND.AWQ_TORCH)
    return model.model.get_submodule(TARGET)


def _natural_codes(layer):
    codes, zeros = unpack_awq(layer.qweight, layer.qzeros, BITS)
    codes, zeros = reverse_awq_order(codes, zeros, BITS)
    return codes.to(torch.int64), zeros.to(torch.int64)


def _expected_affine(row, scale):
    """Independent FP64 nearest-code oracle for one group at a stored scale."""

    row = row.double()
    zero = torch.round(-row.min() / scale.double())
    codes = (torch.round(row / scale.double()) + zero).clamp(0, (1 << BITS) - 1)
    return zero.to(torch.int64), codes.to(torch.int64)


def test_bf16_rtn_affine_scale_stays_finite_at_extreme_endpoints_5857(tmp_path):
    torch.manual_seed(5857)
    get_auto_backend_kernel_mapping()
    extreme_row = torch.zeros(GROUP_SIZE, dtype=torch.bfloat16)
    extreme_row[0] = 2.0**127
    extreme_row[1] = -(2.0**127)
    weight = _save_tiny_llama(tmp_path, extreme_row)
    layer = _quantize(tmp_path)

    row = weight[0, :GROUP_SIZE]
    expected_scale = ((row.max().double() - row.min().double()) / ((1 << BITS) - 1)).float()
    stored_scale = layer.scales[0, 0]

    assert torch.isfinite(stored_scale)
    torch.testing.assert_close(stored_scale, expected_scale.to(stored_scale.dtype), rtol=0, atol=0)
    assert 2.2685e37 < stored_scale.item() < 2.2686e37

    codes, zeros = _natural_codes(layer)
    expected_zero, expected_codes = _expected_affine(row, stored_scale)
    assert zeros[0, 0].item() == expected_zero.item()
    assert torch.equal(codes[:GROUP_SIZE, 0], expected_codes)
    assert codes[0, 0].item() == 14
    assert codes[1, 0].item() == 0


def test_bf16_rtn_affine_normal_range_results_unchanged_5857(tmp_path):
    torch.manual_seed(5857)
    get_auto_backend_kernel_mapping()
    normal_row = torch.linspace(-1.0, 1.0, GROUP_SIZE).to(torch.bfloat16)
    weight = _save_tiny_llama(tmp_path, normal_row)
    layer = _quantize(tmp_path)

    row = weight[0, :GROUP_SIZE].float()
    # The pre-fix formula is exact for a finite span; the repair must leave it
    # bitwise-identical.
    expected_scale = (row.max() - row.min()) / ((1 << BITS) - 1)
    stored_scale = layer.scales[0, 0]
    assert torch.isfinite(stored_scale)
    torch.testing.assert_close(stored_scale, expected_scale, rtol=0, atol=0)

    codes, zeros = _natural_codes(layer)
    expected_zero, expected_codes = _expected_affine(row, stored_scale)
    assert zeros[0, 0].item() == expected_zero.item()
    assert torch.equal(codes[:GROUP_SIZE, 0], expected_codes)


def test_bf16_symmetric_affine_scale_stays_finite_5857():
    """The symmetric zero-centered range uses the same overflow-prone span."""

    qcfg = RTNConfig(bits=BITS, group_size=GROUP_SIZE, sym=True, device="cpu")
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    x = torch.zeros(1, GROUP_SIZE, dtype=torch.bfloat16)
    x[0, 0] = 2.0**127
    x[0, 1] = -(2.0**127)
    quantizer.find_params(x, weight=True)

    expected = ((torch.tensor(2.0**127, dtype=torch.float64) * 2) / ((1 << BITS) - 1)).float()
    assert torch.isfinite(quantizer.scale).all()
    torch.testing.assert_close(quantizer.scale[0, 0], expected, rtol=0, atol=0)
    assert quantizer.zero[0, 0].item() == ((1 << BITS) // 2)
