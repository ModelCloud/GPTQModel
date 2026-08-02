# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU correctness coverage for split-plane (planar) 5/6/7-bit GPTQ.

Covers, in dependency order:
1. planar layout round-trip (rows and cols, endpoint codes included)
2. packer parity: pack_block vs pack_original vs threaded pack_block
3. saturation: out-of-range weights clamp to 0/maxq instead of wrapping
4. torch CPU dequantize_weight against a logical-code reference
5. GPTQ v1 <-> v2 qzeros conversion round-trip
6. tiny-model quantize/save/load/generate lifecycle
"""

import os
from pathlib import Path


# Keep this suite on CPU so it works on CPU-only runners.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.planar_packing import (
    PLANAR_BITS,
    planar_pack_cols,
    planar_pack_rows,
    planar_unpack_cols,
    planar_unpack_rows,
)


pytestmark = [pytest.mark.cpu]


def _rand_codes(rows: int, cols: int, bits: int) -> torch.Tensor:
    maxq = (1 << bits) - 1
    codes = torch.randint(0, maxq + 1, (rows, cols), dtype=torch.int32)
    codes[0, :] = 0
    codes[1, :] = maxq
    return codes


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_planar_rows_roundtrip(bits: int):
    torch.manual_seed(bits)
    codes = _rand_codes(160, 48, bits)
    packed = planar_pack_rows(codes, bits)
    assert packed.dtype == torch.int32
    assert packed.shape == (160 * bits // 32, 48)
    assert torch.equal(planar_unpack_rows(packed, bits), codes)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_planar_cols_roundtrip(bits: int):
    torch.manual_seed(bits)
    codes = _rand_codes(96, 64, bits).T.contiguous()  # [64, 96]
    packed = planar_pack_cols(codes, bits)
    assert packed.dtype == torch.int32
    assert packed.shape == (64, 96 * bits // 32)
    assert torch.equal(planar_unpack_cols(packed, bits), codes)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_planar_rejects_misaligned_shapes(bits: int):
    with pytest.raises(ValueError):
        planar_pack_rows(torch.zeros((33, 4), dtype=torch.int32), bits)
    with pytest.raises(ValueError):
        planar_unpack_rows(torch.zeros((bits + 1, 4), dtype=torch.int32), bits)


def _make_quant_inputs(bits: int, in_features: int = 64, out_features: int = 32, group_size: int = 32):
    torch.manual_seed(1000 + bits)
    maxq = (1 << bits) - 1
    groups = in_features // group_size
    linear = nn.Linear(in_features, out_features, bias=True)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    zeros = torch.randint(0, maxq + 1, (out_features, groups)).float()
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    return linear, scales, zeros, g_idx


def _new_module(bits: int, in_features: int = 64, out_features: int = 32, group_size: int = 32) -> TorchLinear:
    return TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=False,
    )


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_packer_parity_and_threaded_pack(bits: int):
    linear, scales, zeros, g_idx = _make_quant_inputs(bits)

    m_block = _new_module(bits)
    m_block.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_orig = _new_module(bits)
    m_orig.pack_original(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_threaded = _new_module(bits)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("GPTQMODEL_PACK_THREADS", "4")
        m_threaded.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=32, workers=4)
    finally:
        monkeypatch.undo()

    assert torch.equal(m_block.qweight, m_orig.qweight)
    assert torch.equal(m_block.qzeros, m_orig.qzeros)
    assert torch.equal(m_block.qweight, m_threaded.qweight)
    assert torch.equal(m_block.qzeros, m_threaded.qzeros)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_pack_saturates_instead_of_wrapping(bits: int):
    maxq = (1 << bits) - 1
    in_features, out_features, group_size = 32, 32, 32
    linear = nn.Linear(in_features, out_features, bias=False)
    scales = torch.full((out_features, 1), 0.5)
    zeros = torch.full((out_features, 1), float(maxq // 2))
    g_idx = torch.zeros(in_features, dtype=torch.int32)

    # Weights that quantize far below 0 and far above maxq must saturate.
    with torch.no_grad():
        linear.weight.fill_(0.0)
        linear.weight[:, 0] = 0.5 * (maxq + 100 - maxq // 2)  # code maxq + 100 pre-clamp
        linear.weight[:, 1] = 0.5 * (-100 - maxq // 2)  # code -100 pre-clamp

    module = _new_module(bits, in_features=in_features, out_features=out_features, group_size=group_size)
    module.pack_block(linear, scales, zeros, g_idx)

    codes = planar_unpack_rows(module.qweight, bits)
    assert int(codes[0].min()) == maxq and int(codes[0].max()) == maxq
    assert int(codes[1].min()) == 0 and int(codes[1].max()) == 0
    assert int(codes.min()) >= 0 and int(codes.max()) <= maxq


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_dequantize_weight_matches_reference(bits: int):
    linear, scales, zeros, g_idx = _make_quant_inputs(bits)
    maxq = (1 << bits) - 1

    module = _new_module(bits)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    dequant = module.dequantize_weight()  # [in, out]

    weight = linear.weight.data  # [out, in]
    scale_full = scales[:, g_idx.long()]
    zero_full = zeros[:, g_idx.long()]
    # Mirror the packer's rounding formula exactly so ties resolve identically.
    codes_ref = torch.round((weight + zero_full * scale_full) / scale_full).clamp(0, maxq)

    codes_packed = planar_unpack_rows(module.qweight, bits)
    assert torch.equal(codes_packed.T.float(), codes_ref)

    # The module stores scales as float16; use the same precision for the reference.
    ref = (codes_ref - zero_full) * scale_full.to(torch.float16).float()
    assert torch.allclose(dequant.T.float(), ref.float(), atol=1e-4, rtol=0)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_qzeros_v1_v2_conversion_roundtrip(bits: int):
    from gptqmodel.utils.model import (
        convert_gptq_v1_to_v2_format_module,
        convert_gptq_v2_to_v1_format_module,
    )
    from gptqmodel.quantization.config import QuantizeConfig

    linear, scales, zeros, g_idx = _make_quant_inputs(bits)
    module = _new_module(bits)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    qzeros_v2 = module.qzeros.clone()
    logical_v2 = planar_unpack_cols(qzeros_v2, bits)

    convert_gptq_v2_to_v1_format_module(module, quantize_config=QuantizeConfig(bits=bits))
    logical_v1 = planar_unpack_cols(module.qzeros, bits)
    maxq = (1 << bits) - 1
    assert torch.equal(logical_v1, (logical_v2 - 1) & maxq)

    convert_gptq_v1_to_v2_format_module(module, bits=bits, pack_dtype=torch.int32)
    assert torch.equal(module.qzeros, qzeros_v2)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_model_dequant_helpers_roundtrip(bits: int):
    from gptqmodel.utils.model_dequant import pack_cols, unpack_cols, unpack_rows

    torch.manual_seed(bits)
    codes = _rand_codes(64, 32, bits)
    assert torch.equal(unpack_rows(planar_pack_rows(codes, bits), bits), codes)

    cols_codes = codes.T.contiguous()
    packed = pack_cols(cols_codes, bits, pack_dtype=torch.int32)
    assert torch.equal(unpack_cols(packed, bits), cols_codes)


_CALIBRATION_TEXTS = [
    "tiny planar calibration sample one with enough tokens to survive minimum length filtering",
    "tiny planar calibration sample two exercising the five six seven bit quantization path",
    "another synthetic calibration example that is intentionally verbose so filtering keeps it",
] * 2


def _build_tiny_llama_fixture(model_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir)
    return fast_tokenizer


@pytest.mark.slow
@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_tiny_model_quantize_save_load_generate(bits: int, tmp_path: Path):
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir()

    tokenizer = _build_tiny_llama_fixture(model_dir)
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )

    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=32,
        desc_act=False,
        device="cpu",
    )

    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)

    quantized_model = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    assert quantized_model.quantize_config.bits == bits

    quantized_layers = [
        module for module in quantized_model.model.modules() if isinstance(module, TorchLinear)
    ]
    assert quantized_layers, "expected at least one quantized TorchLinear layer"
    assert all(module.bits == bits for module in quantized_layers)

    encoded = tokenizer("tiny planar calibration", return_tensors="pt")
    output = quantized_model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=4,
        do_sample=False,
    )
    assert output.shape[-1] > encoded["input_ids"].shape[-1]
