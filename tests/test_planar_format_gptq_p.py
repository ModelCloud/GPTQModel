# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU coverage for the planar `gptq_p` checkpoint format across 2..8 bits.

Covers:
1. planar layout round-trip for every gptq_p bit width (2,3,4,5,6,7,8)
2. planar 2/4/8 single-plane words are bit-identical to the continuous layout
3. planar 3-bit (2+1 planes): packer parity, distinct words from continuous
   3-bit, identical dequantized weights
4. QuantizeConfig format routing: 5/6/7 auto-route to gptq_p, explicit gptq_p
   is preserved and serialized as `checkpoint_format`
5. GPTQ v1 <-> v2 qzeros conversion round-trip for planar 3-bit modules
6. model_dequant helper planar dispatch for 3-bit
7. tiny-model quantize/save/load/generate lifecycle with format=gptq_p
"""

import json
import os
from pathlib import Path


# Keep this suite on CPU so it works on CPU-only runners.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.planar_packing import (
    PLANAR_FORMAT_BITS,
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


@pytest.mark.parametrize("bits", PLANAR_FORMAT_BITS)
def test_planar_rows_roundtrip_all_bits(bits: int):
    torch.manual_seed(bits)
    codes = _rand_codes(160, 48, bits)
    packed = planar_pack_rows(codes, bits)
    assert packed.dtype == torch.int32
    assert packed.shape == (160 * bits // 32, 48)
    assert torch.equal(planar_unpack_rows(packed, bits), codes)


@pytest.mark.parametrize("bits", PLANAR_FORMAT_BITS)
def test_planar_cols_roundtrip_all_bits(bits: int):
    torch.manual_seed(bits)
    codes = _rand_codes(96, 64, bits).T.contiguous()  # [64, 96]
    packed = planar_pack_cols(codes, bits)
    assert packed.dtype == torch.int32
    assert packed.shape == (64, 96 * bits // 32)
    assert torch.equal(planar_unpack_cols(packed, bits), codes)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_planar_single_plane_matches_continuous_words(bits: int):
    torch.manual_seed(bits)
    codes = _rand_codes(64, 16, bits).to(torch.int64)
    planar = planar_pack_rows(codes, bits)

    pack_factor = 32 // bits
    continuous = torch.zeros(64 * bits // 32, 16, dtype=torch.int64)
    for row in range(continuous.shape[0]):
        for j in range(pack_factor):
            continuous[row] |= codes[row * pack_factor + j] << (bits * j)
    continuous = (continuous & 0xFFFFFFFF).to(torch.int32)
    assert torch.equal(planar, continuous)


def _make_quant_inputs(bits: int, in_features: int = 64, out_features: int = 32, group_size: int = 32):
    torch.manual_seed(1000 + bits)
    maxq = (1 << bits) - 1
    groups = in_features // group_size
    linear = nn.Linear(in_features, out_features, bias=True)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    zeros = torch.randint(0, maxq + 1, (out_features, groups)).float()
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    return linear, scales, zeros, g_idx


def _new_module(bits: int, fmt: FORMAT, in_features: int = 64, out_features: int = 32,
                group_size: int = 32) -> TorchLinear:
    return TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=fmt,
        register_buffers=False,
    )


def test_planar_3bit_module_flag_and_packer_parity():
    linear, scales, zeros, g_idx = _make_quant_inputs(3)

    m_block = _new_module(3, FORMAT.GPTQ_P)
    assert m_block.planar is True
    m_block.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_orig = _new_module(3, FORMAT.GPTQ_P)
    m_orig.pack_original(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_threaded = _new_module(3, FORMAT.GPTQ_P)
    m_threaded.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=32, workers=4)

    assert torch.equal(m_block.qweight, m_orig.qweight)
    assert torch.equal(m_block.qzeros, m_orig.qzeros)
    assert torch.equal(m_block.qweight, m_threaded.qweight)
    assert torch.equal(m_block.qzeros, m_threaded.qzeros)


def test_planar_3bit_distinct_words_same_dequant():
    linear, scales, zeros, g_idx = _make_quant_inputs(3)

    m_planar = _new_module(3, FORMAT.GPTQ_P)
    m_planar.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_continuous = _new_module(3, FORMAT.GPTQ_V2)
    assert m_continuous.planar is False
    m_continuous.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    # Same logical codes, different word layouts.
    assert not torch.equal(m_planar.qweight, m_continuous.qweight)
    assert torch.equal(m_planar.dequantize_weight(), m_continuous.dequantize_weight())


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_planar_248_words_match_continuous_module(bits: int):
    linear, scales, zeros, g_idx = _make_quant_inputs(bits)

    m_planar = _new_module(bits, FORMAT.GPTQ_P)
    m_planar.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    m_continuous = _new_module(bits, FORMAT.GPTQ_V2)
    m_continuous.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    assert torch.equal(m_planar.qweight, m_continuous.qweight)
    assert torch.equal(m_planar.qzeros, m_continuous.qzeros)
    assert torch.equal(m_planar.dequantize_weight(), m_continuous.dequantize_weight())


def test_gptq_p_rejects_unsupported_bits():
    with pytest.raises((ValueError, NotImplementedError)):
        TorchLinear(
            bits=16,
            group_size=32,
            sym=False,
            desc_act=False,
            in_features=64,
            out_features=32,
            bias=False,
            format=FORMAT.GPTQ_P,
            register_buffers=False,
        )


@pytest.mark.parametrize("expect_planar", [True, False])
def test_post_init_rejects_3bit_layout_format_mismatch(expect_planar: bool):
    from gptqmodel.utils.model import gptqmodel_post_init

    module_format = FORMAT.GPTQ_V2 if expect_planar else FORMAT.GPTQ_P
    cfg_format = FORMAT.GPTQ_P if expect_planar else FORMAT.GPTQ_V2
    model = nn.Sequential(_new_module(3, module_format))
    cfg = QuantizeConfig(bits=3, format=cfg_format)
    with pytest.raises(ValueError, match="not interchangeable"):
        gptqmodel_post_init(model, use_act_order=False, quantize_config=cfg)


@pytest.mark.parametrize("bits", [5, 6, 7])
def test_config_auto_routes_planar_bits_to_gptq_p(bits: int):
    cfg = QuantizeConfig(bits=bits)
    assert cfg.format == FORMAT.GPTQ_P


@pytest.mark.parametrize("bits", PLANAR_FORMAT_BITS)
def test_config_accepts_explicit_gptq_p(bits: int):
    cfg = QuantizeConfig(bits=bits, format=FORMAT.GPTQ_P)
    assert cfg.format == FORMAT.GPTQ_P
    assert cfg.to_dict()["checkpoint_format"] == "gptq_p"


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_config_keeps_continuous_formats_for_legacy_bits(bits: int):
    cfg = QuantizeConfig(bits=bits)
    assert cfg.format != FORMAT.GPTQ_P


@pytest.mark.parametrize("bits", [5, 6, 7])
def test_rtn_config_auto_routes_planar_bits_to_gptq_p(bits: int):
    from gptqmodel.quantization.config import RTNConfig

    cfg = RTNConfig(bits=bits)
    assert cfg.format == FORMAT.GPTQ_P
    assert cfg.to_dict()["checkpoint_format"] == "gptq_p"


@pytest.mark.parametrize("bits", [5, 6, 7])
def test_dynamic_bit_overrides_route_to_gptq_p(bits: int):
    cfg = QuantizeConfig(bits=4, dynamic={"re:.*mlp.*": {"bits": bits}})
    assert cfg.format == FORMAT.GPTQ_P


def test_dynamic_exclusions_do_not_route_to_gptq_p():
    cfg = QuantizeConfig(bits=4, dynamic={"-re:.*skip.*": {}})
    assert cfg.format != FORMAT.GPTQ_P


def test_planar_3bit_qzeros_v1_v2_conversion_roundtrip():
    from gptqmodel.utils.model import (
        convert_gptq_v1_to_v2_format_module,
        convert_gptq_v2_to_v1_format_module,
    )

    linear, scales, zeros, g_idx = _make_quant_inputs(3)
    module = _new_module(3, FORMAT.GPTQ_P)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())

    qzeros_v2 = module.qzeros.clone()
    logical_v2 = planar_unpack_cols(qzeros_v2, 3)

    convert_gptq_v2_to_v1_format_module(module, quantize_config=QuantizeConfig(bits=3))
    logical_v1 = planar_unpack_cols(module.qzeros, 3)
    assert torch.equal(logical_v1, (logical_v2 - 1) & 0x7)

    convert_gptq_v1_to_v2_format_module(module, bits=3, pack_dtype=torch.int32)
    assert torch.equal(module.qzeros, qzeros_v2)


def test_model_dequant_helpers_planar_3bit_dispatch():
    from gptqmodel.utils.model_dequant import pack_cols, unpack_cols, unpack_rows

    torch.manual_seed(3)
    codes = _rand_codes(64, 32, 3)
    assert torch.equal(unpack_rows(planar_pack_rows(codes, 3), 3, planar=True), codes)

    cols_codes = codes.T.contiguous()
    packed = pack_cols(cols_codes, 3, pack_dtype=torch.int32, planar=True)
    assert torch.equal(planar_unpack_cols(packed, 3), cols_codes)
    assert torch.equal(unpack_cols(packed, 3, planar=True), cols_codes)


_CALIBRATION_TEXTS = [
    "tiny planar calibration sample one with enough tokens to survive minimum length filtering",
    "tiny planar calibration sample two exercising the planar gptq_p quantization path",
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
@pytest.mark.parametrize("bits", [3, 5])
def test_tiny_model_gptq_p_lifecycle(bits: int, tmp_path: Path):
    from gptqmodel import BACKEND, GPTQModel

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
        format=FORMAT.GPTQ_P,
        device="cpu",
    )

    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)

    with open(quantized_dir / "config.json") as fh:
        saved = json.load(fh)
    assert saved["quantization_config"]["checkpoint_format"] == "gptq_p"

    quantized_model = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    assert quantized_model.quantize_config.bits == bits
    assert quantized_model.quantize_config.format == FORMAT.GPTQ_P

    quantized_layers = [
        module for module in quantized_model.model.modules() if isinstance(module, TorchLinear)
    ]
    assert quantized_layers, "expected at least one quantized TorchLinear layer"
    assert all(module.bits == bits for module in quantized_layers)
    if bits == 3:
        assert all(module.planar for module in quantized_layers)

    encoded = tokenizer("tiny planar calibration", return_tensors="pt")
    output = quantized_model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=4,
        do_sample=False,
    )
    assert output.shape[-1] > encoded["input_ids"].shape[-1]
