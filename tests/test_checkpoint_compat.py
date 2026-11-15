# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatibility regressions for issue #4013.

Every case here failed on the parent commit: 5.4.x checkpoints raised on a
legacy boolean `meta.gptaq`, an untied `lm_head=True` model raised
`Could not find the given module within the target model`, a tied one leaked a
`missing checkpoint tensors` error instead of the explicit rejection, AWQ `-:`
exclusions were absent from `modules_to_not_convert`, and the HF/vLLM
loadability gaps were silent.
"""

import os

import pytest
import torch

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq, unpack_awq
from gptqmodel.quantization.config import (
    FORMAT,
    GGUF_EXPORT_QUANT_METHOD,
    METHOD,
    FOEMConfig,
    GGUFConfig,
    GPTAQConfig,
)


# Qwen2 tokenizers are byte-level BPE; a local copy keeps the fixture offline.
QWEN2_TOKENIZER_SOURCES = (
    "/monster/data/model/Qwen2.5-0.5B-Instruct",
    "/monster/data/model/Qwen2-1.5B-Instruct",
)


def _tokenizer_source() -> str:
    for candidate in QWEN2_TOKENIZER_SOURCES:
        if os.path.isfile(os.path.join(candidate, "tokenizer.json")):
            return candidate
    pytest.skip("no local Qwen2 tokenizer available for the tiny-model fixture")
    raise RuntimeError("unreachable after pytest.skip")


def _build_tiny_qwen2(target_dir: str, *, tied: bool) -> str:
    """Write a 2-layer Qwen2 with a real tokenizer, tied or untied."""

    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    source = _tokenizer_source()
    tokenizer = AutoTokenizer.from_pretrained(source)
    # The head width must be a multiple of 16 so it stays eligible for the
    # AUTO_NF12 calibration-source path this regression exercises.
    vocab_size = ((len(tokenizer) + 15) // 16) * 16

    os.makedirs(target_dir, exist_ok=True)
    tokenizer.save_pretrained(target_dir)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=tied,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(target_dir, safe_serialization=True)
    return target_dir


CALIBRATION_TEXT = [
    "the quick brown fox jumps over the lazy dog and then keeps running "
    "through the field while the others watch"
] * 8


def _legacy_54x_payload(**meta) -> dict:
    """A `quantize_config.json` shaped exactly like a 5.4.x checkpoint."""

    return {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "quant_method": "gptq",
        "meta": {"gptaq": False, "gptaq_alpha": 0.25, "gptaq_memory_device": "auto", **meta},
    }


def test_legacy_boolean_gptaq_false_disables_gptaq():
    """`meta.gptaq = false` is a 5.4.x "GPTAQ off", not an invalid type."""

    config = QuantizeConfig.from_quant_config(_legacy_54x_payload())

    assert config.gptaq is None
    assert config.bits == 4


def test_legacy_boolean_gptaq_true_keeps_alpha_and_device():
    config = QuantizeConfig.from_quant_config(
        _legacy_54x_payload(gptaq=True, gptaq_alpha=0.4, gptaq_memory_device="cuda:0")
    )

    assert isinstance(config.gptaq, GPTAQConfig)
    assert config.gptaq.alpha == pytest.approx(0.4)
    assert config.gptaq.device == "cuda:0"


def test_legacy_boolean_gptaq_true_without_alpha_uses_default():
    payload = _legacy_54x_payload(gptaq=True)
    payload["meta"].pop("gptaq_alpha")

    config = QuantizeConfig.from_quant_config(payload)

    assert isinstance(config.gptaq, GPTAQConfig)
    assert config.gptaq.alpha == pytest.approx(GPTAQConfig().alpha)


@pytest.mark.parametrize("value,expected_none", [(False, True), (True, False)])
def test_legacy_boolean_foem_meta(value, expected_none):
    """The sibling normalizer follows the same on/off contract."""

    config = QuantizeConfig.from_quant_config(_legacy_54x_payload(foem=value))

    assert (config.foem is None) is expected_none
    if not expected_none:
        assert isinstance(config.foem, FOEMConfig)


def test_boolean_gptaq_and_foem_accepted_by_the_constructor():
    assert QuantizeConfig(gptaq=False).gptaq is None
    assert isinstance(QuantizeConfig(gptaq=True).gptaq, GPTAQConfig)
    assert QuantizeConfig(foem=False).foem is None
    assert isinstance(QuantizeConfig(foem=True).foem, FOEMConfig)


def test_awq_dynamic_exclusions_are_written_to_modules_to_not_convert():
    """HF/vLLM only skip modules named in `modules_to_not_convert`."""

    config = QuantizeConfig(
        bits=4,
        group_size=128,
        method=METHOD.AWQ,
        dynamic={
            "-:.*mlp.down_proj$": {},
            "model.layers.0.self_attn.o_proj": {},
            "-:model.layers.1.mlp.gate_proj": {},
        },
    )

    payload = config.to_dict()

    assert payload["modules_to_not_convert"] == [
        ".*mlp.down_proj$",
        "model.layers.1.mlp.gate_proj",
    ]
    # The exclusion rules stay authoritative in `dynamic`.
    assert "-:.*mlp.down_proj$" in payload["dynamic"]
    # And the compatibility key survives a GPT-QModel reload.
    assert QuantizeConfig.from_quant_config(payload).modules_to_not_convert == payload[
        "modules_to_not_convert"
    ]


def test_awq_without_exclusions_writes_no_modules_to_not_convert():
    payload = QuantizeConfig(bits=4, group_size=128, method=METHOD.AWQ).to_dict()

    assert "modules_to_not_convert" not in payload


def test_hf_compat_warning_for_quantized_lm_head():
    from gptqmodel.models.writer import hf_compat_warnings

    warnings = hf_compat_warnings(QuantizeConfig(bits=4, group_size=32, lm_head=True), FORMAT.GPTQ)

    assert len(warnings) == 1
    assert "lm_head" in warnings[0]
    assert "random-initializes" in warnings[0]
    assert hf_compat_warnings(QuantizeConfig(bits=4, group_size=32), FORMAT.GPTQ) == []


def test_hf_compat_warning_for_8bit_awq():
    from gptqmodel.models.writer import hf_compat_warnings

    warnings = hf_compat_warnings(
        QuantizeConfig(bits=8, group_size=128, method=METHOD.AWQ), FORMAT.GEMM
    )

    assert len(warnings) == 1
    assert "8-bit AWQ" in warnings[0]
    assert "[0, 2, 1, 3]" in warnings[0]
    assert hf_compat_warnings(
        QuantizeConfig(bits=4, group_size=128, method=METHOD.AWQ), FORMAT.GEMM
    ) == []


def test_gguf_config_publishes_non_colliding_quant_method():
    """Packed GGUF must not claim transformers' `gguf` quant method."""

    payload = GGUFConfig(bits="q4_k_m").to_dict()

    assert payload["method"] == "gguf"
    assert payload["quant_method"] == GGUF_EXPORT_QUANT_METHOD
    assert GGUF_EXPORT_QUANT_METHOD != "gguf"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert reloaded.method is METHOD.GGUF
    # A payload that only carries the published alias still resolves to GGUF.
    alias_only = dict(payload)
    alias_only.pop("method")
    assert QuantizeConfig.from_quant_config(alias_only).method is METHOD.GGUF


def test_legacy_gguf_quant_method_still_loads():
    """Already-published checkpoints keep their `quant_method: gguf`."""

    payload = GGUFConfig(bits="q4_k_m").to_dict()
    payload["quant_method"] = "gguf"

    assert QuantizeConfig.from_quant_config(payload).method is METHOD.GGUF


def test_hf_compat_warning_for_packed_gguf():
    from gptqmodel.models.writer import hf_compat_warnings

    warnings = hf_compat_warnings(GGUFConfig(bits="q4_k_m"), FORMAT.GGUF)

    assert len(warnings) == 1
    assert "GgufHfQuantizer" in warnings[0]
    assert GGUF_EXPORT_QUANT_METHOD in warnings[0]


def _reference_pack_8bit(codes: torch.Tensor) -> torch.Tensor:
    """Independent little-endian lane-order reference for 8-bit AWQ words."""

    rows, columns = codes.shape
    assert columns % 4 == 0
    packed = torch.zeros((rows, columns // 4), dtype=torch.int64)
    for row in range(rows):
        for word in range(columns // 4):
            value = 0
            for lane in range(4):
                value |= int(codes[row, word * 4 + lane].item()) << (8 * lane)
            packed[row, word] = value
    return packed.to(torch.int32)


def test_awq_8bit_lane_order_is_natural():
    """8-bit AWQ packs lane `i` from output column `i` (documented contract)."""

    codes = torch.arange(1, 9, dtype=torch.int64).reshape(1, 8)
    zeros = torch.ones_like(codes)

    qweight, qzeros = pack_awq(codes, zeros, 8)

    assert torch.equal(qweight, _reference_pack_8bit(codes))
    assert torch.equal(qzeros, _reference_pack_8bit(zeros))
    unpacked_weight, unpacked_zeros = unpack_awq(qweight, qzeros, 8, count=8)
    assert torch.equal(unpacked_weight, codes)
    assert torch.equal(unpacked_zeros, zeros)


def test_awq_8bit_vllm_order_would_swap_lanes_one_and_two():
    """The `[0, 2, 1, 3]` undo vLLM applies is not this checkpoint's order."""

    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    qweight, _ = pack_awq(codes, codes, 8)

    # Independent emulation of vLLM's `undo_pack[4] = {0, 2, 1, 3}`: it reads
    # source lane `i` as output column `undo[i]`.
    undo = (0, 2, 1, 3)
    vllm_order = [int((int(qweight[0, 0]) >> (8 * lane)) & 0xFF) for lane in range(4)]
    decoded = [vllm_order[undo.index(column)] for column in range(4)]

    assert decoded == [1, 3, 2, 4]
    assert decoded != codes[0].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lm_head quantization needs an accelerator")
def test_untied_lm_head_quantizes_and_reloads(tmp_path):
    model_dir = _build_tiny_qwen2(str(tmp_path / "untied"), tied=False)

    model = GPTQModel.load(model_dir, QuantizeConfig(bits=4, group_size=32, lm_head=True))
    model.quantize(CALIBRATION_TEXT, batch_size=2)

    assert isinstance(model.model.lm_head, BaseQuantLinear)

    saved_dir = tmp_path / "saved"
    model.tokenizer.save_pretrained(saved_dir)
    model.save(str(saved_dir))
    del model

    reloaded = GPTQModel.load(str(saved_dir), device_map="auto")
    assert isinstance(reloaded.model.lm_head, BaseQuantLinear)
    input_ids = reloaded.tokenizer("the quick brown fox", return_tensors="pt").input_ids.to(
        reloaded.model.device
    )
    with torch.inference_mode():
        logits = reloaded.model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lm_head quantization needs an accelerator")
def test_tied_lm_head_is_rejected_with_the_explicit_error(tmp_path):
    model_dir = _build_tiny_qwen2(str(tmp_path / "tied"), tied=True)
    model = GPTQModel.load(model_dir, QuantizeConfig(bits=4, group_size=32, lm_head=True))

    with pytest.raises(NotImplementedError, match="tied weights"):
        model.quantize(CALIBRATION_TEXT, batch_size=2)


def test_hf_compat_warning_for_defused_moe_checkpoint():
    """A per-expert checkpoint must warn that plain HF loading drops the experts."""

    from gptqmodel.models.writer import hf_compat_warnings

    class _QuantLinear(torch.nn.Module):
        QUANT_TYPE = "gptq"

    class _ExpertMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList()
            for _ in range(2):
                expert = torch.nn.Module()
                expert.gate_proj = _QuantLinear()
                expert.up_proj = _QuantLinear()
                expert.down_proj = _QuantLinear()
                self.experts.append(expert)

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _ExpertMlp()

    class _MoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([_Layer()])

    warnings = hf_compat_warnings(
        QuantizeConfig(bits=4, group_size=32),
        FORMAT.GPTQ,
        model=_MoE(),
    )

    assert len(warnings) == 1
    assert "experts.<n>" in warnings[0]
    assert "MISSING" in warnings[0] and "random-initializes" in warnings[0]
    # A dense model has no expert warning.
    dense = torch.nn.Module()
    dense.proj = _QuantLinear()
    assert hf_compat_warnings(
        QuantizeConfig(bits=4, group_size=32), FORMAT.GPTQ, model=dense
    ) == []


def test_saved_config_drops_auto_map_when_remote_code_is_not_copied(tmp_path):
    """A native save must not keep an `auto_map` that points at missing `.py` files."""

    import json

    from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

    from gptqmodel.models.writer import _save_model_configs_without_weights

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config.auto_map = {"AutoConfig": "configuration_slimmoe.PhiMoEConfig"}
    model = LlamaForCausalLM(config).eval()

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"auto_map": config.auto_map}), encoding="utf-8"
    )

    native_out = tmp_path / "native"
    _save_model_configs_without_weights(
        model, str(native_out), str(source), trust_remote_code=False
    )
    saved = json.loads((native_out / "config.json").read_text(encoding="utf-8"))
    assert "auto_map" not in saved
    # The exported config still loads natively, including with the model-card
    # instruction that asks for remote code.
    assert AutoConfig.from_pretrained(str(native_out), trust_remote_code=True).model_type == "llama"

    remote_out = tmp_path / "remote"
    _save_model_configs_without_weights(
        model, str(remote_out), str(source), trust_remote_code=True
    )
    saved = json.loads((remote_out / "config.json").read_text(encoding="utf-8"))
    assert saved["auto_map"] == config.auto_map
