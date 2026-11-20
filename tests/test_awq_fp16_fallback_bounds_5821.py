# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""AWQ fallback strategies must survive large finite FP16 weight ranges.

Issue #5821: ``_awq_strategy_bounds`` formed STDCLIP's ``3 * std`` in the
weight dtype. Alternating ``-60000``/``60000`` weights have a population
standard deviation of 60000, so ``3 * std`` overflowed FP16 to ``inf``. The
scale became ``inf`` and ``0 * inf`` made the staged reconstruction NaN, which
tripped ``pseudo_quantize_tensor``'s exit assertion even though the selected
INT4 codes decode finitely. MEAN/MEDIAN failed the same way when
``center +- deviation`` left the FP16 range.
"""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig, Fallback, FallbackStrategy
from gptqmodel.utils.importer import get_auto_backend_kernel_mapping


BITS = 4
GROUP_SIZE = 16
MAX_INT = 2 ** (BITS - 1) - 1
MIN_INT = -(2 ** (BITS - 1))
TARGET = "model.layers.0.mlp.down_proj"


class _Stub:
    def __init__(self, sym: bool) -> None:
        self.qcfg = AWQConfig(bits=BITS, group_size=GROUP_SIZE, sym=sym)

    pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor


def _overflow_weights() -> torch.Tensor:
    """The issue's two finite FP16 rows whose selected intervals exceed FP16."""

    alternating = torch.tensor([-60000.0, 60000.0] * (GROUP_SIZE // 2), dtype=torch.float16)
    skewed = torch.tensor([-50000.0] + [50000.0] * (GROUP_SIZE - 1), dtype=torch.float16)
    return torch.stack((alternating, skewed))


@pytest.mark.parametrize("sym", [True, False], ids=["symmetric", "affine"])
@pytest.mark.parametrize(
    "strategy",
    [FallbackStrategy.MEAN, FallbackStrategy.MEDIAN, FallbackStrategy.STDCLIP],
    ids=["mean", "median", "stdclip"],
)
def test_fp16_large_finite_fallback_bounds_stay_finite(strategy, sym):
    weights = _overflow_weights()
    staged, scales, zeros = _Stub(sym).pseudo_quantize_tensor(weights.clone(), strategy=strategy)

    assert scales.dtype == weights.dtype
    assert zeros.dtype == weights.dtype
    assert torch.isfinite(scales).all()
    assert torch.isfinite(staged).all()
    assert staged.abs().max() <= torch.finfo(weights.dtype).max
    # Every staged weight is a finite grid point the stored metadata decodes.
    if sym:
        codes = staged / scales
        assert torch.isfinite(codes).all()
        assert codes.abs().max() <= max(abs(MIN_INT), MAX_INT) + 1
    else:
        codes = torch.round(staged / scales) + zeros
        assert torch.isfinite(codes).all()
        assert codes.min() >= 0
        assert codes.max() <= 2**BITS - 1


def test_fp16_stdclip_symmetric_uses_the_finite_issue_grid():
    """The issue's oracle: amplitude 180000 -> FP16 scale 25712, codes +-2."""

    weights = _overflow_weights()[:1]
    staged, scales, _ = _Stub(True).pseudo_quantize_tensor(weights.clone(), strategy=FallbackStrategy.STDCLIP)

    expected_scale = torch.tensor(180000.0 / MAX_INT, dtype=torch.float16)
    expected_codes = torch.tensor([-2, 2] * (GROUP_SIZE // 2), dtype=torch.float16)
    assert scales[0, 0] == expected_scale
    torch.testing.assert_close(staged[0], expected_codes * expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("sym", [True, False], ids=["symmetric", "affine"])
@pytest.mark.parametrize(
    "strategy",
    [FallbackStrategy.MIDPOINT, FallbackStrategy.MEAN, FallbackStrategy.MEDIAN, FallbackStrategy.STDCLIP],
    ids=["midpoint", "mean", "median", "stdclip"],
)
def test_normal_range_fallback_grid_stays_finite_and_accurate(strategy, sym):
    generator = torch.Generator().manual_seed(5821)
    weights = (torch.randn(8, 64, generator=generator) * 0.2).half()
    staged, scales, zeros = _Stub(sym).pseudo_quantize_tensor(weights.clone(), strategy=strategy)

    assert torch.isfinite(scales).all()
    assert torch.isfinite(staged).all()
    # Ordinary ranges keep the pre-fix grid: the reconstruction error stays
    # within one code step of the selected scale.
    assert (staged.float() - weights.float()).abs().max() <= scales.max().item()


def _write_tiny_llama(root) -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    model.model.layers[0].mlp.up_proj.weight.data.mul_(1e-4)
    model.get_submodule(TARGET).weight.data[0].copy_(torch.tensor([-60000.0, 60000.0] * 32, dtype=torch.float16))
    with torch.inference_mode():
        assert model(torch.tensor([list(range(3, 35))])).logits.isfinite().all()
    model.save_pretrained(root)

    tokenizer = Tokenizer(models.WordLevel({f"w{i}": i for i in range(128)}, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="w0", pad_token="w0").save_pretrained(root)


def test_public_quantize_stdclip_fallback_on_large_finite_fp16(tmp_path):
    torch.set_num_threads(1)
    get_auto_backend_kernel_mapping()
    ids = list(range(3, 35))
    calibration = CalibrationConfig(
        dataset=[{"input_ids": ids, "attention_mask": [1] * len(ids)}],
        min_length=1,
    )
    _write_tiny_llama(tmp_path)

    def quantize(strategy: str):
        qcfg = AWQConfig(
            bits=BITS,
            group_size=GROUP_SIZE,
            sym=True,
            device="cpu",
            offload_to_disk=False,
            fallback=Fallback(strategy=strategy, threshold=4096),
        )
        model = GPTQModel.load(str(tmp_path), qcfg, dtype=torch.float16, backend=BACKEND.AWQ_TORCH)
        model.quantize(calibration, backend=BACKEND.AWQ_TORCH)
        return model

    # The RTN control already completed on the audited commit.
    rtn = quantize("rtn").model.get_submodule(TARGET)
    assert rtn.scales.isfinite().all()

    model = quantize("stdclip")
    actual = model.model.get_submodule(TARGET)
    assert actual.scales.isfinite().all()

    expected_scale = torch.tensor(180000.0 / MAX_INT, dtype=torch.float16)
    assert actual.scales[0, 0] == expected_scale
    signed_codes = (actual.qweight[:GROUP_SIZE, 0] & 0xF) - 2 ** (BITS - 1)
    torch.testing.assert_close(
        signed_codes,
        torch.tensor([-2, 2] * (GROUP_SIZE // 2), dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    with torch.inference_mode():
        logits = model.model(torch.tensor([ids])).logits
    assert torch.isfinite(logits).all()
