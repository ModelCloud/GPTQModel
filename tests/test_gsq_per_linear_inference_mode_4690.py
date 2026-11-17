# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-linear GSQ must not break the staging Parameter's inference mode (#4690).

AWQ stages the refined weight with ``linear_layer.weight.data = wq`` while the
looper is inside ``torch.inference_mode()``. ``refine_affine_scalar`` runs under
``torch.inference_mode(False)`` so the search can use autograd; handing back a
normal tensor flipped that inference Parameter out of inference mode and the
next calibration replay raised
``RuntimeError: Inference tensors do not track version counter``.

The unit test reproduces the exact GPU sequence on CPU: an inference Parameter,
the real refined tensor, the real ``.data`` assignment, then a normal-mode
forward. The tiny-model test runs the real ``AWQConfig(gsq=...)`` flow so the
per-linear wiring stays covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, GSQConfig
from gptqmodel.quantization.gsq_scalar import refine_affine_scalar


pytestmark = [pytest.mark.cpu]


_CALIBRATION_TEXTS = [
    "tiny calibration sample one with enough tokens to survive minimum length filtering and exercise the linear path",
    "tiny calibration sample two with repeated words so the awq scale search sees more than one activation row",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2


def _rtn_grid(target, *, group_size=4, bits=4, scale=0.25):
    """A plain nearest-rounding grid, the baseline RTN/AWQ hand to affine GSQ."""

    groups = torch.arange(target.shape[1], dtype=torch.int32) // group_size
    scales = torch.full((target.shape[0], int(groups.max()) + 1), scale, dtype=torch.float16)
    zeros = torch.full_like(scales, 2 ** (bits - 1))
    codes = (
        target.float() / scales[:, groups].float() + zeros[:, groups].float()
    ).round().clamp(0, 2**bits - 1)
    return (scales[:, groups] * (codes - zeros[:, groups])).to(target.dtype), scales, zeros, groups


@pytest.mark.parametrize("learn_scales", [False, True])
def test_refined_weight_keeps_the_inference_mode_of_its_inputs(learn_scales):
    """Both the #4691 skip return and the search return must match the caller."""

    torch.manual_seed(23)
    teacher = torch.randn(4, 8, dtype=torch.float16)

    with torch.inference_mode():
        # The AWQ staging path builds these inside inference mode too.
        weight, scales, zeros, groups = _rtn_grid(teacher)
        # The looper builds the layer Parameter inside inference mode.
        linear = torch.nn.Linear(8, 4, bias=False, dtype=torch.float16)
        assert linear.weight.is_inference()

        fitted = refine_affine_scalar(
            weight, scales, zeros, groups, target=teacher, bits=4,
            config=GSQConfig(enabled=True, steps=2, learn_scales=learn_scales),
        )
        assert fitted.weight.is_inference()

        # This is the assignment that crashed the layer replay on GPU.
        linear.weight.data = fitted.weight

    # This forward is the layer replay that raised
    # "RuntimeError: Inference tensors do not track version counter".
    output = torch.nn.functional.linear(torch.randn(2, 8, dtype=torch.float16), linear.weight)
    assert output.shape == (2, 4)
    # The Parameter must keep the mode it was born in.
    assert linear.weight.is_inference()


def test_normal_inputs_still_return_normal_tensors():
    """The fix must not turn ordinary RTN/unit-test tensors into inference ones."""

    torch.manual_seed(27)
    teacher = torch.randn(4, 8, dtype=torch.float16)
    weight, scales, zeros, groups = _rtn_grid(teacher)

    fitted = refine_affine_scalar(
        weight, scales, zeros, groups, target=teacher, bits=4,
        config=GSQConfig(enabled=True, steps=2),
    )

    assert not fitted.weight.is_inference()
    assert not fitted.scales.is_inference()
    assert not fitted.zeros.is_inference()
    assert not fitted.g_idx.is_inference()


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
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
    return fast_tokenizer


def _build_tiny_qwen2_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
    """Save a tiny dense checkpoint so AWQ has real ``nn.Linear`` projections."""

    config = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    return _build_local_tokenizer(model_dir)


def _build_calibration_dataset(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


@pytest.mark.slow
def test_tiny_awq_gsq_quantizes_and_reloads(tmp_path: Path):
    """The crashing configuration from #4690 must complete end to end."""

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    tokenizer = _build_tiny_qwen2_fixture(model_dir)
    calibration = _build_calibration_dataset(tokenizer)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cpu",
            gsq=GSQConfig(enabled=True, steps=2),
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )

    assert model.quant_log, "AWQ must publish per-module rows"
    assert all(row.get("module") for row in model.quant_log)

    model.save(quantized_dir)
    reloaded = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    quantized = {
        name: module
        for name, module in reloaded.named_modules()
        if isinstance(module, AwqTorchLinear)
    }
    # The checkpoint must carry the AWQ payload for the projections the tiny
    # model has, not silently fall back to a dense reload.
    assert {name.rsplit(".", 1)[-1] for name in quantized} >= {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }
