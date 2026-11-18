# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXL3 torch constants must not inherit the first caller's inference mode.

``ExllamaV3TorchLinear`` reads its Hadamard/permutation/codebook constants from
process-global ``functools.lru_cache`` factories. Before #5213 those factories
built the tensors in the caller's context, so the first EXL3 torch forward in a
process running under ``torch.inference_mode()`` (``generate()``, an eval loop,
a server warmup) cached *inference* tensors. Every later grad-mode forward of
any model then raised ``RuntimeError: Inference tensors cannot be saved for
backward`` in ``_apply_hadamard_right`` until the process restarted.

The regression is the transition, not construction: warm the cache with one
module under inference mode, then forward a **second, fresh** module in
ordinary grad mode. A clone taken inside inference mode is still an inference
tensor, so the test also pins the cached entries' kind directly.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules import exllamav3_torch as kernel
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear


_CACHE_FACTORIES = (
    kernel._tensor_core_perm,
    kernel._tensor_core_perm_i,
    kernel._hadamard_128,
    kernel._hadamard_128_as,
    kernel._codebook_lut,
)


def _clear_constant_caches() -> None:
    for factory in _CACHE_FACTORIES:
        factory.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_constant_cache():
    """Start each case with empty caches, like a fresh process.

    The defect is process-wide, so a leftover entry from another test would
    hide the inference-mode build (or the fix) behind an earlier caller.
    """

    _clear_constant_caches()
    yield
    _clear_constant_caches()


def _packed(device: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(5213)
    packed = {
        "trellis": torch.randint(
            -32768, 32767, (8, 8, 64), dtype=torch.int16, generator=generator
        ).to(device),
        "suh": (torch.rand(128, generator=generator) * 0.1).half().to(device),
        "svh": (torch.rand(128, generator=generator) * 0.1).half().to(device),
        "mcg": torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32).to(device),
        "bias": (torch.randn(128, generator=generator) * 0.01).half().to(device),
    }
    return packed


def _module(device: str) -> ExllamaV3TorchLinear:
    return ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=128, name="proj", tensors=_packed(device)
    )


def _device_args(device: str) -> tuple[str, int | None]:
    parsed = torch.device(device)
    return parsed.type, parsed.index


def _devices() -> list:
    cases = [pytest.param("cpu", id="cpu", marks=pytest.mark.cpu)]
    if torch.cuda.is_available():
        cases.append(pytest.param("cuda", id="cuda", marks=pytest.mark.cuda))
    return cases


@pytest.mark.parametrize("device", _devices())
def test_constants_built_under_inference_mode_are_normal_tensors(device):
    device_type, device_index = _device_args(device)
    with torch.inference_mode():
        had = kernel._hadamard_128(device_type, device_index)
        had_as = kernel._hadamard_128_as(device_type, device_index, torch.float16)
        perm = kernel._tensor_core_perm(device_type, device_index)
        perm_i = kernel._tensor_core_perm_i(device_type, device_index)
        lut = kernel._codebook_lut("mcg", device_type, device_index)

    for name, tensor in (
        ("hadamard", had),
        ("hadamard_fp16", had_as),
        ("perm", perm),
        ("perm_i", perm_i),
        ("codebook_lut", lut),
    ):
        assert not tensor.is_inference(), f"cached {name} is an inference tensor"


@pytest.mark.parametrize("device", _devices())
def test_inference_warmup_does_not_poison_a_fresh_grad_module(device):
    """Model A under ``inference_mode`` must not break a new model B's grad."""

    module_a = _module(device)
    x_a = torch.randn(2, 128, device=device)
    with torch.inference_mode():
        out_a = module_a(x_a)
    assert out_a.shape == (2, 128)
    del module_a

    module_b = _module(device)  # A fresh instance, not the warmed one.
    x_b = torch.randn(2, 128, device=device, requires_grad=True)
    out_b = module_b(x_b)
    out_b.float().sum().backward()

    assert out_b.shape == (2, 128)
    assert x_b.grad is not None and torch.isfinite(x_b.grad).all()
    device_type, device_index = _device_args(device)
    assert not kernel._hadamard_128(device_type, device_index).is_inference()


@pytest.mark.parametrize("device", _devices())
def test_grad_first_then_inference_forward_still_works(device):
    """The other order was already fine; keep it as the negative control."""

    module_a = _module(device)
    x_a = torch.randn(2, 128, device=device, requires_grad=True)
    module_a(x_a).float().sum().backward()
    assert x_a.grad is not None

    module_b = _module(device)
    x_b = torch.randn(2, 128, device=device)
    with torch.inference_mode():
        out_b = module_b(x_b)
    assert torch.isfinite(out_b).all()


def _exl3_gpu_reason() -> str | None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "EXL3 quantization needs an NVIDIA CUDA device"
    return None


@pytest.fixture(scope="module")
def tiny_exl3_checkpoint(tmp_path_factory):
    reason = _exl3_gpu_reason()
    if reason is not None:
        pytest.skip(reason)

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel import CalibrationConfig, GPTQModel
    from gptqmodel.quantization import AutoDtypeCast
    from gptqmodel.quantization.config import EXL3Config
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    vocab, hidden, inter, layers = 128, 128, 256, 2
    native = tmp_path_factory.mktemp("exl3-constant-cache-5213") / "native"
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(5213)
    LlamaForCausalLM(config).save_pretrained(native)
    build_wordlevel_tokenizer().save_pretrained(native)

    generator = torch.Generator().manual_seed(5213)
    calibration = [
        {
            "input_ids": torch.randint(5, vocab - 4, (1, 32), generator=generator),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(8)
    ]
    model = GPTQModel.load(
        str(native),
        quantize_config=EXL3Config(
            bits=4,
            device="cuda:0",
            offload_to_disk=False,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        dtype=torch.float16,
    )
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=2, min_length=1))
    checkpoint = native.parent / "exl3"
    model.save(str(checkpoint))
    del model
    torch.cuda.empty_cache()
    return str(checkpoint)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.cuda
def test_second_loaded_model_forwards_in_grad_mode_after_inference_warmup(
    tiny_exl3_checkpoint,
):
    """The issue's repro at the public API: new model B after model A's generate."""

    from gptqmodel import BACKEND, GPTQModel

    ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]], device="cuda:0")
    model_a = GPTQModel.load(
        tiny_exl3_checkpoint,
        backend=BACKEND.EXL3_TORCH,
        dtype=torch.float16,
        device="cuda:0",
    )
    with torch.inference_mode():
        loss_a = model_a.model(ids, labels=ids).loss
    assert torch.isfinite(loss_a)
    del model_a

    model_b = GPTQModel.load(
        tiny_exl3_checkpoint,
        backend=BACKEND.EXL3_TORCH,
        dtype=torch.float16,
        device="cuda:0",
    )
    loss_b = model_b.model(ids, labels=ids).loss
    assert torch.isfinite(loss_b)
    assert not kernel._hadamard_128("cuda", 0).is_inference()
