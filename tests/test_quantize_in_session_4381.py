# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4381.

Weight-only quantization builds a meta shell, so its base modules are already
on meta before the offload capture and nothing restores them afterwards. AWQ,
QQQ and GGUF also keep the calibration-time accelerator kernel after
``post_quantize`` moves the packed layers to CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

import gptqmodel.models.base as base_module
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import rehome_quantized_kernel, select_in_session_quant_linear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, GGUFConfig, QQQConfig, RTNConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import clear_validation_cache
from gptqmodel.utils.offload import is_meta_module


def _tiny_tied_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config).eval()


def _meta_shell() -> LlamaForCausalLM:
    """Build the state a weight-only meta-shell load leaves before the loop."""

    model = _tiny_tied_model()
    for module in (model.model.embed_tokens, model.model.norm, model.model.rotary_emb):
        module.to("meta")
    # ``tie_weights`` mirrors the meta shell: the head follows the meta
    # embedding parameter until the base module is materialized.
    model.tie_weights()
    assert is_meta_module(model.model.embed_tokens)
    return model


def _copy_module_state(source: nn.Module, target: nn.Module) -> None:
    """Replace meta params/buffers on ``target`` with ``source`` storage."""

    with torch.no_grad():
        for name, parameter in source.named_parameters(recurse=False):
            target._parameters[name] = nn.Parameter(
                parameter.detach().clone(), requires_grad=parameter.requires_grad
            )
        for name, buffer in source.named_buffers(recurse=False):
            target._buffers[name] = buffer.detach().clone()


class _StubWeightOnlyQModel:
    def __init__(self, model: nn.Module, source: nn.Module, *, materialize: bool = True):
        self.model = model
        self.source = source
        self.materialize = materialize
        self.quantize_config = RTNConfig(bits=4, group_size=32, offload_to_disk=False, device="cpu")

    def get_base_modules(self, model):
        del model
        return ["model.embed_tokens", "model.norm", "model.rotary_emb"]

    def shell_module_materialize(self, *, target_submodule, device, module_path=None):
        del device
        if not self.materialize:
            return target_submodule
        _copy_module_state(self.source.get_submodule(module_path), target_submodule)
        self.model.tie_weights()
        return target_submodule


def _stub_looper(qmodel) -> WeightOnlyLooper:
    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = qmodel
    looper._offloaded_base_modules = []
    looper._loop_impl = lambda **_kwargs: "finished"
    return looper


def test_weight_only_loop_materializes_meta_base_modules():
    source = _tiny_tied_model()
    model = _meta_shell()
    looper = _stub_looper(_StubWeightOnlyQModel(model, source))

    assert looper.loop() == "finished"

    for name in ("model.embed_tokens", "model.norm", "model.rotary_emb"):
        assert not is_meta_module(model.get_submodule(name)), name
    assert model.device.type == "cpu"
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight

    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1, 3, 5]])).logits
    assert logits.shape == (1, 3, 32)
    assert torch.isfinite(logits).all()


def test_weight_only_loop_raises_when_meta_base_modules_cannot_materialize():
    source = _tiny_tied_model()
    model = _meta_shell()
    looper = _stub_looper(_StubWeightOnlyQModel(model, source, materialize=False))

    with pytest.raises(RuntimeError, match="reload the saved checkpoint"):
        looper.loop()


@pytest.mark.parametrize(
    ("qcfg", "expected"),
    [
        (AWQConfig(bits=4, format="gemm", group_size=128, offload_to_disk=False), AwqTorchLinear),
        (QQQConfig(bits=4, group_size=128, offload_to_disk=False), QQQTorchLinear),
        (GGUFConfig(bits="q4_k_m", offload_to_disk=False), GGUFTorchLinear),
    ],
)
def test_post_quantize_selector_picks_cpu_capable_kernel(monkeypatch, qcfg, expected):
    # The calibration-time choice on a CUDA host would be one of these. Mock
    # them unavailable so the assertion is deterministic on every runner.
    monkeypatch.setattr(
        AwqGEMMLinear,
        "validate_once",
        classmethod(lambda cls: (False, NotImplementedError("CUDA kernel unavailable"))),
    )
    monkeypatch.setattr(
        QQQLinear,
        "validate_once",
        classmethod(lambda cls: (False, ImportError("CUDA kernel unavailable"))),
    )
    monkeypatch.setattr(
        GGUFTritonKernel,
        "validate_once",
        classmethod(lambda cls: (False, NotImplementedError("CUDA kernel unavailable"))),
    )
    clear_validation_cache()

    selected = select_in_session_quant_linear(qcfg, torch.device("cpu"))

    assert selected is expected
    assert DEVICE.CPU in selected.SUPPORTS_DEVICES


class _CudaOnlyAwq(AwqTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMM]
    SUPPORTS_DEVICES = [DEVICE.CUDA]


class _ModuleTree(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.proj = module


def test_rehome_quantized_kernel_switches_cuda_only_module(monkeypatch):
    module = AwqTorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.__class__ = _CudaOnlyAwq
    fake_model = SimpleNamespace(
        model=_ModuleTree(module),
        quantize_config=AWQConfig(bits=4, format="gemm", group_size=128, offload_to_disk=False),
        qlinear_kernel=_CudaOnlyAwq,
    )
    monkeypatch.setattr(base_module, "select_in_session_quant_linear", lambda *_args: AwqTorchLinear)

    rehome_quantized_kernel(fake_model)

    assert type(module) is AwqTorchLinear
    assert module.backend == BACKEND.AWQ_TORCH
    assert fake_model.qlinear_kernel is AwqTorchLinear
    assert module.qweight.device.type == "cpu"


def test_rehome_quantized_kernel_raises_when_no_portable_kernel(monkeypatch):
    module = AwqTorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.__class__ = _CudaOnlyAwq
    fake_model = SimpleNamespace(
        model=_ModuleTree(module),
        quantize_config=AWQConfig(bits=4, format="gemm", group_size=128, offload_to_disk=False),
        qlinear_kernel=_CudaOnlyAwq,
    )

    def _no_portable(*_args):
        raise ValueError("No valid quant linear")

    monkeypatch.setattr(base_module, "select_in_session_quant_linear", _no_portable)

    with pytest.raises(RuntimeError, match="reload the saved checkpoint"):
        rehome_quantized_kernel(fake_model)
