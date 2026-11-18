# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4554 and #4736.

AWQ ``FORMAT.GEMV`` (#4554) and ``FORMAT.GEMV_FAST`` / ``FORMAT.LLM_AWQ``
(#4736) had no in-session kernel for the CPU the quantize loop leaves the
packed layers on: the native classes are CUDA-only and the Triton reader needs
an accelerator. ``quantize()`` therefore raised after the whole calibration and
the documented quantize -> save flow lost the result. These tests pin the CPU
readers and the end-to-end flow.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import rehome_quantized_kernel, select_in_session_quant_linear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear, AwqGEMVTorchLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import (
    AwqGEMVFastLinear,
    AwqGEMVFastTorchLinear,
    LLMAwqLinear,
    LLMAwqTorchLinear,
)
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AWQConfig
from tests.accelerator_gates import requires_nvidia_cuda


HIDDEN_SIZE = 64
VOCAB_SIZE = 64
# GEMV_FAST packs four output rows per word and its kernels require
# out_features % 128 == 0, so its tiny model needs a wider hidden size.
GEMV_FAST_HIDDEN_SIZE = 128

# (format, native CUDA-only class, portable CPU reader)
GEMV_FAST_FAMILY = [
    pytest.param(FORMAT.GEMV_FAST, AwqGEMVFastLinear, AwqGEMVFastTorchLinear, id="gemv_fast"),
    pytest.param(FORMAT.LLM_AWQ, LLMAwqLinear, LLMAwqTorchLinear, id="llm_awq"),
]


def _write_tiny_llama(directory: Path, hidden_size: int = HIDDEN_SIZE) -> None:
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).half().eval().save_pretrained(directory)
    tokenizer = Tokenizer(models.WordLevel(
        vocab={"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(directory)


def _calibration():
    return [
        {
            "input_ids": torch.randint(4, VOCAB_SIZE, (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]


def test_awq_gemv_cpu_insession_kernel_is_the_documented_fallback():
    qcfg = AWQConfig(bits=4, format="gemv", group_size=128, offload_to_disk=False)

    selected = select_in_session_quant_linear(qcfg, torch.device("cpu"))

    assert selected is AwqGEMVTorchLinear
    assert DEVICE.CPU in selected.SUPPORTS_DEVICES
    # Explicit-only: accelerator AUTO order must not change (native CUDA GEMV
    # stays first there, and ROCm keeps its actionable refusal).
    assert selected.SUPPORTS_FORMATS[FORMAT.GEMV] == 0


def test_rehome_switches_native_gemv_module_to_cpu_reader():
    qcfg = AWQConfig(bits=4, format="gemv", group_size=128, offload_to_disk=False)
    module = AwqGEMVLinear(
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

    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = module

    fake_model = SimpleNamespace(
        model=_Tree(),
        quantize_config=qcfg,
        qlinear_kernel=AwqGEMVLinear,
    )

    rehome_quantized_kernel(fake_model)

    assert type(module) is AwqGEMVTorchLinear
    assert fake_model.qlinear_kernel is AwqGEMVTorchLinear
    assert module.qweight.device.type == "cpu"


def test_awq_gemv_cpu_reader_matches_the_gemm_reader_numerics():
    # Both layouts encode the same codes/scales, so the portable GEMV reader
    # must land on the portable GEMM reader's result (the GEMV path keeps the
    # reduction wide, so the residual is one FP16 rounding).
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(4554)
    in_features = out_features = 256
    group_size = 128
    linear = nn.Linear(in_features, out_features, bias=True).half()
    scales = torch.full((out_features, in_features // group_size), 0.01, dtype=torch.float16)
    zeros = torch.full((out_features, in_features // group_size), 8.0, dtype=torch.float16)
    kwargs = {
        "bits": 4,
        "group_size": group_size,
        "sym": False,
        "desc_act": False,
        "in_features": in_features,
        "out_features": out_features,
        "bias": True,
        "pack_dtype": torch.int32,
        "register_buffers": True,
    }
    gemv = AwqGEMVTorchLinear(**kwargs)
    gemv.pack(linear, scales, zeros)
    gemm = AwqTorchLinear(**kwargs)
    gemm.pack(linear, scales, zeros)

    inputs = torch.randn(3, in_features).half()
    with torch.inference_mode():
        gemv_out = gemv(inputs)
        gemm_out = gemm(inputs)

    assert gemv_out.dtype == inputs.dtype
    torch.testing.assert_close(gemv_out, gemm_out, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not requires_nvidia_cuda(),
    reason="the AWQ GEMV pack step needs the native CUDA AWQ extension",
)
def test_awq_gemv_quantize_save_and_reload_on_cpu(tmp_path):
    source = tmp_path / "source"
    _write_tiny_llama(source)
    out = tmp_path / "out"
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        format=FORMAT.GEMV,
        device="cpu",
        offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), quantize_config=qcfg, dtype=torch.float16)

    # The CPU quantize path used to raise before calibration because no packing
    # kernel advertised CPU support.
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))

    quant_linear = [m for m in model.model.modules() if isinstance(m, AwqGEMVLinear)]
    assert quant_linear
    assert all(isinstance(m, AwqGEMVTorchLinear) for m in quant_linear)
    assert model.qlinear_kernel is AwqGEMVTorchLinear

    ids = torch.tensor([[1, 4, 4, 4]])
    with torch.inference_mode():
        logits = model(ids).logits
    assert logits.shape == (1, 4, VOCAB_SIZE)
    assert torch.isfinite(logits).all()

    model.save(str(out))
    saved = json.loads((out / "config.json").read_text())
    assert saved["quantization_config"]["format"] == "gemv"

    # The checkpoint keeps the requested GEMV packing: qweight is
    # [out_features, in_features / 8], not the GEMM [in_features, out/8] layout.
    tensors = load_file(str(out / "model.safetensors"))
    assert tensors["model.layers.0.self_attn.q_proj.qweight"].shape == (
        HIDDEN_SIZE,
        HIDDEN_SIZE // 8,
    )

    reloaded = GPTQModel.load(str(out), dtype=torch.float16, device="cpu")
    assert reloaded.qlinear_kernel is AwqGEMVTorchLinear
    with torch.inference_mode():
        relogits = reloaded(ids).logits
    assert torch.equal(relogits, logits)


def _gemv_fast_oracle(linear: nn.Linear, scales: torch.Tensor, zeros: torch.Tensor, group_size: int, x):
    """Independent fake-quant reference for the GEMV_FAST pack contract.

    Codes are ``round((W + z * s) / s)`` clamped to 4 bits; the checkpoint stores
    the FP16 scale and the FP16-rounded offset ``-s * z``, so the dequantized
    weight is ``code * s + fp16(-s * z)``, accumulated in FP32.
    """

    group_index = torch.arange(linear.in_features) // group_size
    s = scales.float()[:, group_index]
    z = zeros.float()[:, group_index]
    codes = torch.round((linear.weight.float() + z * s) / s).clamp(0, 15)
    offset = (-s * z).to(torch.float16).float()
    weight = codes * s + offset
    out = x.float() @ weight.t()
    if linear.bias is not None:
        out = out + linear.bias.float()
    return out.to(x.dtype)


def _packed_gemv_fast(native_cls, in_features=256, out_features=128, group_size=128):
    torch.manual_seed(4736)
    groups = in_features // group_size
    linear = nn.Linear(in_features, out_features, bias=True).half()
    scales = (torch.rand(out_features, groups) * 0.02 + 0.005).half()
    zeros = torch.randint(0, 16, (out_features, groups)).half()
    module = native_cls(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int16,
        register_buffers=True,
    )
    module.pack(linear, scales, zeros)
    return module, linear, scales, zeros


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_awq_gemv_fast_portable_reader_is_the_documented_fallback(fmt, native_cls, torch_cls):
    qcfg = AWQConfig(bits=4, format=fmt, group_size=128, pack_dtype=torch.int16, offload_to_disk=False)

    selected = select_in_session_quant_linear(qcfg, torch.device("cpu"))

    assert selected is torch_cls
    assert DEVICE.CPU in selected.SUPPORTS_DEVICES
    # Explicit-only: AUTO order must not change.
    assert selected.SUPPORTS_FORMATS == {fmt: 0}
    assert DEVICE.CPU not in native_cls.SUPPORTS_DEVICES


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_awq_gemv_fast_portable_reader_serves_a_rocm_target(fmt, native_cls, torch_cls):
    # #5109: the native class is CUDA-only and no Triton kernel reads this
    # layout, so AUTO on ROCm used to raise "No valid quant linear" and the only
    # named portable backend refused the device. The Torch reader declares ROCm
    # and is reached through the per-family portable fallback.
    qcfg = AWQConfig(bits=4, format=fmt, group_size=128, pack_dtype=torch.int16, offload_to_disk=False)

    selected = select_in_session_quant_linear(qcfg, DEVICE.ROCM)

    assert selected is torch_cls
    assert DEVICE.ROCM in selected.SUPPORTS_DEVICES
    assert selected.SUPPORTS_FORMATS == {fmt: 0}
    assert DEVICE.ROCM not in native_cls.SUPPORTS_DEVICES


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_awq_gemv_fast_portable_reader_keeps_packed_storage_across_forwards(fmt, native_cls, torch_cls):
    # The reader is now reachable on an accelerator, where the packed buffers can
    # still sit on CPU. A forward must move only the packed storage and must not
    # retain a dense weight: no FP32 buffer may outlive the call.
    module, linear, scales, zeros = _packed_gemv_fast(native_cls)
    packed = {name: buf.clone() for name, buf in module.named_buffers()}
    portable = torch_cls(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
        pack_dtype=torch.int16,
        register_buffers=True,
    )
    portable.load_state_dict(
        {"qweight": packed["qweight"], "scales": packed["scales"], portable.zeros_name: packed[portable.zeros_name],
         "bias": packed["bias"]},
        strict=True,
    )

    inputs = torch.randn(2, 3, linear.in_features).half()
    with torch.inference_mode():
        out = portable(inputs)

    torch.testing.assert_close(out, _gemv_fast_oracle(linear, scales, zeros, 128, inputs), rtol=1e-3, atol=1e-2)
    assert out.dtype == inputs.dtype
    # Repeated forwards keep the same packed storage: no dense cache, no
    # re-packing, and no silent dtype narrowing of the stored buffers.
    stored = dict(portable.named_buffers())
    for name, buf in packed.items():
        assert stored[name].dtype == buf.dtype, name
        assert torch.equal(stored[name], buf), name
    # A dense [out_features, in_features] weight would have this many elements;
    # no buffer may reach it, regardless of dtype.
    dense_elements = portable.in_features * portable.out_features
    assert all(buf.numel() < dense_elements for buf in stored.values())


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_rehome_switches_native_gemv_fast_module_to_cpu_reader(fmt, native_cls, torch_cls):
    qcfg = AWQConfig(bits=4, format=fmt, group_size=128, pack_dtype=torch.int16, offload_to_disk=False)
    module, linear, scales, zeros = _packed_gemv_fast(native_cls)
    packed = {name: buf.clone() for name, buf in module.named_buffers()}

    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = module

    fake_model = SimpleNamespace(model=_Tree(), quantize_config=qcfg, qlinear_kernel=native_cls)

    # Used to raise RuntimeError (no CPU kernel) and lose the calibration.
    rehome_quantized_kernel(fake_model)

    assert type(module) is torch_cls
    assert fake_model.qlinear_kernel is torch_cls
    # The switch only reinterprets the packed state; the checkpoint layout is untouched.
    for name, buf in module.named_buffers():
        if name in packed:
            assert torch.equal(buf, packed[name]), name

    inputs = torch.randn(2, 3, linear.in_features).half()
    with torch.inference_mode():
        out = module(inputs)
    assert out.dtype == inputs.dtype
    assert out.shape == (2, 3, linear.out_features)
    torch.testing.assert_close(
        out, _gemv_fast_oracle(linear, scales, zeros, 128, inputs), rtol=1e-3, atol=1e-2
    )
    # No dense weight copy outlives the forward.
    assert all(buf.dtype != torch.float32 for buf in module.buffers())


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_awq_gemv_fast_portable_reader_moves_packed_storage_per_call(fmt, native_cls, torch_cls):
    # The reader is now selectable on an accelerator, where the quantize loop can
    # leave the packed buffers on CPU. One forward must then use device-local
    # packed words without rewriting the module's stored checkpoint buffers, and
    # a repeat call must produce the same result.
    if not torch.cuda.is_available():
        pytest.skip("a runtime accelerator device is required for the cross-device path")
    device = torch.device("cuda", 0)
    # AUTO on a CUDA target keeps the native kernel (checked above); this test
    # covers the portable reader's own accelerator forward.
    module, linear, scales, zeros = _packed_gemv_fast(native_cls)
    portable = torch_cls(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
        pack_dtype=torch.int16,
        register_buffers=True,
    )
    portable.load_state_dict(
        {
            "qweight": module.qweight,
            "scales": module.scales,
            portable.zeros_name: getattr(module, module.zeros_name),
            "bias": module.bias,
        },
        strict=True,
    )
    packed = {name: buf.clone() for name, buf in portable.named_buffers()}

    inputs = torch.randn(2, 3, linear.in_features, dtype=torch.float16, device=device)
    with torch.inference_mode():
        first = portable(inputs)
        second = portable(inputs)

    assert first.device == device
    assert torch.equal(first, second)
    torch.testing.assert_close(
        first.cpu(), _gemv_fast_oracle(linear, scales, zeros, 128, inputs.cpu()), rtol=1e-3, atol=1e-2
    )
    # The checkpoint's packed buffers stay where the checkpoint put them, in
    # their stored dtype: the move is per-call and never a module-state rewrite.
    for name, buf in portable.named_buffers():
        assert buf.device.type == "cpu", name
        assert buf.dtype == packed[name].dtype, name
        assert torch.equal(buf, packed[name]), name


@pytest.mark.parametrize("fmt, native_cls, torch_cls", GEMV_FAST_FAMILY)
def test_awq_gemv_fast_quantize_save_and_reload_on_cpu(tmp_path, fmt, native_cls, torch_cls):
    hidden = GEMV_FAST_HIDDEN_SIZE
    source = tmp_path / "source"
    _write_tiny_llama(source, hidden_size=hidden)
    out = tmp_path / "out"
    qcfg = QuantizeConfig(bits=4, group_size=128, format=fmt, device="cpu", offload_to_disk=False)
    model = GPTQModel.load(str(source), quantize_config=qcfg, dtype=torch.float16)

    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))

    quant_linear = [m for m in model.model.modules() if isinstance(m, native_cls)]
    assert quant_linear
    assert all(type(m) is torch_cls for m in quant_linear)
    assert model.qlinear_kernel is torch_cls

    ids = torch.tensor([[1, 4, 4, 4]])
    with torch.inference_mode():
        logits = model(ids).logits
    assert logits.shape == (1, 4, VOCAB_SIZE)
    assert torch.isfinite(logits).all()

    model.save(str(out))
    saved = json.loads((out / "config.json").read_text())
    assert saved["quantization_config"]["format"] == fmt.value

    # The checkpoint keeps the GEMV_FAST interleaved packing: INT16 qweight of
    # shape [out_features / 4, in_features].
    tensors = load_file(str(out / "model.safetensors"))
    qweight = tensors["model.layers.0.self_attn.q_proj.qweight"]
    assert qweight.dtype == torch.int16
    assert qweight.shape == (hidden // 4, hidden)

    reloaded = GPTQModel.load(str(out), dtype=torch.float16, device="cpu")
    assert reloaded.qlinear_kernel is torch_cls
    with torch.inference_mode():
        relogits = reloaded(ids).logits
    assert torch.equal(relogits, logits)
