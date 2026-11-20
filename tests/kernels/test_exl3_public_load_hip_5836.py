# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``GPTQModel.load`` reaches the packed HIP EXL3 GEMV on RDNA4 (#5836).

Before this fix ``ExllamaV3Linear.SUPPORTS_DEVICES`` was ``[DEVICE.CUDA]``, so
``backend=AUTO`` resolved to ``EXL3_TORCH`` on every ROCm host and a gfx1201
checkpoint never built the native ``LinearEXL3`` wrapper. The packed GEMV was
unreachable from the public loader without passing ``EXL3_EXLLAMA_V3``.

These tests quantize a tiny EXL3 checkpoint once, then reload it through the
public API. They pin that AUTO builds ``ExllamaV3Linear``, that decode-sized
forwards call ``bc_linear_exl3_run_hip_gemv`` while a wider batch stays on the
fused GEMM, and that both paths match an independent FP64 decode of the
packed trellis. Everything here is gated on RDNA4 plus a usable extension.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils.arch import rocm_arch


def _skip_reason() -> str | None:
    if torch.version.hip is None:
        return "public EXL3 HIP load requires a ROCm (HIP) build"
    if not torch.cuda.is_available():
        return "public EXL3 HIP load requires a ROCm device"
    arch = rocm_arch(torch.cuda.current_device())
    if arch is None or not arch.startswith("gfx120"):
        return f"public EXL3 HIP load requires RDNA4 (gfx120x); found {arch or 'none'}"
    from gptqmodel.exllamav3.ext import exl3_native_hip_gemm_usable

    if not exl3_native_hip_gemm_usable(torch.device("cuda:0")):
        return "EXL3 HIP extension is neither loadable nor buildable on this host"
    return None


_SKIP_REASON = _skip_reason()

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]

PROJECTION = "model.layers.0.mlp.down_proj"


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    """Quantize the 256->512 ``down_proj`` to EXL3 3.0 bpw ``mul1``.

    The explicit ``EXL3_EXLLAMA_V3`` here only makes the fixture independent of
    the AUTO branch under test; the reload tests below use AUTO.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import EXL3Config
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    root = tmp_path_factory.mktemp("exl3-public-load-5836")
    source, packed = root / "source", root / "quantized"
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=128,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5836)
        LlamaForCausalLM(config).eval().half().save_pretrained(source)
    build_wordlevel_tokenizer(128).save_pretrained(source)

    tokens = torch.randint(
        4, 128, (2, 64), generator=torch.Generator().manual_seed(5836)
    ).tolist()
    dataset = [{"input_ids": row, "attention_mask": [1] * len(row)} for row in tokens]
    model = GPTQModel.load(
        str(source),
        quantize_config=EXL3Config(bits=3.0, codebook="mul1", offload_to_disk=False),
        device="cuda:0",
        dtype=torch.float16,
        backend=BACKEND.EXL3_EXLLAMA_V3,
    )
    model.quantize(
        CalibrationConfig(dataset=dataset, batch_size=1, min_length=1),
        backend=BACKEND.EXL3_EXLLAMA_V3,
    )
    model.save(packed)
    del model
    return packed


@pytest.fixture(scope="module")
def auto_model(quantized):
    from gptqmodel import GPTQModel

    return GPTQModel.load(str(quantized), device="cuda:0", dtype=torch.float16)


def _projection(model):
    return model.model.get_submodule(PROJECTION)


def _decode(module):
    """Read the packed tensors back for the independent FP64 oracle."""
    packed = {
        "trellis": module.trellis.detach().cpu(),
        "suh": module.suh.detach().cpu(),
        "svh": module.svh.detach().cpu(),
    }
    if module.mul1 is not None:
        packed["mul1"] = module.mul1.detach().cpu()
    if module.bias is not None:
        packed["bias"] = module.bias.detach().cpu()
    return packed


def _expected(module, x):
    from tests.test_exllamav3_torch_padding import _oracle_weight

    packed = _decode(module)
    weight = _oracle_weight(packed, device=x.device).double()
    expected = x.double() @ weight
    if module.bias is not None:
        expected = expected + module.bias.double()
    return expected


def _input(module, rows: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, module.in_features, generator=generator) * 0.1).half().cuda()


def test_auto_backend_reaches_the_native_projection_on_gfx1201(auto_model):
    from gptqmodel.models.loader import _resolve_exl3_auto_backend
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.utils.backend import BACKEND

    assert _resolve_exl3_auto_backend(torch.device("cuda:0")) == BACKEND.EXL3_EXLLAMA_V3
    projection = _projection(auto_model)
    assert type(projection) is ExllamaV3Linear
    assert not isinstance(projection, ExllamaV3TorchLinear)
    # The packed payload stays resident; the native wrapper is built lazily.
    assert projection.trellis.dtype == torch.int16
    assert projection.trellis.shape == (
        projection.in_features // 16, projection.out_features // 16, 48
    )
    assert projection._inner is None
    assert projection._torch_inner is None


def test_auto_load_without_a_device_reaches_native(quantized):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    # The default placement must resolve the same native kernel as an explicit
    # ``device="cuda:0"``; the loader expands it to a one-ordinal sequence.
    model = GPTQModel.load(str(quantized), dtype=torch.float16)
    assert type(_projection(model)) is ExllamaV3Linear


def test_auto_load_calls_the_packed_hip_gemv_for_decode_batches(auto_model, monkeypatch):
    from gptqmodel.exllamav3 import ext as ext_module

    real_gemv = ext_module.exllamav3_ext.bc_linear_exl3_run_hip_gemv
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real_gemv(*args, **kwargs)

    monkeypatch.setattr(ext_module.exllamav3_ext, "bc_linear_exl3_run_hip_gemv", spy)
    projection = _projection(auto_model)
    with torch.inference_mode():
        for rows in (1, 8):
            out = projection(_input(projection, rows, seed=rows))
            assert out.shape == (rows, projection.out_features)
            assert out.dtype is torch.float16
        assert len(calls) == 2
        # Wider batches use the fused RDNA4 GEMM, not the decode GEMV.
        out = projection(_input(projection, 9, seed=9))
        assert out.shape == (9, projection.out_features)
        assert len(calls) == 2

    # Native inference never materializes the portable reconstruction.
    assert projection._inner is not None
    assert projection._torch_inner is None
    assert projection.trellis.dtype == torch.int16


@pytest.mark.parametrize("rows", (1, 8, 9))
@torch.inference_mode()
def test_auto_load_matches_the_fp64_oracle(auto_model, rows):
    projection = _projection(auto_model)
    x = _input(projection, rows, seed=100 + rows)
    out = projection(x).float()
    expected = _expected(projection, x).float()
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


def test_explicit_exllama_v3_loads_native_on_gfx1201(quantized):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.utils.backend import BACKEND

    model = GPTQModel.load(
        str(quantized),
        device="cuda:0",
        dtype=torch.float16,
        backend=BACKEND.EXL3_EXLLAMA_V3,
    )
    projection = _projection(model)
    assert type(projection) is ExllamaV3Linear
    with torch.inference_mode():
        x = _input(projection, 1, seed=7)
        out = projection(x)
    assert out.shape == (1, projection.out_features)
    torch.testing.assert_close(out.float(), _expected(projection, x).float(),
                               rtol=2e-3, atol=2e-3)
