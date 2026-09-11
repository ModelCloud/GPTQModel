# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.model import pack_module


def _pack_linear(
    bits: int,
    in_features: int,
    out_features: int,
    pack_impl: str,
    desc_act: bool = False,
    pack_dtype: torch.dtype = torch.int32,
    group_size: int = 32,
) -> tuple[TorchLinear, torch.nn.Linear]:
    groups = 1 if group_size == -1 else math.ceil(in_features / group_size)
    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=pack_dtype,
        format=FORMAT.GPTQ_V2,
    )
    g_idx = module.g_idx.flip(0) if desc_act else module.g_idx
    codes = torch.arange(in_features * out_features).reshape(in_features, out_features) % (1 << bits)
    zeros = torch.arange(groups * out_features).reshape(groups, out_features) % (1 << bits)
    scales = (torch.arange(groups * out_features).reshape(groups, out_features) % 3 + 1) / 8
    weight = scales[g_idx] * (codes - zeros[g_idx])
    linear = torch.nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.copy_(weight.T)
    linear.bias.data.copy_(torch.arange(out_features) / 8)
    config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=desc_act,
        format=FORMAT.GPTQ_V2,
        pack_impl=pack_impl,
        pack_dtype=pack_dtype,
        offload_to_disk=False,
    )
    pack_module(
        "linear",
        {"linear": module},
        scales.T.contiguous(),
        zeros.T.contiguous(),
        g_idx,
        {"linear": linear},
        TorchLinear,
        None,
        quantize_config=config,
    )
    return module, linear


def _assert_linear_round_trip(module: TorchLinear, linear: torch.nn.Linear) -> None:
    x = (torch.arange(2 * module.in_features).reshape(2, module.in_features) % 7 - 3).float() / 4
    for training in (True, False):
        module.train(training)
        dequantized = module.dequantize_weight()
        assert dequantized.shape == (module.in_features, module.out_features)
        torch.testing.assert_close(dequantized.T.float(), linear.weight, rtol=0, atol=0)
        torch.testing.assert_close(module(x), linear(x), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize(
    "in_features,out_features", [(64, 32), (64, 20), (64, 19), (63, 32), (63, 19), (1, 1), (640, 24)]
)
@pytest.mark.parametrize("pack_impl", ["cpu", "original"])
def test_pack_module_partial_words(
    bits: int, in_features: int, out_features: int, pack_impl: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    module, linear = _pack_linear(bits, in_features, out_features, pack_impl)
    _assert_linear_round_trip(module, linear)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16])
def test_pack_original_partial_smaller_words(bits: int, pack_dtype: torch.dtype) -> None:
    module, linear = _pack_linear(bits, 63, 19, "original", pack_dtype=pack_dtype)
    _assert_linear_round_trip(module, linear)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [-1, 32])
def test_partial_words_desc_act(bits: int, group_size: int) -> None:
    module, linear = _pack_linear(bits, 63, 19, "original", desc_act=True, group_size=group_size)
    _assert_linear_round_trip(module, linear)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_partial_words_checkpoint_round_trip(bits: int, tmp_path: Path) -> None:
    module, linear = _pack_linear(bits, 63, 19, "original")
    checkpoint = tmp_path / "linear.safetensors"
    save_file(module.state_dict(), str(checkpoint))
    reloaded = TorchLinear(
        bits=bits,
        group_size=32,
        sym=False,
        desc_act=False,
        in_features=63,
        out_features=19,
        bias=True,
        format=FORMAT.GPTQ_V2,
    )
    reloaded.load_state_dict(load_file(str(checkpoint)))
    _assert_linear_round_trip(reloaded, linear)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_stream_dequantize_partial_words(bits: int) -> None:
    module, linear = _pack_linear(bits, 63, 19, "original")
    module._init_wf_unsqueeze_buffers()
    zeros = module._stream_decode_qzeros()
    g_idx = module._stream_g_idx_long()
    buffer = torch.empty((module.in_features, 8), dtype=torch.float32)
    for start in range(0, module.out_features, 8):
        end = min(start + 8, module.out_features)
        width = module._stream_dequantize_tile(buffer, zeros, g_idx, start, end, buffer.dtype)
        assert width == end - start
        torch.testing.assert_close(buffer[:, :width].T, linear.weight[start:end], rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_embedding_partial_words_train_and_eval(bits: int) -> None:
    embedding = torch.nn.Embedding(17, 19)
    embedding.weight.data.copy_(torch.arange(17 * 19).reshape(17, 19) % (1 << bits))
    module = TorchQuantEmbeddings(
        bits=bits,
        group_size=32,
        sym=False,
        desc_act=False,
        in_features=17,
        out_features=19,
        format=FORMAT.GPTQ_V2,
    )
    module.pack_original(embedding, torch.ones((19, 1)), torch.zeros((19, 1)), module.g_idx)
    input_ids = torch.tensor([[0, 8, 16]])
    for training in (True, False):
        module.train(training)
        torch.testing.assert_close(module.dequantize_weight().float(), embedding.weight, rtol=0, atol=0)
        torch.testing.assert_close(module(input_ids).float(), embedding(input_ids), rtol=0, atol=0)
