# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Check GPTQ capture sees rounded inputs without a second input rounding."""

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.w4a_llama_replay import install_w4a_llama_replay
from gptqmodel.quantization.config import QuantizeConfig


def _independent_round(x: torch.Tensor, mode: str) -> torch.Tensor:
    blocks = x.float()
    maxima = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maxima > 0, maxima / 448.0, torch.ones_like(maxima))
    return ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)


@pytest.mark.parametrize("mode", ["w4afp8"])
@pytest.mark.parametrize("skip_first", [False, True])
@pytest.mark.parametrize("version", [2, 3])
def test_replay_hessian_input_matches_encoded_stream(mode, skip_first, version):
    torch.manual_seed(703)
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "version": version, "mode": mode,
        },
        offload_to_disk=False,
        dynamic={"-:^model.layers\\.0\\.": {}} if skip_first else None,
    )
    target = model.model.layers[1 if skip_first else 0].self_attn.q_proj
    raw = []
    rounded = []
    raw_output = []
    rounded_output = []
    target.register_forward_pre_hook(lambda _module, args: raw.append(args[0].clone()))
    target.register_forward_hook(lambda _module, _args, out: raw_output.append(out.clone()))
    install_w4a_llama_replay(model, qcfg)
    target.register_forward_pre_hook(lambda _module, args: rounded.append(args[0].clone()))
    target.register_forward_hook(lambda _module, _args, out: rounded_output.append(out.clone()))
    ids = torch.tensor([[1, 7, 3]], dtype=torch.long)
    with torch.inference_mode():
        result = model(input_ids=ids, use_cache=False)
    assert torch.isfinite(result.logits).all()
    assert len(raw) == len(rounded) == 1
    torch.testing.assert_close(rounded[0], _independent_round(raw[0], mode), rtol=0, atol=0)
    assert len(raw_output) == len(rounded_output) == 1
    expected_output = (
        _independent_round(raw_output[0], mode) if version == 2 else raw_output[0]
    )
    torch.testing.assert_close(rounded_output[0], expected_output, rtol=0, atol=0)
    assert getattr(model.model.layers[0], "_w4a_replay_round_input", None) is (None if skip_first else True)
    assert getattr(model.model.layers[1], "_w4a_replay_round_input", None) is (True if skip_first else False)


@pytest.mark.parametrize("mode", ["w4afp8"])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("version", [2, 3])
def test_hooked_linear_preserves_w4a_replay_policy(monkeypatch, mode, rotated, version):
    """The GPTQ stage replacement must retain the versioned replay policy."""
    torch.manual_seed(704)
    dense = torch.nn.Linear(128, 128, bias=True, dtype=torch.float32).eval()
    dense._w4a_stream_replay_mode = mode
    dense._w4a_stream_replay_recipe = None
    dense._w4a_stream_replay_version = version
    dense._w4a_stream_replay_pre_hook = True
    dense.online_full_had = rotated
    dense.online_partial_had = False
    dense.had_dim = -1
    dense.had_K = None
    dense.K = 1
    hooked = HookedLinear.from_linear(dense).eval()

    if rotated:
        # A deterministic orthogonal permutation keeps this check independent
        # of the optional CUDA Hadamard extension.
        monkeypatch.setattr(
            "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
            lambda value, **_kwargs: value.flip(-1),
        )

    seen = []
    hooked.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    x = torch.randn(2, 3, 128, dtype=torch.float32)
    first = _independent_round(x, mode)
    expected_input = _independent_round(first.flip(-1), mode) if rotated else first
    expected_output = F.linear(expected_input, dense.weight, dense.bias)
    if version == 2:
        expected_output = _independent_round(expected_output, mode)

    with torch.inference_mode():
        actual = hooked(x)

    assert hooked._w4a_stream_replay_mode == mode
    assert hooked._w4a_stream_replay_version == version
    assert hooked._w4a_stream_replay_pre_hook is True
    assert len(seen) == 1
    atol = 0
    torch.testing.assert_close(seen[0], expected_input, rtol=0, atol=atol)
    torch.testing.assert_close(actual, expected_output, rtol=0, atol=atol)


@pytest.mark.parametrize("mode", ["w4afp8"])
@pytest.mark.parametrize("version", [2, 3])
def test_hooked_linear_consumes_pre_rotated_replay_operand_once(monkeypatch, mode, version):
    dense = torch.nn.Linear(128, 128, bias=False, dtype=torch.float32).eval()
    dense._w4a_stream_replay_mode = mode
    dense._w4a_stream_replay_recipe = None
    dense._w4a_stream_replay_version = version
    dense._w4a_stream_replay_pre_hook = True
    dense._w4a_rotation_preapplied = True
    dense.online_full_had = True
    dense.online_partial_had = False
    dense.had_dim = -1
    dense.had_K = None
    dense.K = 1
    hooked = HookedLinear.from_linear(dense).eval()
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda _value, **_kwargs: (_ for _ in ()).throw(AssertionError("rotation repeated")),
    )
    seen = []
    hooked.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    operand = _independent_round(torch.randn(2, 3, 128), mode)
    expected = F.linear(operand, dense.weight)
    if version == 2:
        expected = _independent_round(expected, mode)

    with torch.inference_mode():
        actual = hooked(operand)

    assert hooked._w4a_rotation_preapplied is True
    torch.testing.assert_close(seen[0], operand, rtol=0, atol=0)
    atol = 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


@pytest.mark.parametrize("mode", ["w4afp8"])
@pytest.mark.parametrize("version", [2, 3])
def test_installed_replay_rounds_pre_rotated_down_operand_once(monkeypatch, mode, version):
    """The MLP wrapper and generic Linear pre-hook must not both QDQ down input."""
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    down = model.model.layers[0].mlp.down_proj
    down.online_full_had = True
    down.online_partial_had = False
    down.had_dim = -1
    down.had_K = None
    down.K = 1
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "version": version, "mode": mode,
        },
        offload_to_disk=False,
    )
    calls = []

    def count_round(value, *_args, **_kwargs):
        calls.append(tuple(value.shape))
        return value

    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.w4a_llama_replay._round", count_round
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.rotation.hadamard_utils.apply_online_hadamard",
        lambda value, **_kwargs: value,
    )
    install_w4a_llama_replay(model, qcfg)
    with torch.inference_mode():
        model.model.layers[0].mlp(torch.randn(2, 3, 128))

    # Version 2 rounds gate/up inputs and outputs plus the transformed down
    # input and output. Version 3 rounds only the three GEMM inputs.
    assert len(calls) == (6 if version == 2 else 3)
    assert sum(shape[-1] == 256 for shape in calls) == (3 if version == 2 else 1)
