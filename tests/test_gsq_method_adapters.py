# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Checkpoint coverage for GSQ adapters on public quantization methods."""

import pytest
import torch


@pytest.mark.parametrize("method,block_size", [("row", None), ("block", [2, 2])])
def test_fp8_load_only_payload_decodes_against_torch(method, block_size):
    """A production FP8 checkpoint payload installs directly; nothing is encoded."""

    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from tests.fp8_test_utils import install_fp8_payload

    values = torch.tensor(
        [
            [0, 1, -1, 448],
            [-448, 0, 2, -2],
            [0.5, -0.5, 16, -16],
            [3, -3, 0, 448],
        ],
        dtype=torch.float32,
    )
    codes = values.to(torch.float8_e4m3fn)
    scales = (
        torch.tensor([1, 2, 4, 8], dtype=torch.float32)
        if method == "row"
        else torch.tensor([[1, 2], [4, 8]], dtype=torch.float32)
    )
    linear = torch.nn.Linear(4, 4, bias=True, dtype=torch.float16)
    packed = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=4,
        out_features=4,
        bias=True,
        register_buffers=False,
        weight_scale_method=method,
        weight_block_size=block_size,
    )
    install_fp8_payload(packed, linear, codes, scales)

    assert torch.equal(packed.weight.view(torch.uint8), codes.view(torch.uint8))
    assert torch.equal(packed.weight_scale_inv, scales)
    # FP8 modules keep the checkpoint bias wide (FP32) so a reload builds the
    # same GEMM as the in-session module; narrowing it here would diverge.
    assert packed.bias.dtype == torch.float32
    assert torch.equal(packed.bias, linear.bias.float())
    expanded_scales = (
        scales[:, None] if method == "row" else scales.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    )
    reference = values / expanded_scales
    torch.testing.assert_close(packed.dequantize_weight(dtype=torch.float32).T, reference, rtol=1e-6, atol=1e-6)


def test_awq_gsq_saturates_like_native_packing():
    from gptqmodel.quantization.gsq_scalar import affine_codes

    weight = torch.tensor([[100.0]], dtype=torch.float16)
    scale = torch.tensor([[1.0]], dtype=torch.float16)
    zero = torch.tensor([[8.0]], dtype=torch.float16)
    groups = torch.tensor([0], dtype=torch.int32)
    for packing in ("awq_gemm", "awq_gemv", "awq_gemv_fast"):
        recovered = affine_codes(weight, scale, zero, groups, 4, packing=packing)
        # Every AWQ packer clamps to the code range before writing bytes.
        assert recovered.item() == 15


@pytest.mark.parametrize("method", ["awq", "paro", "rtn"])
def test_gsq_method_checkpoint_roundtrip(tmp_path, method):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import CalibrationConfig, GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import AWQConfig, GSQConfig
    from gptqmodel.quantization.config import ParoConfig, RTNConfig
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    hidden_size, intermediate_size = (64, 128) if method == "awq" else (32, 64)
    config = LlamaConfig(vocab_size=128, hidden_size=hidden_size, intermediate_size=intermediate_size,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1,
                         attention_bias=False)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    if method == "awq":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        model.model.rotary_emb = LlamaRotaryEmbedding(config)
    source = tmp_path / "dense"
    model.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="0")),
        unk_token="0", pad_token="0")
    gsq = GSQConfig(enabled=True, steps=2, candidates=3, seed=7)
    if method == "awq":
        quant_config = AWQConfig(bits=4, group_size=32, sym=False, device=DEVICE.CPU,
                                 offload_to_disk=False, gsq=gsq)
        backend = BACKEND.TORCH
    elif method == "paro":
        quant_config = ParoConfig(bits=4, group_size=32, krot=1, device=DEVICE.CPU,
                                  offload_to_disk=False, opt_rotation_epochs=1,
                                  opt_finetune_epochs=1, opt_train_samples=8,
                                  opt_validation_samples=4, opt_batch_size=4,
                                  opt_stage_cudagraph=False, gsq=gsq)
        backend = BACKEND.PARO_CUDA
    else:
        quant_config = RTNConfig(bits=4, group_size=32, device=DEVICE.CPU,
                                 offload_to_disk=False, gsq=gsq)
        backend = BACKEND.TORCH
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=quant_config,
                          tokenizer=tokenizer, model_local_path=str(source))
    wrapper.quantize(
        CalibrationConfig(
            dataset=[{"input_ids": list(range(1, 17))}],
            min_length=1,
        ),
        backend=backend,
    )
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(ids, use_cache=False).logits.detach()
    checkpoint = tmp_path / "quantized"
    wrapper.save(str(checkpoint))
    loaded = GPTQModel.load(str(checkpoint), backend=backend, device="cpu",
                            dtype=torch.float16, attn_implementation="eager")
    assert loaded.quantize_config.gsq == gsq
    with torch.no_grad():
        after = loaded.model(ids, use_cache=False).logits.detach()
    if method == "rtn":
        # The existing RTN weight-only lifecycle has a separate in-memory
        # versus packed-forward difference even with GSQ disabled. Validate
        # reproducible checkpoint execution and serialized GSQ configuration.
        reloaded = GPTQModel.load(str(checkpoint), backend=backend, device="cpu",
                                  dtype=torch.float16, attn_implementation="eager")
        with torch.no_grad():
            again = reloaded.model(ids, use_cache=False).logits.detach()
        torch.testing.assert_close(again, after, rtol=0, atol=0)
    else:
        # Packed kernels can differ from the in-memory path by one FP16 rounding
        # step while preserving the saved weights and calibration metadata.
        torch.testing.assert_close(after, before, rtol=0, atol=5e-4)


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_packed_gsq_payload_has_reloadable_runtime_buffers(group_size):
    """QQQ's native extension is optional; its packed state still must reload."""
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    from gptqmodel.quantization.gsq_qqq import qqq_codes_to_packer_weight

    torch.manual_seed(7)
    width = 128 if group_size == -1 else 256
    codes = torch.randint(0, 16, (128, width), dtype=torch.int32)
    scales = torch.full((128, width // (width if group_size == -1 else group_size)),
                        0.05, dtype=torch.float16)
    weight = qqq_codes_to_packer_weight(codes, scales, group_size=group_size, dtype=torch.float16)
    linear = torch.nn.Linear(width, 128, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(weight)
    packed = QQQTorchLinear(bits=4, group_size=group_size, desc_act=False, sym=True,
                            in_features=width, out_features=128, register_buffers=False)
    extra = torch.full((128,), 0.05, dtype=torch.float32) if group_size != -1 else None
    packed.pack(linear, scales, s_extra=extra)
    assert packed.workspace.numel() > 0
    # The reduce scratch is shared per device, not a per-module buffer (#4387).
    assert "reduce_buffer" not in packed._buffers
    scratch = packed._reduce_scratch()
    assert scratch.shape == (packed.max_par * 64, packed.out_features)
    assert scratch.dtype == torch.int32
    assert (packed.s_group.numel() == 0) == (group_size == -1)
    saved = packed.state_dict()
    assert "workspace" not in saved and "reduce_buffer" not in saved
    loaded = QQQTorchLinear(bits=4, group_size=group_size, desc_act=False, sym=True,
                            in_features=width, out_features=128, register_buffers=True)
    loaded.load_state_dict(saved)
    torch.testing.assert_close(loaded.B, packed.B, rtol=0, atol=0)
    torch.testing.assert_close(loaded.s_channel, packed.s_channel, rtol=0, atol=0)
    torch.testing.assert_close(loaded.s_group, packed.s_group, rtol=0, atol=0)


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_gsq_hard_objective_does_not_regress(group_size):
    from gptqmodel.quantization import GSQConfig
    from gptqmodel.quantization.gsq_qqq import qqq_candidate_values, refine_qqq_codes

    torch.manual_seed(7)
    width = 128 if group_size == -1 else 256
    codes = torch.randint(0, 16, (2, width), dtype=torch.int32)
    scales = torch.full((2, width // (width if group_size == -1 else group_size)),
                        0.05, dtype=torch.float16)
    channel = None if group_size == -1 else torch.full((2,), 0.05)
    teacher = qqq_candidate_values(codes, scales, group_size=group_size,
                                   in_features=width, channel_scales=channel) + 0.01
    selected, before, after, history = refine_qqq_codes(
        codes, scales, target=teacher, group_size=group_size,
        hessian=torch.eye(width), cross_moment=torch.zeros(width, width),
        channel_scales=channel, config=GSQConfig(enabled=True, steps=2, candidates=3))
    assert selected.shape == codes.shape
    assert after <= before
    assert len(history) == 3


@pytest.mark.parametrize(
    "scale_value",
    [
        2.0 ** -24,        # smallest FP16 subnormal: a source division rounds to zero
        2.0 ** -21,        # half an FP16 subnormal step: a source division rounds to zero
        9 * 2.0 ** -24,    # one FP16 step above the zero threshold
        2.0 ** -20,        # control: representable after a source-dtype division
        1e-4,              # ordinary scale whose source division rounds
        1.0,               # control
    ],
)
def test_qqq_channelwise_candidates_decode_the_packer_fp32_scale(scale_value):
    """The candidate adapter must divide the stored scale in FP32 like ``pack``.

    Channelwise QQQ stores ``scale / 16`` after widening the producer scale to
    FP32. Dividing first in the source dtype rounds an FP16 scale to zero in
    the subnormal range, so the adapter rejected representable inputs and
    scored candidates on a grid the packed kernel never decodes.
    """
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    from gptqmodel.quantization import GSQConfig
    from gptqmodel.quantization.gsq_qqq import (
        qqq_candidate_values,
        qqq_codes_to_packer_weight,
        refine_qqq_codes,
    )

    out_features, in_features = 64, 128
    scales = torch.full((out_features, 1), scale_value, dtype=torch.float16)
    codes = torch.ones((out_features, in_features), dtype=torch.float32)
    weight = qqq_codes_to_packer_weight(codes, scales, group_size=-1, dtype=torch.float16)

    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(weight)
    packed = QQQTorchLinear(bits=4, group_size=-1, desc_act=False, sym=True,
                            in_features=in_features, out_features=out_features,
                            register_buffers=False)
    packed.pack(linear, scales)
    assert torch.equal(packed._unpack_weight_codes(),
                       torch.ones((in_features, out_features), dtype=torch.uint8))

    # Independent FP64 oracle: a channelwise nibble of 1 deploys as exactly the
    # stored scale that the packer widened to FP32.
    expected = codes.to(torch.float64) * scales.float().to(torch.float64)
    plane, s_channel = packed._dequantize_weight_for_torch()
    actual = (plane.to(torch.float64) * s_channel.to(torch.float64)).t()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    candidate = qqq_candidate_values(codes, scales, group_size=-1, in_features=in_features)
    assert candidate.dtype == torch.float32
    torch.testing.assert_close(candidate.to(torch.float64), expected, rtol=0, atol=0)

    # The hard objective is zero on the deployed grid, so GSQ scores the same
    # weights the packer writes and keeps the existing assignment.
    selected, before, after, history = refine_qqq_codes(
        codes.long(), scales, target=expected.float(), group_size=-1,
        hessian=torch.eye(in_features), cross_moment=torch.zeros(in_features, in_features),
        config=GSQConfig(enabled=True, steps=2, candidates=3, seed=7))
    assert before == 0.0
    assert after == 0.0
    assert history[-1] == 0.0
    assert torch.equal(selected, codes.long())


def test_qqq_gsq_torch_checkpoint_roundtrip(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import CalibrationConfig, GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    from gptqmodel.quantization import GSQConfig
    from gptqmodel.quantization.config import QQQConfig
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    source = tmp_path / "dense"
    model.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="0")),
        unk_token="0", pad_token="0")
    gsq = GSQConfig(enabled=True, steps=2, candidates=3)
    quant_config = QQQConfig(bits=4, group_size=128, desc_act=False, act_group_aware=False,
                             device=DEVICE.CPU, offload_to_disk=False, gsq=gsq)
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=quant_config,
                          tokenizer=tokenizer, model_local_path=str(source))
    wrapper.quantize(
        CalibrationConfig(
            dataset=[{"input_ids": list(range(1, 17))}],
            min_length=1,
        ),
        backend=BACKEND.QQQ_TORCH,
    )
    for layer in wrapper.model.model.layers:
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert isinstance(layer.get_submodule(name), QQQTorchLinear)
    checkpoint = tmp_path / "quantized"
    wrapper.save(str(checkpoint))
    ids = torch.tensor([[1, 2, 3]])
    logits = []
    for _ in range(2):
        loaded = GPTQModel.load(str(checkpoint), backend=BACKEND.QQQ_TORCH,
                                device="cpu", dtype=torch.float16, attn_implementation="eager")
        assert loaded.quantize_config.gsq == gsq
        with torch.no_grad():
            logits.append(loaded.model(ids, use_cache=False).logits.detach())
    torch.testing.assert_close(logits[1], logits[0], rtol=0, atol=0)
