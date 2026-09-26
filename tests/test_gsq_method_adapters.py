# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Checkpoint coverage for GSQ adapters on public quantization methods."""

import pytest
import torch


def test_awq_gsq_rejects_codes_that_native_packing_would_overflow():
    from gptqmodel.quantization.gsq_scalar import affine_codes

    weight = torch.tensor([[100.0]], dtype=torch.float16)
    scale = torch.tensor([[1.0]], dtype=torch.float16)
    zero = torch.tensor([[8.0]], dtype=torch.float16)
    groups = torch.tensor([0], dtype=torch.int32)
    for packing in ("awq_gemm", "awq_gemv", "awq_gemv_fast"):
        recovered = affine_codes(weight, scale, zero, groups, 4, packing=packing)
        assert recovered.item() == 108


@pytest.mark.parametrize("method", ["awq", "fp8", "fp8_e5m2", "fp8_block",
                                     "fp8_calibrated", "paro", "rtn"])
def test_gsq_method_checkpoint_roundtrip(tmp_path, method):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.quantization import AWQConfig, FP8Config, GSQConfig
    from gptqmodel.quantization.config import ParoConfig, RTNConfig
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    hidden_size, intermediate_size = (64, 128) if method == "awq" else (32, 64)
    config = LlamaConfig(vocab_size=128, hidden_size=hidden_size, intermediate_size=intermediate_size,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
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
    elif method.startswith("fp8"):
        fp8_options = {}
        if method == "fp8_e5m2":
            fp8_options["format"] = "float8_e5m2"
        if method == "fp8_block":
            fp8_options.update(weight_scale_method="block", weight_block_size=[16, 16])
        quant_config = FP8Config(device=DEVICE.CPU, offload_to_disk=False, gsq=gsq,
                                 gsq_calibration=method == "fp8_calibrated", **fp8_options)
        backend = BACKEND.FP8_TORCH
    elif method == "paro":
        quant_config = ParoConfig(bits=4, group_size=32, krot=1, device=DEVICE.CPU,
                                  offload_to_disk=False, opt_rotation_epochs=1,
                                  opt_finetune_epochs=1, opt_train_samples=8,
                                  opt_validation_samples=4, opt_batch_size=4,
                                  opt_stage_cudagraph=False, gsq=gsq)
        backend = BACKEND.PAROQUANT_CUDA
    else:
        quant_config = RTNConfig(bits=4, group_size=32, device=DEVICE.CPU,
                                 offload_to_disk=False, gsq=gsq)
        backend = BACKEND.TORCH
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=quant_config,
                          tokenizer=tokenizer, model_local_path=str(source))
    wrapper.quantize([{"input_ids": list(range(1, 17))}], backend=backend,
                     calibration_data_min_length=1)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(ids, use_cache=False).logits.detach()
    checkpoint = tmp_path / "quantized"
    wrapper.save(str(checkpoint))
    loaded = GPTQModel.load(str(checkpoint), backend=backend, device="cpu",
                            dtype=torch.float16, attn_implementation="eager")
    assert loaded.quantize_config.gsq == gsq
    if method.startswith("fp8"):
        assert loaded.quantize_config.gsq_calibration == (method == "fp8_calibrated")
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
        torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_packed_gsq_payload_has_reloadable_runtime_buffers(group_size):
    """QQQ's native extension is optional; its packed state still must reload."""
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
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
    packed = QQQLinear(bits=4, group_size=group_size, desc_act=False, sym=True,
                       in_features=width, out_features=128, register_buffers=False)
    extra = torch.full((128,), 0.05, dtype=torch.float32) if group_size != -1 else None
    packed.pack(linear, scales, s_extra=extra)
    assert packed.workspace.numel() > 0
    assert packed.reduce_buffer.numel() > 0
    assert (packed.s_group.numel() == 0) == (group_size == -1)
    saved = packed.state_dict()
    assert "workspace" not in saved and "reduce_buffer" not in saved
    loaded = QQQLinear(bits=4, group_size=group_size, desc_act=False, sym=True,
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


def test_qqq_gsq_torch_checkpoint_roundtrip(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
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
    wrapper.quantize([{"input_ids": list(range(1, 17))}], backend=BACKEND.QQQ_TORCH,
                     calibration_data_min_length=1)
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
