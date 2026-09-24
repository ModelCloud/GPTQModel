# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Saved model coverage for the public staged scalar GSQ lifecycle."""

import pytest
import torch
import math


@pytest.mark.parametrize("method,bits,initializer", [
    ("gptq", 2, "gptq"), ("gptq", 3, "gptq"), ("gptq", 4, "gptq"),
    ("gptq", 4, "rtn"), ("awq", 4, "awq"),
])
def test_staged_llama_checkpoint_roundtrip(tmp_path, method, bits, initializer):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig, FORMAT, GPTQConfig
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=2)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    if method == "awq":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        model.model.rotary_emb = LlamaRotaryEmbedding(config)
    model.save_pretrained(tmp_path / "dense")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="0")),
        unk_token="0", pad_token="0")
    training = dict(enabled=True, initializer=initializer, epochs=1, qk_steps=2,
                    batch_size=1, microbatch_size=1)
    if method == "gptq":
        quant_config = GPTQConfig(bits=bits, group_size=32, format=FORMAT.GPTQ_V2,
                                  act_group_aware=False, device=DEVICE.CPU, offload_to_disk=False,
                                  gsq_training=training)
    else:
        quant_config = AWQConfig(bits=4, group_size=32, sym=False, device=DEVICE.CPU,
                                 offload_to_disk=False, gsq_training=training)
    wrapper = LlamaQModel(model=model, quantized=False, quantize_config=quant_config,
                          tokenizer=tokenizer, model_local_path=str(tmp_path / "dense"))
    wrapper.quantize([{"input_ids": list(range(1, 17))}], backend=BACKEND.TORCH,
                     calibration_data_min_length=1)
    assert wrapper.quantized
    assert wrapper.quantize_config.gsq_training.enabled
    packed_type = TorchLinear if method == "gptq" else AwqTorchLinear
    for layer in wrapper.model.model.layers:
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert isinstance(layer.get_submodule(name), packed_type)
    if method == "awq":
        assert len(quant_config.meta["gsq_training_runs"]) == 2
        runs = quant_config.meta["gsq_training_runs"]
    else:
        assert len(wrapper.gsq_training_run["blocks"]) == 2
        runs = wrapper.gsq_training_run["blocks"]
    for block in runs:
        assert set(block["stages"]) == {"self_attn.q_proj", "self_attn.k_proj", "attention", "mlp"}
        for stage in block["stages"].values():
            assert math.isfinite(stage["hard_loss_before"])
            assert math.isfinite(stage["hard_loss_after"])

    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = wrapper.model(ids, use_cache=False).logits
    wrapper.save(str(tmp_path / "quantized"))
    loaded = GPTQModel.load(str(tmp_path / "quantized"), backend=BACKEND.TORCH,
                            device="cpu", dtype=torch.float16, attn_implementation="eager")
    assert loaded.quantize_config.gsq_training.to_dict() == wrapper.quantize_config.gsq_training.to_dict()
    with torch.no_grad():
        after = loaded.model(ids, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
