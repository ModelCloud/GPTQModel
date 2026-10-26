# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the nested thinker output head (issue #1841)."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT, RTNConfig
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format


def _tiny_omni():
    config = Qwen2_5OmniConfig(
        enable_audio_output=False,
        thinker_config={
            "vision_start_token_id": 120, "vision_end_token_id": 121,
            "audio_start_token_id": 122, "audio_end_token_id": 123,
            "image_token_index": 124, "audio_token_index": 125, "video_token_index": 126,
            "user_token_id": 127,
            "text_config": {
                "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
                "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
                "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]},
            },
            "vision_config": {
                "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                "out_hidden_size": 64, "fullatt_block_indexes": [0],
            },
            "audio_config": {
                "d_model": 16, "encoder_layers": 1, "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32, "num_mel_bins": 8, "output_dim": 64,
            },
        },
    )
    torch.manual_seed(1841)
    return Qwen2_5OmniForConditionalGeneration(config).eval()


@pytest.fixture
def tiny_omni_checkpoint(tmp_path, monkeypatch):
    # This regression exercises pretokenized text calibration, independently of
    # the multimodal processor and its external audio/video dependencies.
    monkeypatch.setattr(Qwen2_5_OmniGPTQ, "require_load_processor", False)
    monkeypatch.setattr(Qwen2_5_OmniGPTQ, "require_pkgs", [])
    monkeypatch.setattr(
        Qwen2_5_OmniGPTQ, "prepare_dataset", lambda self, calibration_dataset, **kwargs: calibration_dataset,
    )
    native = tmp_path / "native"
    _tiny_omni().save_pretrained(native)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"tok{i}": i for i in range(128)}, unk_token="tok3")),
        pad_token="tok0", bos_token="tok1", eos_token="tok2", unk_token="tok3",
    )
    tokenizer.save_pretrained(native)
    torch.save({"Chelsie": {}}, native / "spk_dict.pt")
    return native


def test_omni_lm_head_resolves_to_thinker():
    model = _tiny_omni()
    assert model.get_submodule(Qwen2_5_OmniGPTQ.lm_head) is model.thinker.lm_head


@pytest.mark.parametrize("method", ["gptq", "rtn"])
@pytest.mark.parametrize("head_bits", [4, 8])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_omni_lm_head_quantize_save_reload(tiny_omni_checkpoint, tmp_path, method, head_bits, device):
    config_cls = QuantizeConfig if method == "gptq" else RTNConfig
    # GPTQ defaults the output head to 8-bit; other cases use an explicit override.
    dynamic = None if method == "gptq" and head_bits == 8 else {
        r"thinker\.lm_head$": {"bits": head_bits, "group_size": 32, "desc_act": False},
    }
    qcfg = config_cls(
        bits=4, group_size=32, desc_act=False, lm_head=True, device=device,
        offload_to_disk=False, format=FORMAT.GPTQ, dynamic=dynamic,
    )
    model = GPTQModel.load(
        str(tiny_omni_checkpoint), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16,
    )
    inputs = {"input_ids": torch.arange(4, 36).unsqueeze(0), "attention_mask": torch.ones(1, 32, dtype=torch.long)}
    model.quantize([inputs], backend=BACKEND.TORCH)
    head = model.model.thinker.lm_head
    assert isinstance(head, TorchLinear)
    assert head.bits == head_bits
    assert isinstance(model.model.thinker.model.layers[0].self_attn.q_proj, TorchLinear)
    assert not any(isinstance(module, TorchLinear) for module in model.model.thinker.visual.modules())
    output = tmp_path / "quantized"
    model.save(str(output))
    # Packing produces checkpoint-format zeros. Prepare the in-memory model
    # for Torch inference exactly as the quantized loader does after reading v1.
    convert_gptq_v1_to_v2_format(model.model, qcfg, TorchLinear)
    model.model.to(device).eval()
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    runtime_state = {name: tensor.clone() for name, tensor in model.model.state_dict().items()}

    reloaded = GPTQModel.load(str(output), backend=BACKEND.TORCH, device=device, dtype=torch.float16)
    assert isinstance(reloaded.model.thinker.lm_head, TorchLinear)
    assert reloaded.model.thinker.lm_head.bits == head_bits
    for name, tensor in reloaded.model.state_dict().items():
        torch.testing.assert_close(tensor, runtime_state[name], rtol=0, atol=0, msg=lambda msg: f"{name}: {msg}")
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**inputs).logits, logits, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["gptq", "rtn"])
def test_omni_lm_head_rejects_shared_weights(tiny_omni_checkpoint, method):
    config_cls = QuantizeConfig if method == "gptq" else RTNConfig
    qcfg = config_cls(bits=4, group_size=32, lm_head=True, device="cpu", offload_to_disk=False)
    model = GPTQModel.load(str(tiny_omni_checkpoint), quantize_config=qcfg, backend=BACKEND.TORCH)
    embedding = model.model.thinker.model.embed_tokens
    model.model.thinker.lm_head.weight = embedding.weight
    inputs = {"input_ids": torch.arange(4, 36).unsqueeze(0), "attention_mask": torch.ones(1, 32, dtype=torch.long)}

    with pytest.raises(NotImplementedError, match="tied weights"):
        model.quantize([inputs], backend=BACKEND.TORCH)

    assert model.model.thinker.lm_head.weight is embedding.weight
    assert not any(isinstance(module, TorchLinear) for module in model.model.modules())

