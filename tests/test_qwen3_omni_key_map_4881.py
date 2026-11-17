# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni checkpoint <-> runtime key map (issue #4881).

transformers 5.22 builds the Omni thinker with the towers under
``thinker.model.visual``/``thinker.model.audio_tower`` and the decoder under
``thinker.model.language_model``, while a Qwen3-Omni checkpoint stores
``thinker.visual.*``, ``thinker.audio_tower.*`` and ``thinker.model.*``. The
conversion registry carries those renames only for the nested
``qwen3_omni_moe_thinker`` model type, so the lazy loader left ``embed_tokens``,
the rotary embeddings and both towers on META. ``quantize()`` then died during
input capture and a save would have written the runtime layout (#4758).

The map is shared with Qwen2.5-Omni because the two families use the same
checkpoint layout. The end-to-end test uses a tiny random model (no 30B
download, no processor files) and the Torch backend on CPU.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel


pytest.importorskip("transformers.models.qwen3_omni_moe")

from transformers import (  # noqa: E402
    AutoProcessor,
    PreTrainedTokenizerFast,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
)

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ  # noqa: E402
from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ  # noqa: E402
from gptqmodel.quantization.config import FORMAT  # noqa: E402
from gptqmodel.utils.model import thinker_multimodal_model, thinker_text_model  # noqa: E402


RUNTIME_TO_CHECKPOINT = {
    "thinker.model.visual.patch_embed.proj.weight": "thinker.visual.patch_embed.proj.weight",
    "thinker.model.audio_tower.layers.0.self_attn.q_proj.weight": (
        "thinker.audio_tower.layers.0.self_attn.q_proj.weight"
    ),
    "thinker.model.language_model.embed_tokens.weight": "thinker.model.embed_tokens.weight",
    "thinker.model.language_model.layers.0.input_layernorm.weight": (
        "thinker.model.layers.0.input_layernorm.weight"
    ),
    "thinker.model.language_model.norm.weight": "thinker.model.norm.weight",
}


def test_qwen3_omni_uses_the_shared_omni_key_map():
    assert Qwen3OmniMoeGPTQ.HF_CONVERSION_MAP_REVERSED is not None
    assert (
        Qwen3OmniMoeGPTQ.HF_CONVERSION_MAP_REVERSED
        == Qwen2_5_OmniGPTQ.HF_CONVERSION_MAP_REVERSED
    )


@pytest.mark.parametrize("runtime,checkpoint", sorted(RUNTIME_TO_CHECKPOINT.items()))
def test_qwen3_omni_maps_runtime_keys_to_the_checkpoint_layout(runtime, checkpoint):
    assert Qwen3OmniMoeGPTQ.checkpoint_key_for_save(runtime) == checkpoint
    assert Qwen3OmniMoeGPTQ.checkpoint_key_for_load(checkpoint) == runtime


def test_qwen3_omni_key_map_round_trips_every_runtime_key():
    for runtime, checkpoint in RUNTIME_TO_CHECKPOINT.items():
        saved = Qwen3OmniMoeGPTQ.checkpoint_key_for_save(runtime)
        assert saved == checkpoint
        assert Qwen3OmniMoeGPTQ.checkpoint_key_for_load(saved) == runtime


@pytest.mark.parametrize("checkpoint", sorted(RUNTIME_TO_CHECKPOINT.values()))
def test_qwen3_omni_save_map_passes_checkpoint_keys_through(checkpoint):
    assert Qwen3OmniMoeGPTQ.checkpoint_key_for_save(checkpoint) == checkpoint


def _tiny_config():
    return Qwen3OmniMoeConfig(
        enable_audio_output=False,
        thinker_config={
            "vision_start_token_id": 60,
            "vision_end_token_id": 61,
            "audio_start_token_id": 62,
            "audio_end_token_id": 63,
            "image_token_index": 58,
            "audio_token_index": 59,
            "video_token_index": 57,
            "user_token_id": 56,
            "text_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "tie_word_embeddings": False,
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
            "audio_config": {
                "d_model": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "num_mel_bins": 8,
                "output_dim": 32,
            },
        },
    )


@pytest.fixture
def tiny_qwen3_omni_checkpoint(tmp_path, monkeypatch):
    """A random tiny Qwen3-Omni checkpoint plus a tokenizer.

    The processor files of the real checkpoint are not available here, and the
    tiny audio/vision towers are not exercised by text-only calibration, so the
    definition's optional processor load is disabled and ``AutoProcessor`` is
    pointed at the tokenizer.
    """

    monkeypatch.setattr(Qwen3OmniMoeGPTQ, "require_pkgs", [])
    monkeypatch.setattr(Qwen3OmniMoeGPTQ, "require_load_processor", False)
    monkeypatch.setattr(
        Qwen3OmniMoeGPTQ,
        "prepare_dataset",
        lambda self, calibration_dataset, **kwargs: calibration_dataset,
    )

    native = tmp_path / "native"
    torch.manual_seed(4881)
    Qwen3OmniMoeForConditionalGeneration(_tiny_config()).eval().save_pretrained(native)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"tok{i}": i for i in range(64)}, unk_token="tok3")),
        pad_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
        unk_token="tok3",
    )
    tokenizer.save_pretrained(native)
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: tokenizer),
    )
    return native


def test_qwen3_omni_quantize_save_reload_round_trip(tiny_qwen3_omni_checkpoint, tmp_path):
    """Capture no longer sees meta tensors, and the save uses the source layout."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        lm_head=False,
        device="cpu",
        format=FORMAT.GPTQ,
    )
    model = GPTQModel.load(
        str(tiny_qwen3_omni_checkpoint),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
    )
    inputs = {
        "input_ids": torch.arange(4, 36).unsqueeze(0),
        "attention_mask": torch.ones(1, 32, dtype=torch.long),
    }

    # On the unfixed definition this raises while materializing the thinker
    # (`Tensor on device meta is not on the expected device`).
    model.quantize(CalibrationConfig(dataset=[inputs]), backend=BACKEND.TORCH)

    # The lazy loader must have materialized the endpoint the crash was about.
    text_model = thinker_text_model(model.model)
    assert text_model.embed_tokens.weight.device.type == "cpu"
    assert text_model.rotary_emb.inv_freq.device.type == "cpu"

    output = tmp_path / "quantized"
    model.save(str(output))
    saved = load_file(str(output / "model.safetensors"))

    assert "thinker.audio_tower.layers.0.self_attn.q_proj.weight" in saved
    assert "thinker.visual.patch_embed.proj.weight" in saved
    assert "thinker.model.embed_tokens.weight" in saved
    assert "thinker.model.layers.0.input_layernorm.weight" in saved
    assert not any(
        key.startswith(
            ("thinker.model.audio_tower.", "thinker.model.visual.", "thinker.model.language_model.")
        )
        for key in saved
    )

    dtype = saved["thinker.model.embed_tokens.weight"].dtype
    reloaded = GPTQModel.load(str(output), backend=BACKEND.TORCH, device="cpu", dtype=dtype)
    multimodal = thinker_multimodal_model(reloaded.model)
    text_model = thinker_text_model(reloaded.model)
    for name, tensor in (
        (
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
            multimodal.audio_tower.layers[0].self_attn.q_proj.weight,
        ),
        ("thinker.visual.patch_embed.proj.weight", multimodal.visual.patch_embed.proj.weight),
        ("thinker.model.embed_tokens.weight", text_model.embed_tokens.weight),
    ):
        torch.testing.assert_close(
            tensor, saved[name], rtol=0, atol=0, msg=lambda message, name=name: f"{name}: {message}"
        )

    with torch.inference_mode():
        logits = reloaded.model.thinker(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits
    assert logits.shape == (1, 32, _tiny_config().thinker_config.text_config.vocab_size)
    assert torch.isfinite(logits).all()
