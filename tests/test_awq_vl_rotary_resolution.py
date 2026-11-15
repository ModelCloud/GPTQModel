# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must find the decoder rotary inside a multimodal wrapper.

Issue #4121 item 6: `_get_root_rotary` looked only at `model.rotary_emb`.
Qwen2-VL / Qwen2.5-VL / Gemma 3 / Mistral 3 keep the decoder rotary on
`model.language_model.rotary_emb` and do not declare `rotary_embedding`, so the
lookup returned `None`, #4109's replay then dropped the captured
`position_embeddings`, and the attention unpacked `None`:

    TypeError: cannot unpack non-iterable NoneType object

The wrapper also owns a *vision* rotary (`model.visual.rotary_pos_emb`) that
must never be used to rebuild the decoder's position embeddings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor


QWEN2_VL_PATH = Path("/monster/data/model/Qwen2-VL-2B-Instruct")


class _Rotary(nn.Module):
    """Minimal rotary embedding: returns per-position cos/sin of a known width."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(dim, dtype=torch.float32))
        self.dim = dim

    def forward(self, hidden_states, position_ids):
        batch = position_ids.shape[0]
        seq = position_ids.shape[-1]
        return (
            torch.ones(batch, seq, self.dim),
            torch.zeros(batch, seq, self.dim),
        )


class _Attention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, position_ids=None, position_embeddings=None):
        # Mirrors the multimodal attention bodies: `cos, sin = position_embeddings`.
        cos, sin = position_embeddings
        return hidden_states + cos.mean() + sin.mean()


class _Layer(nn.Module):
    def __init__(self, dim: int = 8, *, module_rotary: bool = False):
        super().__init__()
        self.self_attn = _Attention(dim)
        if module_rotary:
            self.self_attn.rotary_emb = _Rotary(dim)

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _LanguageModel(nn.Module):
    def __init__(self, dim: int = 8, *, module_rotary: bool = False):
        super().__init__()
        self.rotary_emb = _Rotary(dim)
        self.layers = nn.ModuleList([_Layer(dim, module_rotary=module_rotary)])


class _Visual(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.rotary_pos_emb = _Rotary(dim)


class _Core(nn.Module):
    def __init__(self, dim: int = 8, *, module_rotary: bool = False):
        super().__init__()
        self.visual = _Visual()
        self.language_model = _LanguageModel(dim, module_rotary=module_rotary)


class _MultimodalModel(nn.Module):
    """`model.visual` + `model.language_model.rotary_emb`, as on Qwen2-VL."""

    def __init__(self, *, module_rotary: bool = False):
        super().__init__()
        self.model = _Core(module_rotary=module_rotary)


class _PerLayerRotaryModel(nn.Module):
    """No model-level rotary: only `layers.0.self_attn.rotary_emb` (Nanbeige)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(module_rotary=True)])


def _processor(model: nn.Module, *, declared: str | None = None) -> AWQProcessor:
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(
        rotary_embedding=declared,
        awq_preserve_explicit_position_embeddings=False,
    )
    processor.model = model
    processor._rotary_lock = threading.Lock()
    processor._rotary_cache = {}
    processor._rotary_source_id = None
    return processor


def test_root_rotary_resolves_nested_language_model_rotary():
    model = _MultimodalModel()
    processor = _processor(model)

    rotary = processor._get_root_rotary()

    assert rotary is model.model.language_model.rotary_emb


def test_root_rotary_never_uses_the_vision_tower():
    model = _MultimodalModel()
    processor = _processor(model)

    rotary = processor._get_root_rotary()

    assert rotary is not model.model.visual.rotary_pos_emb
    # The vision rotary is the shallowest match by name, so a naive scan would
    # pick it and rebuild the decoder's cos/sin with the wrong width.
    assert rotary.dim == model.model.language_model.rotary_emb.dim


def test_root_rotary_prefers_a_declared_path():
    model = _MultimodalModel()
    processor = _processor(model, declared="model.visual.rotary_pos_emb")

    assert processor._get_root_rotary() is model.model.visual.rotary_pos_emb


def test_root_rotary_ignores_per_layer_rotaries():
    """Nanbeige/DeciLM keep one rotary per layer; those are not the outer one.

    They are old-style `(x, seq_len)` modules that compute cos/sin inside the
    attention, so `_prepare_module_forward` must keep handling them through its
    module-local fallback instead of replaying them from the model root.
    """

    model = _PerLayerRotaryModel()
    processor = _processor(model)

    assert processor._get_root_rotary() is None


def test_root_rotary_is_cached_per_processor():
    model = _MultimodalModel()
    processor = _processor(model)

    first = processor._get_root_rotary()
    assert processor._get_root_rotary() is first


def test_prepare_module_forward_rebuilds_embeddings_for_a_nested_rotary():
    """The crash path: stale cos/sin must be rebuilt, not dropped."""

    model = _MultimodalModel()
    processor = _processor(model)
    attention = model.model.language_model.layers[0].self_attn
    x = torch.randn(20, 8)
    stale = (torch.ones(1, 48, 8), torch.zeros(1, 48, 8))

    _, kwargs = processor._prepare_module_forward(
        x,
        attention,
        {"position_ids": torch.arange(48).unsqueeze(0), "position_embeddings": stale},
    )

    assert "position_embeddings" in kwargs, "dropping them makes the attention unpack None"
    cos, sin = kwargs["position_embeddings"]
    assert cos.shape == (1, 20, 8)
    assert sin.shape == (1, 20, 8)
    assert kwargs["position_ids"].shape == (1, 20)


@pytest.mark.skipif(
    not QWEN2_VL_PATH.is_dir(),
    reason=f"local Qwen2-VL fixture is unavailable: {QWEN2_VL_PATH}",
)
@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_tiny_qwen2_vl_quantizes_with_awq_and_gptq(method, tmp_path, monkeypatch):
    """End-to-end: AWQ reaches the decoder's nested rotary instead of crashing."""

    from gptqmodel import GPTQModel
    from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
    from gptqmodel.quantization.config import AWQConfig, GPTQConfig
    from tests.tiny_remote_models import build_tiny_qwen2_vl

    # `AutoProcessor` builds a `Qwen2VLVideoProcessor`, which requires
    # torchvision. Quantization only needs the tokenizer, so skip the processor
    # and hand the VL definition the already-tokenized calibration batches
    # instead of chat conversations. The rotary bug is in the AWQ scale search,
    # not in dataset rendering.
    monkeypatch.setattr(Qwen2VLQModel, "require_load_processor", False)
    monkeypatch.setattr(
        Qwen2VLQModel,
        "prepare_dataset",
        lambda self, calibration_dataset, batch_size=1, **kwargs: list(
            calibration_dataset
        ),
    )

    target = build_tiny_qwen2_vl(QWEN2_VL_PATH, tmp_path / "tiny")

    quantize_config = (
        AWQConfig(bits=4, group_size=32, device="cpu")
        if method == "awq"
        else GPTQConfig(bits=4, group_size=32, device="cpu")
    )
    model = GPTQModel.load(
        str(target),
        quantize_config=quantize_config,
        trust_remote_code=True,
        # `AwqTorchLinear` only supports fp16/bf16.
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(4, 128, (1, length), generator=generator),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        for length in (48, 20)
    ]
    model.quantize(calibration, batch_size=1, calibration_sort="desc")

    q_proj = model.model.get_submodule(
        "model.language_model.layers.0.self_attn.q_proj"
    )
    assert not isinstance(q_proj, nn.Linear)
