# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4875: a quantize device sequence may span GPU generations.

`device=("cuda:0", "cuda:1")` on a 5090 + 4090 host was refused with
"Quantized kernel selection does not support a device map spanning different
GPU compute capabilities", although the default pool and a
`compute_device_filter` that keeps both GPUs quantize on the same pair.

The single-capability rule exists because one *inference* kernel is selected
for a whole placement map (#4632/#4705/#4734). A quantization load runs the
dense model on CPU and selects no inference kernel for its pool, so its entry
points must not apply that rule.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models import auto, loader
from gptqmodel.utils import importer
from gptqmodel.utils.importer import normalize_device_device_map


def _mixed_capabilities(monkeypatch):
    """cuda:0 = sm_120 (5090), cuda:1 = sm_89 (4090)."""

    monkeypatch.setattr(
        importer,
        "_device_capability",
        lambda target: (12, 0) if target.index == 0 else (8, 9),
    )


def test_inference_sequence_still_requires_one_capability(monkeypatch):
    _mixed_capabilities(monkeypatch)

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        loader._explicit_requested_devices(("cuda:0", "cuda:1"))
    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        normalize_device_device_map(("cuda:0", "cuda:1"), None)


def test_quantize_sequence_may_span_capabilities(monkeypatch):
    _mixed_capabilities(monkeypatch)

    assert loader._explicit_requested_devices(
        ("cuda:0", "cuda:1"), validate_families=False
    ) == [torch.device("cuda:0"), torch.device("cuda:1")]
    assert normalize_device_device_map(
        ("cuda:0", "cuda:1"), None, validate_families=False
    ) == (torch.device("cuda:0"), torch.device("cuda:1"))


def test_quantize_warmup_accepts_a_mixed_sequence(monkeypatch):
    """The prewarm resolver must not re-apply the inference capability rule."""

    from gptqmodel.quantization.config import GPTQConfig

    _mixed_capabilities(monkeypatch)
    monkeypatch.setattr(auto, "auto_select_device", lambda device, backend: device)

    devices = auto._quantize_warmup_devices(
        GPTQConfig(bits=4, group_size=128), ("cuda:0", "cuda:1")
    )

    assert devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_mixed_generation_sequence_loads_for_quantization(tmp_path):
    """End to end on a host that really has two generations."""

    import transformers

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible GPUs")
    capabilities = {
        torch.cuda.get_device_capability(index)
        for index in range(torch.cuda.device_count())
    }
    if len(capabilities) < 2:
        pytest.skip("the visible GPUs share one compute capability")

    model_dir = tmp_path / "tiny"
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=GPTQConfig(bits=4, group_size=32),
        dtype=torch.float16,
        device=(0, 1),
    )

    assert model.quantize_config.compute_device_filter is not None
    wanted = {
        str(target) for target in model.quantize_config.compute_device_filter(
            [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
        )
    }
    assert wanted == {"cuda:0", "cuda:1"}
