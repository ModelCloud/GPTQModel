# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4395: quantization and checkpoint load must release pinned host memory.

``move_module_to_cpu_pinned`` and accelerate's checkpoint staging allocate
non-swappable host blocks. Deleting the staging tensor only returns the block
to PyTorch's process-wide pinned cache, so ``RssShmem`` never drops without an
explicit ``torch._C._host_emptyCache()``.
"""

from __future__ import annotations

import gc

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import CalibrationConfig
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights
from gptqmodel.utils.torch import release_pinned_host_cache


_PINNED_BYTES = 8 * 1024 * 1024


def _pinned_bytes() -> int | None:
    stats = getattr(torch._C, "_cuda_hostMemoryStats", None)
    if stats is None:
        return None
    try:
        return int(stats()["allocated_bytes"]["current"])
    except Exception:  # pragma: no cover - other Torch builds
        return None


def _require_pinned_cache() -> int:
    if not torch.cuda.is_available():
        pytest.skip("pinned host memory requires CUDA")
    current = _pinned_bytes()
    if current is None:
        pytest.skip("this Torch build exposes no pinned-host allocator stats")
    return current


def _seed_pinned_block() -> None:
    tensor = torch.empty(_PINNED_BYTES, dtype=torch.uint8, pin_memory=True)
    assert tensor.is_pinned()
    del tensor
    gc.collect()


def test_release_pinned_host_cache_returns_cached_blocks():
    baseline = _require_pinned_cache()
    _seed_pinned_block()
    cached = _pinned_bytes()
    assert cached is not None and cached >= baseline + _PINNED_BYTES

    assert release_pinned_host_cache() is True

    after = _pinned_bytes()
    assert after is not None and after <= baseline


def test_checkpoint_loader_releases_pinned_host_cache(tmp_path):
    baseline = _require_pinned_cache()
    _seed_pinned_block()
    assert _pinned_bytes() >= baseline + _PINNED_BYTES

    torch.manual_seed(4395)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
    ).eval()
    checkpoint = tmp_path / "model.safetensors"
    save_file({name: value.detach().clone() for name, value in model.state_dict().items()}, str(checkpoint))

    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
    )

    after = _pinned_bytes()
    assert after is not None and after <= baseline


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = (
        "[UNK]", "[PAD]", "<s>", "</s>", "A", "short", "calibration",
        "sentence", "about", "fox", "Another", "moon", ".", "runs",
    )
    inner = Tokenizer(
        models.WordLevel(vocab={word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )


def test_quantize_releases_pinned_host_cache(tmp_path, monkeypatch):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models import base as base_module
    from gptqmodel.quantization.config import QuantizeConfig

    calls: list[bool] = []
    monkeypatch.setattr(base_module, "release_pinned_host_cache", lambda: calls.append(True) or True)

    torch.manual_seed(4395)
    source = tmp_path / "source"
    source.mkdir()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    _tiny_tokenizer().save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        QuantizeConfig(bits=4, group_size=32, offload_to_disk=False),
        device="cpu",
        dtype=torch.float32,
    )
    model.quantize(
        CalibrationConfig(
            dataset=["A short calibration sentence about a fox. A short calibration sentence about a fox."],
            batch_size=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )

    assert calls == [True]
