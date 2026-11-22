# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4880: AWQ replay must align token-aligned kwargs.

DeepSeek-V4 begins with ``hash_moe`` layers whose router selects experts with
``tid2eid[input_ids.reshape(-1)]``. AWQ replays a *packed sample* of the
captured token rows (``token_rows`` aggregation) and used to forward the
current calibration batch's ``input_ids`` unchanged, so the router produced a
different number of index rows than the hidden states and the expert loop
raised ``IndexError`` in the first decoder layer. The ids now travel through the
same provenance as the packed rows, and a residual mismatch becomes an
``_AWQScaleGroupError`` so the configured fallback applies instead of crashing.
"""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel import GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


def _processor() -> AWQProcessor:
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace(awq_module_replay_inputs=None)
    processor.model = nn.Module()
    # Resolve "no model-root rotary" once; the stubs below take no rotary kwargs.
    processor._root_rotary_resolved = None
    processor.inputs_cache = None
    processor.tasks = {}
    processor.lock = threading.Lock()
    return processor


class _HashMoeStub(nn.Module):
    """A MoE block whose router needs one ``input_ids`` entry per token row."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))

    def forward(self, hidden_states, input_ids):  # noqa: ARG002 - signature under test
        return hidden_states


def test_packed_token_aligned_kwargs_follow_the_replayed_rows():
    processor = _processor()
    processor.inputs_cache = SimpleNamespace(
        layer_input_kwargs=[
            {"input_ids": torch.tensor([[10, 11, 12]])},
            {"input_ids": torch.tensor([[20, 21]])},
        ]
    )

    packed = processor._pack_token_aligned_kwargs(
        [(0, 2), (1, 0), (0, 0)],
        [0, 1],
    )

    assert torch.equal(packed["input_ids"], torch.tensor([[12, 20, 10]]))
    # A provenance entry the capture cannot satisfy is skipped, not guessed.
    assert processor._pack_token_aligned_kwargs([(0, 9)], [0]) == {}


def test_replay_rejects_a_token_aligned_kwarg_with_the_wrong_row_count():
    processor = _processor()
    module = _HashMoeStub()
    x = torch.zeros(1, 20, 4)

    with pytest.raises(_AWQScaleGroupError, match="token-aligned kwarg"):
        processor._prepare_module_forward(
            x, module, {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
        )

    prepared_x, prepared_kwargs = processor._prepare_module_forward(
        x, module, {"input_ids": torch.zeros(1, 20, dtype=torch.long)}
    )
    assert tuple(prepared_x.shape) == (1, 20, 4)
    assert prepared_kwargs["input_ids"].numel() == 20


def _tiny_config():
    config_cls = getattr(importlib.import_module("transformers"), "DeepseekV4Config", None)
    if config_cls is None:
        pytest.skip("transformers does not provide DeepseekV4Config")

    return config_cls(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        layer_types=["compressed_sparse_attention"] * 2,
        mlp_layer_types=["hash_moe", "hash_moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        tie_word_embeddings=False,
        pad_token_id=0,
    )


def _build_tiny_model(model_dir) -> None:
    model_cls = getattr(importlib.import_module("transformers"), "DeepseekV4ForCausalLM", None)
    if model_cls is None:
        pytest.skip("transformers does not provide DeepseekV4ForCausalLM")

    torch.manual_seed(0)
    model_cls(_tiny_config()).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)


def _calibration():
    generator = torch.Generator().manual_seed(0)
    rows = []
    # Different lengths: the packed replay covers rows from several batches, so
    # a single batch's ids can never line up with the replayed token count.
    for length in (12, 20, 8):
        input_ids = torch.randint(5, 120, (1, length), generator=generator)
        rows.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    return rows


def _enable_hash_moe_scale_search(monkeypatch):
    # The port inherits the generic planner; pin it so these hash-MoE replay
    # regressions keep exercising scale search even if the port narrows it.
    monkeypatch.setattr(DeepSeekV4QModel, "awq_get_modules_for_scaling", BaseQModel.awq_get_modules_for_scaling)


def test_awq_quantizes_a_hash_moe_layer_with_aligned_token_rows(tmp_path, monkeypatch):
    _enable_hash_moe_scale_search(monkeypatch)
    model_dir = tmp_path / "dsv4-hash-moe"
    _build_tiny_model(model_dir)

    recorded = []
    original = AWQProcessor._search_best_scale

    def recorder(self, module, prev_op, layers, inp, module2inspect=None, kwargs=None, **rest):
        token_rows = int(inp.shape[0]) * int(inp.shape[1]) if inp.dim() >= 3 else int(inp.shape[0])
        ids = (kwargs or {}).get("input_ids")
        recorded.append(
            {
                "target": type(module2inspect).__name__ if module2inspect is not None else None,
                "token_rows": token_rows,
                "ids": int(ids.numel()) if torch.is_tensor(ids) else None,
            }
        )
        return original(
            self,
            module,
            prev_op,
            layers,
            inp,
            module2inspect=module2inspect,
            kwargs=kwargs if kwargs is not None else {},
            **rest,
        )

    monkeypatch.setattr(AWQProcessor, "_search_best_scale", recorder)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(bits=4, group_size=32, format=FORMAT.GEMM, device="cpu"),
        dtype=torch.float16,
    )
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))

    moe_replays = [entry for entry in recorded if entry["target"] == "DeepseekV4SparseMoeBlock"]
    assert moe_replays, "the routed-expert gate/up group must replay the hash_moe block"
    for entry in moe_replays:
        assert entry["ids"] == entry["token_rows"], (
            "a token-aligned kwarg must describe every replayed token row: "
            f"{entry}"
        )
