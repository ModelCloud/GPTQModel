# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import threading

import pytest
import torch
from torch import nn

from gptqmodel.looper.exllamav3_processor import (
    EXL3Processor,
    _router_recovery_candidates,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.exl3_error_ledger import ROUTE_EVIDENCE_SCHEMA
from gptqmodel.utils.exl3_capture_batch_spool import CAPTURE_BATCH_SPOOL_ENV


class _Router(nn.Module):
    def forward(self, hidden_states):
        del hidden_states
        logits = torch.zeros((3, 3), dtype=torch.float32)
        weights = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]],
            dtype=torch.float32,
        )
        indices = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64)
        return logits, weights, indices


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _Router()


class _MemorySpool:
    phase = "gate-up"

    def __init__(self):
        self.committed_indices = set()
        self.records = {}

    def commit(self, batch_index, *, tensors, metadata):
        self.records[batch_index] = (tensors, metadata)
        self.committed_indices.add(batch_index)


def test_hash_router_emits_no_learned_recovery_candidates() -> None:
    router = nn.Module()
    router.register_buffer("tid2eid", torch.arange(256))
    logits = torch.randn(7, 256)
    indices = torch.randint(0, 256, (7, 6), dtype=torch.int64)

    candidate_indices, candidate_gaps = _router_recovery_candidates(
        router,
        logits,
        indices,
        candidate_rank_min=7,
        candidate_rank_max=12,
    )

    assert candidate_indices.shape == (7, 0)
    assert candidate_indices.dtype is torch.int64
    assert candidate_gaps.shape == (7, 0)
    assert candidate_gaps.dtype is torch.float32


def _processor_and_subset():
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {
                    "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
                }
            }
        }
    )
    processor._mask_tls = threading.local()
    processor._hooks_paused_tls = threading.local()
    processor.tasks = {}
    subset = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj"):
            task_name = f"mlp.experts.{expert}.{projection}"
            processor.tasks[task_name] = {}
            subset[task_name] = SimpleNamespace(full_name=f"model.layers.7.{task_name}")
    return processor, subset


def test_existing_subset_forward_captures_masked_route_count_and_gate_mass():
    processor, subset = _processor_and_subset()
    layer = _Layer()
    processor._mask_tls.value = torch.tensor([[True, False, True]])

    context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    )
    with context:
        layer.mlp.gate(torch.zeros((3, 4)))

    expert0 = processor.tasks["mlp.experts.0.gate_proj"]["route_evidence"]
    expert1 = processor.tasks["mlp.experts.1.up_proj"]["route_evidence"]
    assert expert0["router_calls"] == 1
    assert expert0["router_token_count"] == 2
    assert expert0["router_selected_route_count"] == 4
    assert expert0["router_top_k"] == 2
    assert expert0["expert_route_count"] == 2
    assert expert0["expert_gate_weight_sum"] == pytest.approx(1.5)
    assert expert0["expert_gate_squared_mass"] == pytest.approx(1.13)
    assert expert0["total_gate_weight_sum"] == pytest.approx(2.0)
    assert expert0["total_gate_squared_mass"] == pytest.approx(1.26)
    assert expert0["mask_modes"] == ["filtered"]
    assert processor.tasks["mlp.experts.0.up_proj"]["route_evidence"] == expert0

    assert expert1["expert_route_count"] == 1
    assert expert1["expert_gate_weight_sum"] == pytest.approx(0.2)
    assert expert1["expert_gate_squared_mass"] == pytest.approx(0.04)

    down_subset = {}
    for expert in range(2):
        task_name = f"mlp.experts.{expert}.down_proj"
        processor.tasks[task_name] = {}
        down_subset[task_name] = SimpleNamespace(
            full_name=f"model.layers.7.{task_name}"
        )
    cached_context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=down_subset,
    )
    with cached_context:
        pass
    assert processor.tasks["mlp.experts.0.down_proj"]["route_evidence"] == expert0
    assert not layer.mlp.gate._forward_hooks


def test_failed_subset_forward_removes_router_hook_without_committing_evidence():
    processor, subset = _processor_and_subset()
    layer = _Layer()
    context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        with context:
            layer.mlp.gate(torch.zeros((3, 4)))
            raise RuntimeError("forward failed")

    assert not layer.mlp.gate._forward_hooks
    assert all("route_evidence" not in task for task in processor.tasks.values())


def test_masked_capture_verifies_raw_topk_major_expert_fanout():
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {
                    "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
                }
            }
        }
    )
    processor._mask_tls = threading.local()
    processor._hooks_paused_tls = threading.local()
    processor._batch_tls = threading.local()
    processor._active_natural_route_capture = None
    processor._restored_route_accumulators = {}
    processor._natural_route_evidence_cache = {}
    processor._active_capture_batch_spool = _MemorySpool()
    processor._active_capture_batch_layer = 7
    processor.tasks = {}
    subset = {}
    for expert in range(3):
        task_name = f"mlp.experts.{expert}.gate_proj"
        processor.tasks[task_name] = {}
        subset[task_name] = SimpleNamespace(
            full_name=f"model.layers.7.{task_name}"
        )

    layer = _Layer()
    layer.mlp.gate.register_buffer("tid2eid", torch.arange(3))
    rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    processor._mask_tls.value = torch.tensor([[True, False, True]])
    processor._set_current_batch_index(0)
    with processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    ) as capture:
        _logits, _weights, indices = layer.mlp.gate(rows)
        for expert in range(3):
            # Match the reference fused-expert traversal: top-k slot first,
            # then token index. This intentionally includes the masked row.
            token_indices = torch.where(indices.transpose(0, 1).eq(expert))[1]
            capture.capture_expert_input(
                f"mlp.experts.{expert}.gate_proj",
                rows.index_select(0, token_indices),
            )
        capture.commit_batch(0)
    processor._set_current_batch_index(None)

    tensors, metadata = processor._active_capture_batch_spool.records[0]
    assert torch.equal(tensors["router_input"], rows[[0, 2]])
    assert tensors["top_indices"].shape == (2, 2)
    assert metadata["mask_mode"] == "filtered"
    assert metadata["pre_fanout_gate_input_verified"] is True


class _TwoExpertRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("tid2eid", torch.arange(2))

    def forward(self, hidden_states):
        logits = hidden_states[:, :2].float()
        indices = logits.argmax(dim=-1, keepdim=True)
        weights = torch.ones_like(indices, dtype=torch.float32)
        return logits, weights, indices


class _TwoExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _TwoExpertRouter()


def _recoverable_processor_and_subset():
    processor = EXL3Processor.__new__(EXL3Processor)
    family_join = {
        "source": {"revision": "test"},
        "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
    }
    processor.qcfg = SimpleNamespace(
        meta={"ds4rt_error_ledger": {"family_join": family_join}}
    )
    processor._mask_tls = threading.local()
    processor._hooks_paused_tls = threading.local()
    processor._batch_tls = threading.local()
    processor._active_capture_batch_spool = None
    processor._active_capture_batch_layer = None
    processor._active_natural_route_capture = None
    processor._restored_route_accumulators = {}
    processor._natural_route_evidence_cache = {}
    processor.tasks = {}
    subset = {}
    for expert in range(2):
        gate = GPTQ(nn.Linear(4, 2, bias=False))
        up = GPTQ(nn.Linear(4, 2, bias=False))
        gate.set_hessian_accumulator_device("cpu")
        up.share_hessian_state_from(gate)
        for projection, capture in (("gate_proj", gate), ("up_proj", up)):
            task_name = f"mlp.experts.{expert}.{projection}"
            processor.tasks[task_name] = {"capture": capture}
            subset[task_name] = SimpleNamespace(
                full_name=f"model.layers.7.{task_name}"
            )
    return processor, subset


def test_capture_batch_resume_rebuilds_shared_hessians_and_route_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(CAPTURE_BATCH_SPOOL_ENV, str(tmp_path))
    processor, subset = _recoverable_processor_and_subset()
    layer = _TwoExpertLayer()
    committed = processor.restore_subset_capture_batches(
        layer_index=7,
        subset_index=0,
        subset_total=2,
        expected_batches=2,
        subset=subset,
    )
    assert committed == frozenset()

    rows = torch.tensor(
        [[4.0, 1.0, 2.0, 3.0], [1.0, 5.0, 6.0, 7.0], [3.0, 2.0, 8.0, 9.0]]
    )
    processor._set_current_batch_index(0)
    with processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    ):
        _logits, _weights, indices = layer.mlp.gate(rows)
        for expert in range(2):
            expert_rows = rows[indices[:, 0] == expert]
            for projection in ("gate_proj", "up_proj"):
                task_name = f"mlp.experts.{expert}.{projection}"
                processor.pre_process_fwd_hook(task_name)(
                    None, (expert_rows,), torch.empty(0)
                )
        processor.forward_batch_completed(layer_index=7, batch_index=0)
    processor._set_current_batch_index(None)
    tensors, _metadata = processor._active_capture_batch_spool.load(0)
    assert tensors["candidate_indices"].shape == (rows.shape[0], 0)
    assert tensors["candidate_score_gaps"].shape == (rows.shape[0], 0)

    restored, restored_subset = _recoverable_processor_and_subset()
    restored_indices = restored.restore_subset_capture_batches(
        layer_index=7,
        subset_index=0,
        subset_total=2,
        expected_batches=2,
        subset=restored_subset,
    )
    assert restored_indices == frozenset({0})
    for expert in range(2):
        expected_rows = rows[indices[:, 0] == expert]
        gate = restored.tasks[f"mlp.experts.{expert}.gate_proj"]["capture"]
        up = restored.tasks[f"mlp.experts.{expert}.up_proj"]["capture"]
        assert gate.nsamples == up.nsamples == expected_rows.shape[0]
        assert gate._hessian_state is up._hessian_state
        assert torch.equal(
            gate.snapshot_hessian(torch.device("cpu")),
            (expected_rows.T @ expected_rows)
            * (2.0 / expected_rows.shape[0]),
        )
    route_state = restored._restored_route_accumulators[("base", 7)]
    assert route_state["router_calls"] == 1
    assert route_state["router_token_count"] == 3
    assert route_state["expert_counts"].tolist() == [2, 1]
