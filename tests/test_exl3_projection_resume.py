# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file as save_safetensors_file

import gptqmodel.looper.exllamav3_processor as processor_module
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils.exl3_error_ledger import append_exl3_error_journal
from gptqmodel.utils.exl3_capture_frontier import EXL3CaptureRecord
from gptqmodel.utils.exl3_projection_checkpoint import (
    CHECKPOINT_CONTRACT,
    EXL3ProjectionCheckpointStore,
    build_projection_request,
)
from gptqmodel.utils.exl3_inline_mixed import (
    INLINE_MIXED_META_KEY,
    InlineMixedPolicy,
    InlineMixedTierPlanStore,
    build_layer_tier_plan,
)


class _Capture:
    def __init__(self, module: NamedModule, hessian: torch.Tensor) -> None:
        self.module = module
        self.H = hessian
        self.nsamples = 1024
        self._device_hessian_partials = {}
        self._device_sample_counts = {}
        self._hessian_dirty = False
        self._final_hessian_device_hint = None
        self.clone_devices = []

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(target_device)
        return self.H

    def clone_module(self, copy=True, device=None):
        self.clone_devices.append(torch.device(device))
        return self.module.module.weight.detach().to(device=device, copy=copy).float()

    def free(self):
        self.H = None


def test_base_inline_policy_leaves_mtp_projection_uniform_k3(tmp_path) -> None:
    policy = InlineMixedPolicy(
        namespace="base",
        base_bits=3,
        upgrade_bits=4,
        extra_bits=Fraction(1, 4),
        projection_ratio=(3, 5, 8),
        tier_plan_root=tmp_path / "tier-plans",
        logical_layer_start=3,
        logical_layer_count=42,
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            INLINE_MIXED_META_KEY: {
                **policy.policy_body,
                "tier_plan_root": str(policy.tier_plan_root),
            }
        }
    )
    processor._remote_client_for_run = lambda _provenance: None
    observed = {}

    def observe_process(**kwargs):
        observed.update(kwargs)
        return "uniform-mtp"

    processor._process_on_slot = observe_process
    module = NamedModule(
        nn.Linear(2, 2, bias=False),
        "mlp.experts.0.gate_proj",
        "mtp.0.mlp.experts.0.gate_proj",
        0,
    )

    assert processor.process(module) == "uniform-mtp"
    assert observed["bits_override"] is None
    assert observed["inline_context"] is None
    assert observed["defer_stats"] is False


def _processor_and_module(
    root: Path,
    weight: torch.Tensor,
    hessian: torch.Tensor,
) -> tuple[EXL3Processor, NamedModule]:
    linear = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="cuda:0")
    linear.weight.data.copy_(weight)
    module = NamedModule(
        linear,
        "mlp.experts.31.gate_proj",
        "model.layers.7.mlp.experts.31.gate_proj",
        7,
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {"source_revision": "test-source"},
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(root),
                    }
                },
            }
        },
        dynamic=None,
    )
    processor.tasks = {
        module.name: {
            "capture": _Capture(module, hessian.clone()),
            "qcfg": SimpleNamespace(
                head_bits=None,
                runtime_bits=2,
                out_scales="auto",
                codebook="mcg",
            ),
        }
    }
    processor.lm_head_name = "lm_head"
    processor.error_journal_path = str(root.parent / "error-journal.jsonl")
    processor._stats_lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []
    processor.draw_progress = lambda *args, **kwargs: None
    processor.formatted_fwd_time = lambda: "0.000"
    processor.device_memory_report = lambda: "test"
    processor.log_new_row = lambda *args, **kwargs: None
    return processor, module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_process_resumes_from_packed_checkpoint_without_requantizing(
    tmp_path,
    monkeypatch,
) -> None:
    torch.manual_seed(787)
    weight = (torch.randn((128, 128), dtype=torch.float32, device="cuda:0") * 0.02).to(
        torch.bfloat16
    )
    activations = torch.randn((1024, 128), dtype=torch.float32, device="cuda:0")
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    checkpoint_root = tmp_path / "projection-checkpoints"

    first_processor, first_module = _processor_and_module(
        checkpoint_root,
        weight,
        hessian,
    )
    original_quantize_exl3 = processor_module.quantize_exl3
    quant_lock_observations = []

    def observe_device_trellis_lock(*args, **kwargs):
        lock = first_processor._distributed_local_quant_lock(torch.device("cuda:0"))
        quant_lock_observations.append(lock.locked())
        return original_quantize_exl3(*args, **kwargs)

    monkeypatch.setattr(
        processor_module,
        "quantize_exl3",
        observe_device_trellis_lock,
    )
    first_processor.process(first_module)
    assert first_module.module.weight.is_meta
    first_processor.prepare_runtime_weight_for_forward(
        module=first_module,
        target_device=torch.device("cuda:0"),
    )
    assert quant_lock_observations == [True]
    first_replay_weight = first_module.module.weight.detach().cpu().clone()
    assert first_processor.log[-1]["exl3_projection_checkpoint_hit"] is False
    assert len(list(checkpoint_root.rglob("*.json"))) == 1
    assert len(list(checkpoint_root.rglob("*.safetensors"))) == 1

    second_processor, second_module = _processor_and_module(
        checkpoint_root,
        weight,
        hessian,
    )
    frontier_root = tmp_path / "capture-frontier"
    frontier_root.mkdir()
    frontier_payload = frontier_root / "expert-31-gate.safetensors"
    save_safetensors_file({"H": hessian.cpu()}, frontier_payload)
    second_capture = second_processor.tasks[second_module.name]["capture"]
    second_capture.H = None
    second_processor.tasks[second_module.name]["capture_frontier_record"] = (
        EXL3CaptureRecord(
            module=second_module.full_name,
            path=frontier_payload,
            payload={
                "tensor": {
                    "dtype": "torch.float32",
                    "shape": [128, 128],
                    "bytes": 128 * 128 * 4,
                }
            },
            sample_count=1024,
            route_evidence=None,
        )
    )
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(frontier_root))

    def fail_requantization(*args, **kwargs):
        raise AssertionError("checkpoint hit attempted to run trellis quantization")

    monkeypatch.setattr(processor_module, "quantize_exl3", fail_requantization)
    second_processor.process(second_module)
    assert second_module.module.weight.is_meta
    second_processor.prepare_runtime_weight_for_forward(
        module=second_module,
        target_device=torch.device("cuda:0"),
    )
    assert second_capture.clone_devices == [torch.device("cpu")]
    assert second_processor.log[-1]["exl3_projection_checkpoint_hit"] is True
    assert torch.equal(
        second_module.module.weight.detach().cpu().view(torch.int16),
        first_replay_weight.view(torch.int16),
    )


def test_capture_memory_summary_attributes_cache_model_and_heap_release(
    monkeypatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    shared = torch.zeros(8, dtype=torch.float32)
    position_ids = torch.zeros(3, dtype=torch.int64)
    processor.inputs_cache = SimpleNamespace(
        layer_inputs=[[shared]],
        layer_input_kwargs=[{"alias": shared[:4]}],
        position_ids=[position_ids],
        attention_masks=[],
    )
    processor.tasks = {}

    model_root = nn.Module()
    model_root.register_parameter(
        "weight", nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))
    )
    model_root.register_buffer("deferred", torch.empty(7, device="meta"))
    model = SimpleNamespace(model=model_root)

    summary = processor.capture_memory_summary(model=model)
    assert summary["process_rss_bytes"] > 0
    assert summary["input_cache_tensor_count"] == 3
    assert summary["input_cache_storage_count"] == 2
    assert summary["input_cache_host_bytes"] == (
        shared.untyped_storage().nbytes()
        + position_ids.untyped_storage().nbytes()
    )
    assert summary["model_tensor_count"] == 2
    assert summary["model_storage_count"] == 1
    assert summary["model_meta_tensor_count"] == 1
    assert summary["model_host_bytes"] == 5 * torch.bfloat16.itemsize

    monkeypatch.setattr(processor_module.gc, "collect", lambda: 7)
    monkeypatch.setattr(
        EXL3Processor,
        "_malloc_trim",
        staticmethod(lambda: 1),
    )
    released = processor.release_host_memory("test", model=model)
    assert released["gc_collected"] == 7
    assert released["malloc_trim_result"] == 1
    assert released["before"]["input_cache_host_bytes"] == summary[
        "input_cache_host_bytes"
    ]
    assert released["after"]["model_host_bytes"] == summary["model_host_bytes"]


def test_capture_memory_limit_trims_only_unallocated_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    before = {
        "process_rss_bytes": 100,
        "cuda_devices": {
            "cuda:0": {"allocated_bytes": 70, "reserved_bytes": 95},
            "cuda:1": {"allocated_bytes": 75, "reserved_bytes": 80},
        },
    }
    after = {
        "process_rss_bytes": 100,
        "cuda_devices": {
            "cuda:0": {"allocated_bytes": 70, "reserved_bytes": 93},
            "cuda:1": {"allocated_bytes": 75, "reserved_bytes": 77},
        },
    }
    summaries = iter((before, after))
    contexts: list[str] = []
    trims: list[bool] = []

    def log_summary(context: str):
        contexts.append(context)
        return next(summaries)

    processor.log_capture_memory_summary = log_summary
    monkeypatch.setenv(processor_module.HOST_RSS_LIMIT_ENV, "200")
    monkeypatch.setenv(processor_module.CUDA_ALLOCATION_LIMIT_ENV, "90")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: trims.append(True))

    result = processor._enforce_capture_memory_limits(context="batch-64")

    assert result is after
    assert trims == [True]
    assert contexts == ["batch-64", "batch-64-after-cache-trim"]


def test_completed_layer_deferral_replaces_packed_storage_with_meta_shell() -> None:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    root.model.layers[0].proj = ExllamaV3Linear.from_tensors(
        in_features=4,
        out_features=8,
        name="model.layers.0.proj",
        tensors={
            "trellis": torch.zeros((8, 1, 8), dtype=torch.int16),
            "suh": torch.ones(4, dtype=torch.float16),
            "svh": torch.ones(8, dtype=torch.float16),
            "mcg": torch.tensor([123], dtype=torch.int32),
        },
    )
    model = SimpleNamespace(model=root)
    processor = EXL3Processor.__new__(EXL3Processor)
    processor._stats_lock = threading.Lock()
    processor.log = [
        {
            "layer": 0,
            "exl3_error_ledger_record": {"module": "model.layers.0.proj"},
            "exl3_projection_checkpoint": "request",
            "exl3_error_record_sha256": "record",
        }
    ]
    entries = [
        {
            "module": "model.layers.0.proj",
            "request_sha256": "request",
            "record_sha256": "record",
        }
    ]

    before = processor._model_tensor_summary(model)
    processor.defer_completed_layer_checkpoints(
        model=model,
        layer_index=0,
        projection_entries=entries,
    )
    deferred = root.get_submodule("model.layers.0.proj")
    after = processor._model_tensor_summary(model)

    assert isinstance(deferred, ExllamaV3Linear)
    assert deferred.trellis.device.type == "meta"
    assert before["host_bytes"] > 0
    assert after["host_bytes"] == 0
    assert after["meta_tensor_count"] > 0


def test_completed_layer_deferral_normalizes_disk_offloaded_meta_shell() -> None:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    materialized = ExllamaV3Linear.from_tensors(
        in_features=4,
        out_features=8,
        name="model.layers.0.proj",
        tensors={
            "trellis": torch.zeros((8, 1, 8), dtype=torch.int16),
            "suh": torch.ones(4, dtype=torch.float16),
            "svh": torch.ones(8, dtype=torch.float16),
            "mcg": torch.tensor([123], dtype=torch.int32),
        },
    )
    root.model.layers[0].proj = ExllamaV3Linear(
        in_features=materialized.in_features,
        out_features=materialized.out_features,
        name=materialized.name,
        tensor_storage=materialized.tensor_storage_entry(),
        out_dtype=materialized.out_dtype,
    )
    model = SimpleNamespace(model=root)
    processor = EXL3Processor.__new__(EXL3Processor)
    processor._stats_lock = threading.Lock()
    processor.log = [
        {
            "layer": 0,
            "exl3_error_ledger_record": {"module": "model.layers.0.proj"},
            "exl3_projection_checkpoint": "request",
            "exl3_error_record_sha256": "record",
        }
    ]
    entries = [
        {
            "module": "model.layers.0.proj",
            "request_sha256": "request",
            "record_sha256": "record",
        }
    ]

    processor.defer_completed_layer_checkpoints(
        model=model,
        layer_index=0,
        projection_entries=entries,
    )

    deferred = root.get_submodule("model.layers.0.proj")
    assert isinstance(deferred, ExllamaV3Linear)
    assert deferred.trellis.device.type == "meta"
    assert deferred.tensor_storage_entry() == materialized.tensor_storage_entry()


def test_runtime_reconstruction_is_deferred_and_shares_the_device_trellis_lock(
    monkeypatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    processor._stats_lock = threading.Lock()
    processor._distributed_local_quant_locks = {}
    linear = nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    module = NamedModule(
        linear,
        "mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        0,
    )
    module.state["quant_source_module"] = nn.Linear(
        2, 3, bias=False, dtype=torch.bfloat16
    )
    device = torch.device("cpu")
    observed = []

    def reconstruct(_out_tensors, *, device, dtype):
        lock = processor._distributed_local_quant_lock(torch.device(device))
        observed.append((lock.locked(), dtype))
        return torch.arange(6, dtype=torch.float32).reshape(2, 3)

    monkeypatch.setattr(processor_module, "reconstruct_exl3_tensors", reconstruct)
    packed = {
        "trellis": torch.zeros(1),
        "suh": torch.zeros(1),
        "svh": torch.zeros(1),
    }
    module.state.update(packed)
    processor._stage_runtime_weight(
        module=module,
        out_tensors=packed,
        target_device=device,
    )

    assert observed == []
    assert module.module.weight.is_meta
    processor.prepare_runtime_weight_for_forward(
        module=module,
        target_device=device,
    )
    assert observed == [(True, torch.bfloat16)]
    assert torch.equal(
        module.module.weight.detach(),
        torch.arange(6, dtype=torch.float32).reshape(2, 3).T.to(torch.bfloat16),
    )
    assert module.module.weight.device.type == "cpu"
    assert "quant_source_module" not in module.state


def test_terminal_layer_finalization_does_not_reconstruct_deferred_weight(
    monkeypatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    processor._stats_lock = threading.Lock()
    linear = nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    module = NamedModule(
        linear,
        "mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        0,
    )
    packed = {
        "trellis": torch.zeros(1),
        "suh": torch.zeros(1),
        "svh": torch.zeros(1),
    }
    module.state.update(packed)
    processor._stage_runtime_weight(
        module=module,
        out_tensors=packed,
        target_device=torch.device("cpu"),
    )

    assert module.module.weight.is_meta
    processor.prepare_layer_post_quantize(
        model=SimpleNamespace(),
        layer_module=nn.Module(),
        layer_index=0,
        processed_modules={module.name: module},
        is_lm_head_module=False,
    )

    assert module.module.weight.device.type == "cpu"
    assert module.module.weight.numel() == 0
    assert "exl3_deferred_runtime_weight" not in module.state
    assert module.state["exl3_deferred_finalize_weight"] == {
        "dtype": "torch.bfloat16",
        "shape": [3, 2],
        "requires_grad": True,
    }

    finalized = []

    def create_packed(**kwargs):
        finalized.append(kwargs)
        return nn.Identity()

    monkeypatch.setattr(processor_module, "create_exllamav3_module", create_packed)
    processor.submodule_finalize(module, SimpleNamespace(model=nn.Module()))

    assert len(finalized) == 1
    assert set(finalized[0]["tensors"]) == set(packed)
    assert "exl3_deferred_finalize_weight" not in module.state
    assert not hasattr(module.module, "weight")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_runtime_reconstruction_retains_no_dense_weight_until_forward(
    monkeypatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    processor._stats_lock = threading.Lock()
    processor._distributed_local_quant_locks = {}
    device = torch.device("cuda:0")
    module = NamedModule(
        nn.Linear(2, 3, bias=False, dtype=torch.bfloat16, device=device),
        "mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        0,
    )
    module.state["quant_source_module"] = nn.Linear(
        2, 3, bias=False, dtype=torch.bfloat16
    )

    def reconstruct(_out_tensors, *, device, dtype):
        assert processor._distributed_local_quant_lock(
            torch.device(device)
        ).locked()
        assert dtype == torch.bfloat16
        return torch.arange(
            6, dtype=torch.float32, device=device
        ).reshape(2, 3)

    monkeypatch.setattr(processor_module, "reconstruct_exl3_tensors", reconstruct)
    packed = {
        "trellis": torch.zeros(1),
        "suh": torch.zeros(1),
        "svh": torch.zeros(1),
    }
    module.state.update(packed)
    processor._stage_runtime_weight(
        module=module,
        out_tensors=packed,
        target_device=device,
    )

    assert module.module.weight.is_meta
    assert "quant_source_module" not in module.state
    processor.prepare_runtime_weight_for_forward(
        module=module,
        target_device=device,
    )

    assert module.module.weight.device == device
    assert torch.equal(
        module.module.weight.detach(),
        torch.arange(6, dtype=torch.float32, device=device)
        .reshape(2, 3)
        .T.to(torch.bfloat16),
    )
    assert "exl3_deferred_runtime_weight" not in module.state


def test_restore_completed_layer_installs_packed_modules_without_hessian(
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "projection-checkpoints"
    offload_root = tmp_path / "offload"
    journal = tmp_path / "error-journal.jsonl"
    family_join = {"source_revision": "test-source", "corpus": "test-corpus"}
    store = EXL3ProjectionCheckpointStore(checkpoint_root)

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)

    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    root.model.layers[0].mlp = nn.Module()
    root.model.layers[0].mlp.experts = nn.ModuleList([Expert(), Expert()])
    model = SimpleNamespace(model=root)

    entries = []
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module_name = f"model.layers.0.mlp.experts.{expert}.{projection}"
            request = build_projection_request(
                module_full_name=module_name,
                layer_index=0,
                input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                hessian=torch.eye(4, dtype=torch.float32),
                sample_count=32,
                quantizer_contract={"bits": 2, "codebook": "mcg"},
                family_join=family_join,
                route_evidence=None,
            )
            ledger_record = {
                "schema": "ds4rt.exl3-error-ledger",
                "schema_version": 1,
                "record_kind": "projection",
                "module": module_name,
                "processor_layer_index": 0,
                "provenance": {"family_join": family_join},
            }
            record_sha256 = append_exl3_error_journal(journal, ledger_record)
            store.commit(
                request,
                {
                    "trellis": torch.arange(4096, dtype=torch.int16).reshape(1, 1, 4096),
                    "suh": torch.ones(4, dtype=torch.float16),
                    "svh": torch.ones(4, dtype=torch.float16),
                    "mcg": torch.tensor([123], dtype=torch.int32),
                },
                {
                    "duration_seconds": 1.0,
                    "proxy_error": 0.1,
                    "device_names": ["cuda:0"],
                    "quantizer_metrics": {"reported_metric_kind": "test"},
                    "ledger_record": ledger_record,
                    "execution_contract": None,
                    "execution_result": {"kind": "test"},
                },
            )
            entries.append(
                {
                    "module": module_name,
                    "request_sha256": request["request_sha256"],
                    "record_sha256": record_sha256,
                }
            )

    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": family_join,
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(checkpoint_root),
                    }
                },
            }
        },
        offload_to_disk=True,
        offload_to_disk_path=str(offload_root),
    )
    processor.error_journal_path = str(journal)
    processor._stats_lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []

    processor.restore_completed_layer_checkpoints(
        model=model,
        layer_index=0,
        projection_entries=entries,
    )

    restored = [
        root.get_submodule(entry["module"])
        for entry in entries
    ]
    assert all(isinstance(module, ExllamaV3Linear) for module in restored)
    assert all(module.trellis.device.type == "meta" for module in restored)
    assert all(not hasattr(module, "_hf_hook") for module in restored)
    assert len(processor.log) == 6
    assert all(stat["exl3_layer_boundary_restore"] for stat in processor.log)
    assert len(list(offload_root.rglob("index.json"))) == 6
    assert not list(offload_root.rglob("module.safetensors"))
    for index_path in offload_root.rglob("index.json"):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        assert all(
            Path(entry["safetensors_file"]).is_relative_to(checkpoint_root)
            for entry in index.values()
        )

    # A metadata-only EXL3 shell is a deferred target, not a live duplicate.
    # Publication may materialize it again from the same checkpoints.
    processor.restore_completed_layer_checkpoints(
        model=model,
        layer_index=0,
        projection_entries=entries,
    )
    assert all(
        isinstance(root.get_submodule(entry["module"]), ExllamaV3Linear)
        for entry in entries
    )
    assert len(processor.log) == 6
    assert len(processor.module_names) == 6
    assert len(processor.avg_losses) == 6
    assert len(processor.durations) == 6

    processor.log[0]["exl3_projection_checkpoint"] = "0" * 64
    with pytest.raises(RuntimeError, match="conflicting projection history"):
        processor.restore_completed_layer_checkpoints(
            model=model,
            layer_index=0,
            projection_entries=entries,
        )


def test_restore_completed_layer_rejects_duplicate_existing_history(
    tmp_path,
) -> None:
    # Duplicate history is rejected before checkpoint or model mutation, so an
    # empty restore request is sufficient to exercise the preflight guard.
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {},
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(tmp_path / "projection-checkpoints"),
                    }
                },
            }
        },
        offload_to_disk=True,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    processor.error_journal_path = str(tmp_path / "error-journal.jsonl")
    processor._stats_lock = threading.Lock()
    stat = {
        "layer": 0,
        "module": "gate_proj",
        "exl3_error_ledger_record": {"module": "model.layers.0.gate_proj"},
        "exl3_projection_checkpoint": "1" * 64,
        "exl3_error_record_sha256": "2" * 64,
    }
    processor.log = [stat, dict(stat)]
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []

    with pytest.raises(RuntimeError, match="duplicate projection history"):
        processor.restore_completed_layer_checkpoints(
            model=SimpleNamespace(model=nn.Module()),
            layer_index=0,
            projection_entries=[],
        )


def _inline_mtp_restore_fixture(
    tmp_path: Path,
    *,
    omit_module: str | None = None,
):
    checkpoint_root = tmp_path / "projection-checkpoints"
    offload_root = tmp_path / "offload"
    journal = tmp_path / "error-journal.jsonl"
    tier_root = tmp_path / "tier-plans"
    family_join = {"source_revision": "test-source", "corpus": "test-corpus"}
    policy = InlineMixedPolicy(
        namespace="mtp",
        base_bits=2,
        upgrade_bits=3,
        extra_bits=Fraction(1, 6),
        projection_ratio=(1, 1, 1),
        tier_plan_root=tier_root,
    )
    store = EXL3ProjectionCheckpointStore(checkpoint_root)

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)

    root = nn.Module()
    root.mtp = nn.ModuleList([nn.Module()])
    root.mtp[0].mlp = nn.Module()
    root.mtp[0].mlp.experts = nn.ModuleList([Expert(), Expert()])
    projection_names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    candidate_records = []
    candidates = {}
    for expert in range(2):
        for projection in ("w1", "w3", "w2"):
            module = f"mtp.0.mlp.experts.{expert}.{projection_names[projection]}"
            record = {
                "schema": "ds4rt.exl3-error-ledger",
                "schema_version": 1,
                "record_kind": "projection",
                "module": module,
                "processor_layer_index": 0,
                "block_namespace": "mtp",
                "logical_layer": 0,
                "expert": expert,
                "projection": projection,
                "bits": 2,
                "codebook": "mcg",
                "sample_count": 32,
                "duration_seconds": 1.0,
                "encoded_bytes": 1,
                "quantizer_metrics": {
                    "hessian_weighted_relative_error": float(expert + 1)
                },
                "route_evidence": {
                    "expert_gate_squared_mass_fraction": 0.25
                },
                "provenance": {"family_join": family_join},
            }
            candidate_records.append(record)
            request = build_projection_request(
                module_full_name=module,
                layer_index=0,
                input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                hessian=torch.eye(4, dtype=torch.float32),
                sample_count=32,
                quantizer_contract={
                    "bits": 2,
                    "codebook": "mcg",
                    "inline_mixed": {
                        "base_bits": 2,
                        "upgrade_bits": 3,
                        "policy_sha256": policy.policy_sha256,
                        "role": "candidate_k2",
                    },
                },
                family_join=family_join,
                route_evidence=None,
            )
            candidates[module] = (request, record)
            if module != omit_module:
                store.commit(
                    request,
                    {
                        "trellis": torch.full((1, 1, 64), 2, dtype=torch.int16),
                        "suh": torch.ones(4, dtype=torch.float16),
                        "svh": torch.ones(4, dtype=torch.float16),
                        "mcg": torch.tensor([123], dtype=torch.int32),
                    },
                    {
                        "duration_seconds": 1.0,
                        "proxy_error": 0.1,
                        "device_names": ["cuda:0"],
                        "quantizer_metrics": {
                            "reported_metric_kind": "test",
                        },
                        "ledger_record": record,
                        "execution_contract": None,
                        "execution_result": {"kind": "test"},
                    },
                )

    plan = build_layer_tier_plan(
        policy=policy,
        layer_index=0,
        layer_count=1,
        candidate_records=candidate_records,
    )
    InlineMixedTierPlanStore(policy).commit(plan)
    selected_module = plan["selected"][0]["module"]
    candidate_request, candidate_record = candidates[selected_module]
    selected_record = {**candidate_record, "bits": 3, "encoded_bytes": 2}
    selected_request = build_projection_request(
        module_full_name=selected_module,
        layer_index=0,
        input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        hessian=torch.eye(4, dtype=torch.float32),
        sample_count=32,
        quantizer_contract={
            "bits": 3,
            "codebook": "mcg",
            "inline_mixed": {
                "base_bits": 2,
                "upgrade_bits": 3,
                "policy_sha256": policy.policy_sha256,
                "role": "selected_k3",
                "candidate_request_sha256": candidate_request["request_sha256"],
                "tier_plan_sha256": plan["tier_plan_sha256"],
            },
        },
        family_join=family_join,
        route_evidence=None,
    )
    if selected_module != omit_module:
        store.commit(
            selected_request,
            {
                "trellis": torch.full((1, 1, 96), 3, dtype=torch.int16),
                "suh": torch.ones(4, dtype=torch.float16),
                "svh": torch.ones(4, dtype=torch.float16),
                "mcg": torch.tensor([123], dtype=torch.int32),
            },
            {
                "duration_seconds": 1.0,
                "proxy_error": 0.05,
                "device_names": ["cuda:0"],
                "quantizer_metrics": {"reported_metric_kind": "test"},
                "ledger_record": selected_record,
                "execution_contract": None,
                "execution_result": {"kind": "test"},
            },
        )

    qcfg = SimpleNamespace(
        bits=2,
        meta={
            INLINE_MIXED_META_KEY: {
                **policy.policy_body,
                "tier_plan_root": str(tier_root),
            },
            "ds4rt_error_ledger": {
                "family_join": family_join,
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(checkpoint_root),
                    }
                },
            },
        },
        offload_to_disk=True,
        offload_to_disk_path=str(offload_root),
        method=None,
        format=None,
    )
    model = SimpleNamespace(
        model=root,
        quantize_config=qcfg,
        quantized=False,
        quant_log=[],
        qlinear_kernel=None,
    )
    return model, policy, journal, selected_module


def test_complete_inline_mixed_mtp_tree_restores_without_replay(tmp_path) -> None:
    model, _policy, journal, selected_module = _inline_mtp_restore_fixture(tmp_path)
    assert EXL3Processor.restore_completed_checkpoint_tree_if_complete(
        model=model,
        block_namespace="mtp",
        layer_count=1,
        experts_per_layer=2,
        error_journal_path=journal,
    )
    assert model.quantized
    assert len(model.quant_log) == 6
    assert all(entry["exl3_layer_boundary_restore"] for entry in model.quant_log)
    assert model.model.get_submodule(selected_module).trellis.shape[-1] == 96
    assert all(
        isinstance(module, ExllamaV3Linear)
        for name, module in model.model.named_modules()
        if name.endswith(("gate_proj", "up_proj", "down_proj"))
    )


def test_partial_inline_mixed_mtp_tree_does_not_mutate_model(tmp_path) -> None:
    omitted = "mtp.0.mlp.experts.0.down_proj"
    model, _policy, journal, _selected = _inline_mtp_restore_fixture(
        tmp_path,
        omit_module=omitted,
    )
    assert not EXL3Processor.restore_completed_checkpoint_tree_if_complete(
        model=model,
        block_namespace="mtp",
        layer_count=1,
        experts_per_layer=2,
        error_journal_path=journal,
    )
    assert not model.quantized
    assert model.quant_log == []
    assert isinstance(model.model.get_submodule(omitted), nn.Linear)


def test_corrupt_inline_mixed_tier_plan_is_not_replayed(tmp_path) -> None:
    model, policy, journal, _selected = _inline_mtp_restore_fixture(tmp_path)
    path = InlineMixedTierPlanStore(policy).path(0)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["selected"][0]["score"] += 1.0
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="tier plan failed validation"):
        EXL3Processor.restore_completed_checkpoint_tree_if_complete(
            model=model,
            block_namespace="mtp",
            layer_count=1,
            experts_per_layer=2,
            error_journal_path=journal,
        )
    assert not model.quantized
