#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Reproduce issue #2326 with a minimal BALANCED multi-GPU forward leak."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import types
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("GPTQMODEL_DEVICE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.loop_processor import ExecutionConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import build_subset_plan
from gptqmodel.nn_modules.hooked_linear import replace_module_with_hooked_legacy
from gptqmodel.quantization.config import VramStrategy
from gptqmodel.utils.device_telemetry import (
    clear_device_telemetry_records,
    get_device_telemetry_records,
)


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)
        for mod in (self.gate_proj, self.up_proj, self.down_proj):
            torch.nn.init.eye_(mod.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(hidden_states) + self.up_proj(hidden_states))


class _SelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            torch.nn.init.eye_(mod.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.o_proj(
            self.q_proj(hidden_states) + self.k_proj(hidden_states) + self.v_proj(hidden_states)
        )


class _ToyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.ModuleList([_Expert(), _Expert()])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        **_kwargs,
    ) -> torch.Tensor:
        hidden_states = self.self_attn(hidden_states)
        output = 0
        for expert in self.mlp.experts:
            output = output + expert(hidden_states)
        return output


class _PlanLooper:
    def __init__(self, devices: list[torch.device]) -> None:
        self._quant_devices = devices
        self._dense_quant_devices = [devices[0]]
        self._moe_quant_devices = devices
        self._dense_vram_strategy = VramStrategy.EXCLUSIVE
        self._moe_vram_strategy = VramStrategy.BALANCED
        self._dense_vram_strategy_explicit = False
        self._moe_vram_strategy_explicit = True
        self._moe_subset_threshold = 2
        self.gptq_model = types.SimpleNamespace(
            lm_head=None,
            quantize_config=types.SimpleNamespace(
                auto_forward_data_parallel=True,
                moe=None,
            ),
        )

    @staticmethod
    def _is_attention_module_name(name: str) -> bool:
        return name.startswith("self_attn.")

    @staticmethod
    def _extract_moe_group_key(name: str) -> str | None:
        parts = name.split(".")
        if "experts" not in parts:
            return None
        idx = parts.index("experts")
        return ".".join(parts[: idx + 2])

    @staticmethod
    def _resolve_batch_total(_num_batches, layer_inputs) -> int:
        return len(layer_inputs)

    @staticmethod
    def _collect_row_counts(layer_inputs) -> list[int]:
        return [int(batch[0].shape[0]) for batch in layer_inputs]


class _DummyGptqModel:
    def __init__(self) -> None:
        self.quantize_config = types.SimpleNamespace(
            auto_forward_data_parallel=True,
            calibration_data_device=None,
        )

    @staticmethod
    def shell_module_materialize(target_submodule, device, role, named_module=None):
        return target_submodule

    @staticmethod
    def prepare_layer_replay_kwargs(layer, layer_input, additional_inputs, target_device):
        return additional_inputs


class _ExecLooper:
    support_batch_quantize = False
    moe_routing_override = None
    moe_routing_bypass = False
    _current_subset = None
    MoERoutingOverrideContext = staticmethod(lambda *args, **kwargs: nullcontext())
    MoELifecycleContext = staticmethod(lambda *args, **kwargs: nullcontext())
    _assign_quant_device_for_module = ModuleLooper._assign_quant_device_for_module
    _rehome_processor_task = ModuleLooper._rehome_processor_task
    _prepare_named_module_for_quantization = ModuleLooper._prepare_named_module_for_quantization
    _apply_forward_device_overrides = ModuleLooper._apply_forward_device_overrides
    _restore_forward_device_overrides = ModuleLooper._restore_forward_device_overrides

    def __init__(self, devices: list[torch.device]) -> None:
        self.gptq_model = _DummyGptqModel()
        self._quant_devices = devices
        self._module_device_map = {}
        self._quant_device_lock = threading.Lock()
        self._quant_device_rr = 0

    @staticmethod
    def _resolve_batch_total(_num_batches, layer_inputs) -> int:
        return len(layer_inputs)

    @staticmethod
    def _collect_row_counts(layer_inputs) -> list[int]:
        return [int(batch[0].shape[0]) for batch in layer_inputs]

    @staticmethod
    def _batch_row_count(batch_inputs) -> int:
        return int(batch_inputs[0].shape[0])

    @staticmethod
    def _set_processor_mask(processor, mask) -> None:
        return None


def _nvidia_smi_snapshot() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,name",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception as exc:
        return [f"nvidia-smi unavailable: {exc}"]
    return [line.strip() for line in output.splitlines() if line.strip()]


def _named_modules_for_layer(layer: _ToyLayer) -> dict[str, NamedModule]:
    module_refs = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.o_proj": layer.self_attn.o_proj,
        "mlp.experts.0.gate_proj": layer.mlp.experts[0].gate_proj,
        "mlp.experts.0.up_proj": layer.mlp.experts[0].up_proj,
        "mlp.experts.0.down_proj": layer.mlp.experts[0].down_proj,
        "mlp.experts.1.gate_proj": layer.mlp.experts[1].gate_proj,
        "mlp.experts.1.up_proj": layer.mlp.experts[1].up_proj,
        "mlp.experts.1.down_proj": layer.mlp.experts[1].down_proj,
    }
    return {
        name: NamedModule(
            mod,
            name=name,
            full_name=f"model.layers.0.{name}",
            layer_index=0,
        )
        for name, mod in module_refs.items()
    }


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Requires at least 2 visible CUDA devices.", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        print("Set CUDA_DEVICE_ORDER=PCI_BUS_ID before running.", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        print("Set CUDA_VISIBLE_DEVICES=0,1 before running.", file=sys.stderr)
        return 2
    if getattr(sys, "_is_gil_enabled", lambda: True)():
        print("Run with PYTHON_GIL=0 so the free-threaded build keeps the GIL disabled.", file=sys.stderr)
        return 2

    clear_device_telemetry_records()

    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    layer = _ToyLayer()
    replace_module_with_hooked_legacy(layer)
    layer = layer.to(devices[0])
    layer.target_device = devices[0]

    full_named = _named_modules_for_layer(layer)
    full = {name: named.module for name, named in full_named.items()}
    subset_names = [
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
    ]
    subset = {name: full_named[name] for name in subset_names}
    layer_inputs = [[torch.ones(1, 2, 4, device=devices[0])]]

    plan = build_subset_plan(
        _PlanLooper(devices),
        processor=types.SimpleNamespace(
            execution_config=ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=True,
            )
        ),
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full,
        fallback=True,
        layer_inputs=layer_inputs,
    )

    looper = _ExecLooper(devices)
    empty_tasks = types.SimpleNamespace(tasks={})
    processor = types.SimpleNamespace(
        num_batches=None,
        _set_current_batch_index=lambda idx: None,
    )

    looper._prepare_named_module_for_quantization(
        processor=empty_tasks,
        named_module=full_named["self_attn.q_proj"],
        fallback_device=devices[0],
    )
    looper._prepare_named_module_for_quantization(
        processor=empty_tasks,
        named_module=full_named["self_attn.o_proj"],
        fallback_device=devices[0],
    )
    o_proj_device_after_quant_prepare = str(layer.self_attn.o_proj.weight.device)

    previous_devices = looper._apply_forward_device_overrides(
        subset,
        plan.forward_device_map,
        fallback_modules=full,
    )

    executor = ForwardExecutor(looper)
    executor.run_single(
        module=layer,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=devices[0],
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        preserve_module_devices=plan.preserve_module_devices,
    )

    if plan.restore_forward_device_overrides:
        looper._restore_forward_device_overrides(
            subset,
            previous_devices,
            fallback_modules=full,
        )

    telemetry = get_device_telemetry_records()
    o_proj_forward = [
        record
        for record in telemetry
        if record.get("event") == "hooked_linear_forward"
        and record.get("module") == "model.layers.0.self_attn.o_proj"
    ]

    summary = {
        "python_gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_devices": [
            {
                "index": idx,
                "name": torch.cuda.get_device_properties(idx).name,
            }
            for idx in range(torch.cuda.device_count())
        ],
        "nvidia_smi": _nvidia_smi_snapshot(),
        "plan_forward_device_map": {name: str(device) for name, device in plan.forward_device_map.items()},
        "o_proj_in_forward_device_map": "self_attn.o_proj" in plan.forward_device_map,
        "restore_forward_device_overrides": plan.restore_forward_device_overrides,
        "o_proj_device_after_quant_prepare": o_proj_device_after_quant_prepare,
        "o_proj_device_after_forward_restore": str(layer.self_attn.o_proj.weight.device),
        "o_proj_forward_records": o_proj_forward,
    }

    reproduced = (
        not summary["o_proj_in_forward_device_map"]
        and summary["o_proj_device_after_quant_prepare"] == "cuda:1"
        and summary["o_proj_device_after_forward_restore"] == "cuda:1"
        and any(
            record.get("input_device") == "cuda:0" and record.get("weight_device") == "cuda:1"
            for record in o_proj_forward
        )
    )
    summary["issue_2326_reproduced"] = reproduced

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
