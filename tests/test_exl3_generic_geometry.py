# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Generic EXL3 geometry and deferred runtime-weight regressions."""

import types

import pytest
import torch
from torch import nn

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_layer import _replay_layer_outputs
from gptqmodel.looper.stage_subset import SubsetPlan
from gptqmodel.quantization.config import EXL3Config


def _named(linear: nn.Module, name: str, full_name: str) -> NamedModule:
    return NamedModule(linear, name=name, full_name=full_name, layer_index=0)


def test_quant_input_weight_pads_output_columns_to_128() -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    weight = torch.randn((320, 128), dtype=torch.float16)
    module = _named(
        nn.Linear(128, 320, bias=False),
        name="gate_proj",
        full_name="model.layers.0.mlp.gate_proj",
    )

    class Capture:
        _named_module = module

        @staticmethod
        def clone_module(copy=True, device=None):
            del copy, device
            return weight.detach().float()

    exported = processor._quant_input_weight(Capture(), torch.device("cpu"))

    assert tuple(exported.shape) == (128, 384)
    assert torch.equal(exported[:, :320], weight.t().float())
    assert torch.count_nonzero(exported[:, 320:]) == 0


def test_restore_module_weight_crops_padded_output_columns() -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    module = _named(
        nn.Linear(128, 320, bias=False),
        name="gate_proj",
        full_name="model.layers.0.mlp.gate_proj",
    )
    stored = torch.arange(128 * 384, dtype=torch.float32).reshape(128, 384)

    restored = processor._restore_module_weight(module, stored)

    assert tuple(restored.shape) == (320, 128)
    assert torch.equal(restored, stored[:, :320].t().contiguous())


def test_preprocess_rejects_unaligned_input_features_before_capture() -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = EXL3Config(bits=3)
    processor.tasks = {}
    module = _named(
        nn.Linear(192, 320, bias=False),
        name="q_proj",
        full_name="model.layers.0.self_attn.q_proj",
    )

    with pytest.raises(ValueError, match="multiple of 128"):
        processor.preprocess(module)

    assert processor.tasks == {}


@pytest.mark.parametrize("shape", [(144, 208), (48, 32), (16, 16)])
def test_direct_quantize_exl3_rejects_unaligned_geometry(shape) -> None:
    weight = torch.zeros(shape, dtype=torch.float32)
    hessian = torch.eye(shape[0], dtype=torch.float32)
    h_data = {"H": hessian, "count": 1, "finalized": False}
    quant_args = {
        "K": 3,
        "devices": [torch.device("cpu")],
        "seed": 787,
        "sigma_reg": 0.025,
    }

    with pytest.raises(ValueError, match="multiple of 128"):
        quantize_exl3(weight, h_data, quant_args, return_weight_q=False)


def test_layer_runtime_weight_preparation_covers_every_named_module() -> None:
    from gptqmodel.looper.loop_processor import (
        prepare_layer_runtime_weights_for_forward,
    )

    prepared = _named(
        nn.Linear(4, 4, bias=False, device="meta"),
        name="q_proj",
        full_name="model.layers.0.self_attn.q_proj",
    )
    prepared.state["deferred"] = True
    untouched = _named(
        nn.Linear(4, 4, bias=False),
        name="v_proj",
        full_name="model.layers.0.self_attn.v_proj",
    )
    seen = []

    class Processor:
        def prepare_runtime_weight_for_forward(self, *, module, target_device):
            seen.append(module.full_name)
            if not module.state.get("deferred"):
                return None
            module.module.weight = nn.Parameter(
                torch.ones((4, 4), dtype=torch.float32), requires_grad=False
            )
            return module.module

    count = prepare_layer_runtime_weights_for_forward(
        processors=[Processor()],
        full={
            "q_proj": prepared,
            "v_proj": untouched,
            "raw": nn.Linear(4, 4, bias=False),
        },
        fallback_device=torch.device("cpu"),
    )

    assert count == 1
    assert seen == [prepared.full_name, untouched.full_name]
    assert not prepared.module.weight.is_meta


def test_replay_prepares_deferred_weights_for_modules_outside_last_subset() -> None:
    packed = _named(
        nn.Linear(4, 4, bias=False, device="meta"),
        name="q_proj",
        full_name="model.layers.0.self_attn.q_proj",
    )
    packed.state["deferred"] = True
    current = _named(
        nn.Linear(4, 4, bias=False),
        name="o_proj",
        full_name="model.layers.0.self_attn.o_proj",
    )

    class Processor:
        def prepare_runtime_weight_for_forward(self, *, module, target_device):
            del target_device
            if not module.state.get("deferred"):
                return None
            module.module.weight = nn.Parameter(
                torch.ones((4, 4), dtype=torch.float32), requires_grad=False
            )
            return module.module

    class PB:
        def manual(self):
            return self

        def set(self, **_kwargs):
            return self

        def title(self, *_args):
            return self

        def subtitle(self, *_args):
            return self

        def draw(self):
            return self

        def close(self):
            return None

    class Logger:
        def pb(self, _iterable):
            return PB()

    class Looper:
        processors = [Processor()]

        @staticmethod
        def _run_forward_batches(**_kwargs):
            assert not packed.module.weight.is_meta
            return [[torch.zeros(1, 4)]]

    plan = SubsetPlan(
        modules={"o_proj": current},
        subset_index=0,
        subset_total=1,
        execute_forward=False,
        replay_after_process=True,
        forward_mode="serial",
        batch_count=1,
        forward_row_counts=[1],
        forward_total_rows=1,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=types.SimpleNamespace(),
        module_chunks=[{"o_proj": current}],
    )

    outputs = _replay_layer_outputs(
        Looper(),
        module=nn.Identity(),
        processor=object(),
        layer_inputs=[[torch.zeros(1, 4)]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={"q_proj": packed, "o_proj": current},
        log=Logger(),
        region_timer=None,
        replay_plan=plan,
    )

    assert len(outputs) == 1
    assert torch.equal(outputs[0][0], torch.zeros(1, 4))
