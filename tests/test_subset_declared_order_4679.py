# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Subset capture must not stop on the declared-last member (#4679).

``stage_subset`` set ``forward_hook_last`` on the member at the declared end of
the subset and ``HookedLinear`` raised ``StopForward`` when it ran. That assumes
the declaration order matches the order the model calls the modules:

* Falcon-H1's MLP computes ``up_proj(x)`` before ``gate_proj(x)``, so with
  ``("gate_proj:0", "up_proj:0", ...)`` the forward stopped at ``up_proj`` and
  ``gate_proj`` never received a calibration sample (GPTQ silently fell back to
  RTN; AWQ crashed on the empty feature).
* bailing_hybrid's hybrid attention calls ``b_proj`` before ``g_proj``, with the
  same effect on ``self_attn.g_proj``.

Capture now stops only once every hooked member of the subset has fired.
"""

import os
import threading
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import torch
from module_tree.looper_doubles import no_auto_dtype_cast_forward

from gptqmodel.looper.loop_processor import CalibrationConfig, ForwardConfig, LoopProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import build_subset_plan, run_subset_stage
from gptqmodel.nn_modules.hooked_linear import replace_module_with_hooked_legacy
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.quantization.config import VramStrategy
from gptqmodel.utils.model import find_modules


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytestmark = [pytest.mark.cpu]


def _prepare_dataset_func(**kwargs):
    return kwargs["calibration_dataset"]


class _StubProcessor(LoopProcessor):
    """Record which subset members the model actually called."""

    def __init__(self, qcfg: QuantizeConfig):
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=CalibrationConfig(
                dataset=[{"input_ids": torch.tensor([1, 2, 3])}],
                prepare_dataset=_prepare_dataset_func,
            ),
            forward=ForwardConfig(
                required=True,
                replay_after_process=False,
                subset_early_stop=True,
            ),
        )
        self.hook_calls: List[str] = []
        self.process_calls: List[str] = []
        self._lock = threading.Lock()

    @classmethod
    def name(cls) -> str:
        return "stub-4679"

    def preprocess(self, module: NamedModule, fallback=None, **_kwargs):
        self.tasks[module.name] = {"inputs": []}

    def pre_process_fwd_hook(self, name: str):
        def _hook(_module, _inp, _out):
            with self._lock:
                self.hook_calls.append(name)

        return _hook

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        self.process_calls.append(module.name)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        return True

    def pack_module(self, module):
        pass

    def submodule_finalize(self, module: NamedModule, model, **kwargs):
        pass


class _OrderedMLP(torch.nn.Module):
    """Run ``up_proj`` before ``gate_proj``, like ``FalconH1MLP.forward``."""

    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states):
        up = self.up_proj(hidden_states)
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(up * gate)


class _OrderedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = _OrderedMLP()
        self.after_subset = torch.nn.Linear(4, 4, bias=False)
        self.after_subset_called = False

    def forward(self, hidden_states, **kwargs):
        x = self.feed_forward(hidden_states)
        x = self.after_subset(x)
        self.after_subset_called = True
        return (x,)


class _DummyProgress:
    def title(self, *_args, **_kwargs):
        return self

    def subtitle(self, *_args, **_kwargs):
        return self

    def draw(self):
        return self


def _run_subset(layer, subset_names):
    quant_cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        device="cpu",
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
    )
    replace_module_with_hooked_legacy(layer)
    dummy_model = SimpleNamespace(
        support_batch_quantize=False,
        quantize_config=quant_cfg,
        layer_callback=None,
        subset_callback=None,
        supported_dense_vram_strategies=[VramStrategy.EXCLUSIVE, VramStrategy.BALANCED],
        layer_modules_strict=True,
        lm_head="lm_head",
        shell_module_materialize=lambda target_submodule, device, role, named_module=None: target_submodule,
        prepare_layer_replay_kwargs=lambda layer, layer_input, additional_inputs, target_device: additional_inputs,
        prepare_layer_replay_keep_mask=lambda **kwargs: kwargs["keep_mask"],
        _maybe_auto_dtype_cast_forward_module=no_auto_dtype_cast_forward,
    )
    processor = _StubProcessor(quant_cfg)
    looper = ModuleLooper(model=dummy_model, processors=[processor])

    layer_inputs = [[torch.randn(2, 4)]]
    full_modules = find_modules(layer)
    subset = looper.create_named_modules(
        module=layer,
        full=full_modules,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="layers",
        names=subset_names,
        processor=processor,
        fallback=False,
        layer_module=layer,
    )
    plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=2,
        full=full_modules,
        fallback=False,
        layer_inputs=layer_inputs,
    )
    run_subset_stage(
        looper=looper,
        plan=plan,
        processor=processor,
        module=layer,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="layers.0",
        layer_title="subset-order-check",
        layer_index=0,
        full=full_modules,
        fallback=False,
        shared_kv_cache_dict={},
        pb=_DummyProgress(),
        log=None,
        region_timer=None,
        previous_processed_subset=None,
        subset_event_cb=looper._subset_event_dispatch,
    )
    return processor


def test_subset_captures_a_sibling_the_model_runs_after_the_declared_last_member():
    layer = _OrderedLayer()

    processor = _run_subset(
        layer, ["feed_forward.gate_proj", "feed_forward.up_proj"]
    )

    # `up_proj` is declared last but runs first; `gate_proj` must still be
    # captured instead of being cut off by the early stop.
    assert set(processor.hook_calls) == {
        "feed_forward.gate_proj",
        "feed_forward.up_proj",
    }
    assert layer.after_subset_called is False
    assert set(processor.process_calls) == {
        "feed_forward.gate_proj",
        "feed_forward.up_proj",
    }


def test_subset_stops_once_the_declared_order_member_completes_the_set():
    layer = _OrderedLayer()

    processor = _run_subset(
        layer, ["feed_forward.up_proj", "feed_forward.gate_proj"]
    )

    # Declaration matches execution here: the second member completes the
    # subset and the forward still stops before the next module.
    assert processor.hook_calls == [
        "feed_forward.up_proj",
        "feed_forward.gate_proj",
    ]
    assert layer.after_subset_called is False
