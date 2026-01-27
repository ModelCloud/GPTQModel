# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from gptqmodel.models import BaseQModel


repo_root = Path(__file__).resolve().parents[2]
repo_str = str(repo_root)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import run_subset_stage
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.nn_modules.hooked_linear import replace_module_with_hooked_legacy
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig, VramStrategy
from gptqmodel.utils.model import find_modules, get_module_by_name_prefix


# honour the request to bind the test harness to GPU index 5 when CUDA is available
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def _prepare_dataset_func(**kwargs):
    return kwargs["calibration_dataset"]


def _make_quant_config(device: torch.device | str = "cpu") -> QuantizeConfig:
    return QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        device=device,
        vram_strategy=VramStrategy.EXCLUSIVE,
    )


def test_mlp_capture_flag_propagates_to_layer_modules():
    cfg = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
        per_token_num_experts=1,
        num_experts=2,
    )
    model = Qwen3MoeForCausalLM(cfg)

    model_config = cfg
    quant_cfg = SimpleNamespace(dynamic=None)

    tree = Qwen3MoeQModel.build_layer_modules(
        Qwen3MoeQModel.module_tree,
        include_capture_only=True,
    )
    assert any("mlp:?" in group for group in tree)

    baseline_tree = Qwen3MoeQModel.build_layer_modules(Qwen3MoeQModel.module_tree)
    assert all(":?" not in name for block in baseline_tree for name in block)

    full = Qwen3MoeQModel.full_layer_modules(
        model_config=model_config,
        is_awq_quantize=True,
        include_capture_only=True,
    )
    capture_blocks = [block for block in full if any(":?" in name for name in block)]
    assert capture_blocks and "mlp:?" in capture_blocks[0]

    simple = Qwen3MoeQModel.simple_layer_modules(
        model_config=model_config,
        quantize_config=quant_cfg,
        is_awq_quantize=True,
    )
    assert all(":?" not in name for block in simple for name in block)

    layer = model.model.layers[0]
    mlp_module, _ = get_module_by_name_prefix(layer, "mlp")
    assert isinstance(mlp_module, Qwen3MoeSparseMoeBlock)


def test_qwen2_moe_shared_expert_merges_with_experts():
    blocks = Qwen2MoeQModel.build_layer_modules(Qwen2MoeQModel.module_tree)

    gate_block = next(block for block in blocks if "mlp.shared_expert.gate_proj" in block)
    assert "mlp.experts.{expert_index}.gate_proj" in gate_block
    assert "mlp.experts.{expert_index}.up_proj" in gate_block

    down_block = next(block for block in blocks if "mlp.shared_expert.down_proj" in block)
    assert "mlp.experts.{expert_index}.down_proj" in down_block

    expert_gate_blocks = [block for block in blocks if "mlp.experts.{expert_index}.gate_proj" in block]
    assert len(expert_gate_blocks) == 1


def test_awq_processor_enables_subset_early_stop():
    calibration = [{"input_ids": torch.tensor([1, 2, 3])}]
    qcfg = _make_quant_config()
    dummy_gptq_model = SimpleNamespace()
    dummy_model = torch.nn.Linear(3, 3)

    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration,
        prepare_dataset_func=_prepare_dataset_func,
        calibration_concat_size=None,
        calibration_sort=None,
        calibration_concat_separator=None,
        batch_size=1,
        gptq_model=dummy_gptq_model,
        model=dummy_model,
    )

    assert processor.subset_forward_early_stop is True


def test_module_looper_subset_callback_invoked():
    quant_cfg = _make_quant_config()
    dummy_model = SimpleNamespace(
        support_batch_quantize=False,
        quantize_config=quant_cfg,
        layer_callback=None,
        subset_callback=None,
        supported_vram_strategies=[VramStrategy.EXCLUSIVE],
    )

    looper = ModuleLooper(model=dummy_model, processors=[])

    events: List[Dict[str, object]] = []
    looper.register_subset_callback(lambda **payload: events.append(payload))

    looper._subset_event_dispatch(
        stage="forward_start",
        layer_idx=0,
        subset_index=0,
        subset_total=1,
        module_names=["self_attn.q_proj"],
        processor="stub",
    )

    assert events and events[0]["module_names"] == ["self_attn.q_proj"]


class _DummyProgress:
    def title(self, *_args, **_kwargs):
        return self

    def subtitle(self, *_args, **_kwargs):
        return self

    def draw(self):
        return self


class _SubsetRecorder:
    def __init__(self):
        self.events: List[Dict[str, object]] = []

    def __call__(
        self,
        *,
        stage: str,
        layer_idx: int,
        subset_index: int,
        subset_total: int,
        module_names: List[str],
        processor: str,
    ):
        self.events.append(
            {
                "stage": stage,
                "layer_idx": layer_idx,
                "subset_index": subset_index,
                "subset_total": subset_total,
                "module_names": module_names,
                "processor": processor,
            }
        )


class _StubAWQProcessor(LoopProcessor):
    def __init__(self, qcfg: QuantizeConfig):
        calibration = [{"input_ids": torch.tensor([1, 2, 3])}]
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=calibration,
            prepare_dataset_func=_prepare_dataset_func,
            batch_size=1,
            require_fwd=True,
            fwd_after_process=False,
            subset_forward_early_stop=True,
        )
        self.hook_calls: List[str] = []
        self.process_calls: List[str] = []

        self._layer_states: Dict[int, _AWQLayerState] = {}
        self._layer_states_lock = threading.Lock()

    @classmethod
    def name(cls) -> str:
        return "stub-awq"

    def preprocess(self, module: NamedModule, failsafe=None, **_kwargs):
        self.tasks[module.name] = {"inputs": []}

    def pre_process_fwd_hook(self, name: str) -> Callable[[torch.nn.Module, tuple, torch.Tensor], None]:
        def _hook(_module, _inp, _out):
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
        layer_index = module.layer_index
        state = self._get_layer_state(layer_index)
        self.process_calls.append(module.name)
        state.quantized = True

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        return True

    def _get_layer_state(self, layer_index: int) -> _AWQLayerState:
        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _AWQLayerState()
                self._layer_states[layer_index] = state
        return state

    def pack_module(self, module):
        pass

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        self.pack_module(module)


class _MiniSelfAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)


class _MiniLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MiniSelfAttn()
        self.after_o_proj_called = False

    def forward(self, hidden_states, **kwargs):
        x = self.self_attn.q_proj(hidden_states)
        x = self.self_attn.k_proj(x)
        x = self.self_attn.v_proj(x)
        x = self.self_attn.o_proj(x)
        self.after_o_proj_called = True
        return (x,)


def test_stage_subset_early_stop_and_callbacks():
    quant_cfg = _make_quant_config()
    mini_layer = _MiniLayer()
    replace_module_with_hooked_legacy(mini_layer)

    dummy_model = SimpleNamespace(
        support_batch_quantize=False,
        quantize_config=quant_cfg,
        layer_callback=None,
        subset_callback=None,
        supported_vram_strategies=[VramStrategy.EXCLUSIVE, VramStrategy.BALANCED],
        layer_modules_strict=True,
        lm_head="lm_head",
    )

    processor = _StubAWQProcessor(quant_cfg)
    looper = ModuleLooper(model=dummy_model, processors=[processor])

    recorder = _SubsetRecorder()
    looper.register_subset_callback(recorder)

    layer_inputs = [[torch.randn(2, 4)]]
    layer_input_kwargs = [{}]
    position_ids: List[Optional[torch.Tensor]] = [None]
    attention_masks: List[Optional[torch.Tensor]] = [None]
    shared_kv_cache_dict: Dict[int, torch.Tensor] = {}

    full_modules = find_modules(mini_layer)
    subset_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    subset = {
        name: NamedModule(
            full_modules[name],
            name=name,
            full_name=f"layers.0.{name}",
            layer_index=0,
        )
        for name in subset_names
    }

    run_subset_stage(
        looper=looper,
        processor=processor,
        module=mini_layer,
        layer_inputs=layer_inputs,
        layer_input_kwargs=layer_input_kwargs,
        position_ids=position_ids,
        attention_masks=attention_masks,
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="layers.0",
        layer_title="subset-check",
        layer_index=0,
        layers_prefix="layers",
        subset=subset,
        subset_index=0,
        subset_total=2,
        full=full_modules,
        failsafe=False,
        shared_kv_cache_dict=shared_kv_cache_dict,
        pb=_DummyProgress(),
        log=None,
        region_timer=None,
        previous_processed_subset=None,
        subset_event_cb=looper._subset_event_dispatch,
    )

    assert mini_layer.after_o_proj_called is False
    assert recorder.events and [evt["stage"] for evt in recorder.events] == [
        "forward_start",
        "forward_end",
        "quant_start",
        "quant_complete",
    ]
    assert recorder.events[0]["module_names"] == subset_names
    assert processor.hook_calls and processor.hook_calls[-1] == subset_names[-1]
    assert set(processor.process_calls) == set(subset_names)
    assert len(processor.process_calls) == len(subset_names)
