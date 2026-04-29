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
from defuser import convert_model
from defuser.modeling.unfused_moe.qwen2_moe import LinearQwen2MoeSparseMoeBlock
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from gptqmodel.models import BaseQModel
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER


repo_root = Path(__file__).resolve().parents[2]
repo_str = str(repo_root)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.loop_processor import ExecutionConfig, LoopProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import build_subset_plan, run_subset_stage
from gptqmodel.models.definitions.phi3 import Phi3QModel
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.nn_modules.hooked_linear import HookedLinear, StopForward, replace_module_with_hooked_legacy
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig, VramStrategy
from gptqmodel.utils.model import find_modules, get_module_by_name_prefix, restore_moe_topk, set_moe_topk


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
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
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
    assert gate_block.index("mlp.shared_expert.gate_proj") < gate_block.index("mlp.experts.{expert_index}.gate_proj")
    assert gate_block.index("mlp.shared_expert.up_proj") < gate_block.index("mlp.experts.{expert_index}.up_proj")
    assert "mlp.experts.{expert_index}.gate_proj" in gate_block
    assert "mlp.experts.{expert_index}.up_proj" in gate_block

    down_block = next(block for block in blocks if "mlp.shared_expert.down_proj" in block)
    assert down_block.index("mlp.shared_expert.down_proj") < down_block.index("mlp.experts.{expert_index}.down_proj")
    assert "mlp.experts.{expert_index}.down_proj" in down_block

    expert_gate_blocks = [block for block in blocks if "mlp.experts.{expert_index}.gate_proj" in block]
    assert len(expert_gate_blocks) == 1


def test_phi3_defused_mlp_modules_match_strict_subset_builder():
    cfg = Phi3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
    )
    model = Phi3ForCausalLM(cfg)
    convert_model(model, cleanup_original=False)

    layer = model.model.layers[0]
    full = find_modules(layer)
    assert "mlp.gate_up_proj" not in full
    assert {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}.issubset(full)

    gate_block = next(
        block for block in Phi3QModel.full_layer_modules(model_config=cfg)
        if "mlp.gate_proj" in block
    )
    assert gate_block == ["mlp.gate_proj", "mlp.up_proj"]

    class _NoOpProcessor:
        def preprocess(self, named_module):
            return None

        def is_skipped(self, named_module):
            return False

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(
        support_batch_quantize=True,
        quantize_config=SimpleNamespace(device="cpu", compute_device_filter=None, method=None),
        layer_modules_strict=True,
        lm_head="lm_head",
    )
    subset = looper.create_named_modules(
        layer,
        full,
        False,
        0,
        "model.layers",
        gate_block,
        _NoOpProcessor(),
        fallback=None,
        layer_module=layer,
    )

    assert set(subset) == {"mlp.gate_proj", "mlp.up_proj"}


def test_qwen2_moe_awq_expansion_keeps_shared_expert_before_experts():
    blocks = Qwen2MoeQModel.simple_layer_modules(
        model_config=SimpleNamespace(num_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
        is_awq_quantize=True,
    )

    gate_block = next(block for block in blocks if "mlp.shared_expert.gate_proj" in block)
    assert gate_block == [
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
    ]
    down_block = next(block for block in blocks if "mlp.shared_expert.down_proj" in block)
    assert down_block == [
        "mlp.shared_expert.down_proj",
        "mlp.experts.0.down_proj",
        "mlp.experts.1.down_proj",
    ]

def test_qwen3_5_moe_shared_expert_merges_with_experts():
    blocks = Qwen3_5_MoeQModel.build_layer_modules(Qwen3_5_MoeQModel.module_tree)

    gate_block = next(block for block in blocks if "mlp.shared_expert.gate_proj" in block)
    assert gate_block.index("mlp.shared_expert.gate_proj") < gate_block.index("mlp.experts.{expert_index}.gate_proj")
    assert gate_block.index("mlp.shared_expert.up_proj") < gate_block.index("mlp.experts.{expert_index}.up_proj")
    assert "mlp.experts.{expert_index}.gate_proj" in gate_block
    assert "mlp.experts.{expert_index}.up_proj" in gate_block

    down_block = next(block for block in blocks if "mlp.shared_expert.down_proj" in block)
    assert down_block.index("mlp.shared_expert.down_proj") < down_block.index("mlp.experts.{expert_index}.down_proj")
    assert "mlp.experts.{expert_index}.down_proj" in down_block

    expert_gate_blocks = [block for block in blocks if "mlp.experts.{expert_index}.gate_proj" in block]
    assert len(expert_gate_blocks) == 1


def test_awq_moe_expansion_preserves_non_expert_segments():
    class _MockOrderedMoEModel(BaseQModel):
        dynamic_expert_index = "num_experts"

    expanded = _MockOrderedMoEModel.build_moe_modules_if_need(
        SimpleNamespace(num_experts=2),
        [[
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj",
            f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj",
            "mlp.shared_expert.down_proj",
        ]],
        is_awq_quantize=True,
    )

    assert expanded == [[
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
        "mlp.shared_expert.down_proj",
    ]]


def test_moe_lifecycle_execution_order_follows_ordered_module_names():
    hooks = GateUpDownMoELifecycleHooks()

    shared_first = hooks.get_subset_execution_order(
        ordered_module_names=[
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.experts.0.gate_proj",
        ],
        moe_block_prefix="mlp",
        experts_attr_name="experts",
        shared_expert_attr_name="shared_expert",
    )
    assert shared_first == ["shared", "experts"]

    experts_first = hooks.get_subset_execution_order(
        ordered_module_names=[
            "mlp.experts.0.gate_proj",
            "mlp.experts.0.up_proj",
            "mlp.shared_expert.gate_proj",
        ],
        moe_block_prefix="mlp",
        experts_attr_name="experts",
        shared_expert_attr_name="shared_expert",
    )
    assert experts_first == ["experts", "shared"]


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

    assert processor.execution_config.subset_forward_early_stop is True


def test_module_looper_subset_callback_invoked():
    quant_cfg = _make_quant_config()
    dummy_model = SimpleNamespace(
        support_batch_quantize=False,
        quantize_config=quant_cfg,
        layer_callback=None,
        subset_callback=None,
        supported_dense_vram_strategies=[VramStrategy.EXCLUSIVE],
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
            execution_config=ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=False,
                subset_forward_early_stop=True,
            ),
        )
        self.hook_calls: List[str] = []
        self.process_calls: List[str] = []

        self._layer_states: Dict[int, _AWQLayerState] = {}
        self._layer_states_lock = threading.Lock()

    @classmethod
    def name(cls) -> str:
        return "stub-awq"

    def preprocess(self, module: NamedModule, fallback=None, **_kwargs):
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


class _MiniRouterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(4, 4)
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        x = self.router(hidden_states)
        x = self.proj(x)
        return (x,)


def test_replace_module_with_hooked_legacy_skips_not_quantized_paths():
    layer = _MiniRouterLayer()

    replace_module_with_hooked_legacy(layer, skip_module_paths={"router"})

    assert isinstance(layer.router, torch.nn.Linear)
    assert not isinstance(layer.router, HookedLinear)
    assert isinstance(layer.proj, HookedLinear)


def test_stage_subset_early_stop_and_callbacks():
    quant_cfg = _make_quant_config()
    mini_layer = _MiniLayer()
    replace_module_with_hooked_legacy(mini_layer)

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

    # Create subset from names
    subset = looper.create_named_modules(
        module=mini_layer,
        full=full_modules,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="layers",
        names=subset_names,
        processor=processor,
        fallback=False,
        layer_module=mini_layer,
    )

    subset_plan = build_subset_plan(
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
        plan=subset_plan,
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
        full=full_modules,
        fallback=False,
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


def test_qwen3_5_moe_subset_early_stop_follows_module_tree_execution_order():
    """Regression for Qwen 3.5/3.6 shared-expert ordering inside merged MoE subsets."""

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3_5MoeForCausalLM(cfg)
    convert_model(model, cleanup_original=False)
    layer = model.model.layers[0]
    replace_module_with_hooked_legacy(layer)

    quant_cfg = _make_quant_config()

    class _DummyQwen3_5Model:
        moe_lifecycle_hooks = Qwen3_5_MoeQModel.moe_lifecycle_hooks
        layer_modules_strict = True
        lm_head = "lm_head"
        supported_dense_vram_strategies = [VramStrategy.EXCLUSIVE, VramStrategy.BALANCED]

        def __init__(self, qcfg: QuantizeConfig):
            self.support_batch_quantize = False
            self.quantize_config = qcfg
            self.layer_callback = None
            self.subset_callback = None

        @classmethod
        def get_moe_module_name(cls):
            return Qwen3_5_MoeQModel.get_moe_module_name()

        def shell_module_materialize(self, target_submodule, device, role=None, named_module=None):
            return target_submodule

        def prepare_layer_replay_kwargs(self, layer, layer_input, additional_inputs, target_device):
            del layer, target_device
            hidden_states = layer_input[0]
            position_ids = additional_inputs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                additional_inputs["position_ids"] = position_ids
            additional_inputs["position_embeddings"] = model.model.rotary_emb(hidden_states, position_ids)
            return additional_inputs

    processor = _StubAWQProcessor(quant_cfg)
    looper = ModuleLooper(model=_DummyQwen3_5Model(quant_cfg), processors=[processor])

    subset_names = next(
        block
        for block in Qwen3_5_MoeQModel.simple_layer_modules(
            model_config=cfg,
            quantize_config=SimpleNamespace(dynamic=None),
            is_awq_quantize=True,
        )
        if "mlp.shared_expert.gate_proj" in block
    )
    assert subset_names[:2] == [
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
    ]
    assert subset_names[-1] == "mlp.experts.3.up_proj"

    layer_inputs = [[torch.randn(1, 4, cfg.hidden_size)]]
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
    subset_plan = build_subset_plan(
        looper,
        processor=processor,
        subset=subset,
        subset_index=0,
        subset_total=1,
        full=full_modules,
        fallback=False,
        layer_inputs=layer_inputs,
    )

    run_subset_stage(
        looper=looper,
        plan=subset_plan,
        processor=processor,
        module=layer,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="layers.0",
        layer_title="subset-check",
        layer_index=0,
        full=full_modules,
        fallback=False,
        shared_kv_cache_dict={},
        pb=_DummyProgress(),
        log=None,
        region_timer=None,
        previous_processed_subset=None,
        subset_event_cb=None,
    )

    assert processor.hook_calls[:2] == [
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
    ]
    assert any(name.startswith("mlp.experts.") for name in processor.hook_calls)
    assert processor.hook_calls[-1] == "mlp.experts.3.up_proj"


def test_qwen2_moe_routing_override_all_runs_shared_expert_before_last_expert():
    """Regression for Qwen2 MoE: routing override must not early-stop before shared expert runs."""

    cfg = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen2MoeForCausalLM(cfg)
    layer = model.model.layers[0]
    # Quantization patches Qwen2 to defuser's explicit-expert block, so the
    # regression must exercise that execution path instead of the raw fused HF block.
    layer.mlp = LinearQwen2MoeSparseMoeBlock(cfg)
    replace_module_with_hooked_legacy(layer)

    down_block = next(
        block
        for block in Qwen2MoeQModel.simple_layer_modules(
            model_config=cfg,
            quantize_config=SimpleNamespace(dynamic=None),
            is_awq_quantize=True,
        )
        if "mlp.shared_expert.down_proj" in block
    )
    assert down_block == [
        "mlp.shared_expert.down_proj",
        "mlp.experts.0.down_proj",
        "mlp.experts.1.down_proj",
        "mlp.experts.2.down_proj",
        "mlp.experts.3.down_proj",
    ]

    hook_calls: List[str] = []
    hooked_modules: List[HookedLinear] = []
    for idx, name in enumerate(down_block):
        hooked_module, _ = get_module_by_name_prefix(layer, name)
        assert isinstance(hooked_module, HookedLinear)
        hooked_module.forward_hook = lambda _module, _inp, _out, module_name=name: hook_calls.append(module_name)
        hooked_module.forward_hook_last = idx == (len(down_block) - 1)
        hooked_modules.append(hooked_module)

    routing_state = set_moe_topk(layer, cfg.num_experts)
    stopped = False
    try:
        with torch.inference_mode():
            layer.mlp(torch.randn(1, 4, cfg.hidden_size))
    except StopForward:
        stopped = True
    finally:
        restore_moe_topk(routing_state)
        for hooked_module in hooked_modules:
            hooked_module.forward_hook = None
            hooked_module.forward_hook_last = False

    assert stopped is True
    assert hook_calls == down_block
