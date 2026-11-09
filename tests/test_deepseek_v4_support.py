from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
)

from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import (
    MTP_CAPTURE_ATTENTION_MASK,
    MTP_CAPTURE_DECODE_MASK,
    MTP_CAPTURE_INPUT_IDS,
    DeepSeekV4MTPAuxiliary,
    DeepSeekV4MTPAuxiliaryShell,
    DeepSeekV4MTPPrefixRuntime,
    DeepSeekV4MTPQuantizationModel,
    DeepSeekV4MTPReplay,
    DeepSeekV4MTPReplayBatch,
    DeepSeekV4MTPTargetTapEvent,
    DeepSeekV4QModel,
    DeepSeekV4TargetAnchorResolver,
    deepseek_v4_mtp_source_projection_prefix,
    expected_deepseek_v4_mtp_checkpoint_keys,
    patch_deepseek_v4_router_precision,
    validate_deepseek_v4_mtp_checkpoint_keys,
)
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.quantization.config import AutoModuleDecoderConfig, EXL3Config
from gptqmodel.utils.exl3_capture_frontier import (
    EXL3CaptureFrontierStore,
    EXL3CaptureState,
)


def _tiny_v4_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        n_routed_experts=3,
        num_experts_per_tok=2,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3,
        mlp_layer_types=["moe"] * 3,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
        dspark_block_size=5,
        dspark_noise_token_id=31,
        partial_rotary_factor=0.5,
        dtype="bfloat16",
    )


@pytest.mark.parametrize(
    ("runtime", "checkpoint"),
    (
        ("mtp.0.mlp.experts.7.gate_proj", "mtp.0.ffn.experts.7.w1"),
        ("mtp.1.mlp.experts.8.up_proj", "mtp.1.ffn.experts.8.w3"),
        ("mtp.2.mlp.experts.9.down_proj", "mtp.2.ffn.experts.9.w2"),
    ),
)
def test_mtp_save_overlay_maps_runtime_projections_to_checkpoint_prefixes(
    runtime,
    checkpoint,
):
    assert deepseek_v4_mtp_source_projection_prefix(runtime) == checkpoint


@pytest.mark.parametrize(
    ("runtime", "checkpoint"),
    (
        (
            "model.layers.4.mlp.experts.7.gate_proj",
            "layers.4.ffn.experts.7.w1",
        ),
        ("mtp.1.mlp.experts.8.up_proj", "mtp.1.ffn.experts.8.w3"),
    ),
)
def test_deepseek_v4_exl3_resolves_native_checkpoint_projection(
    runtime,
    checkpoint,
):
    owner = object.__new__(DeepSeekV4QModel)
    assert owner.exllamav3_checkpoint_module_candidates(runtime) == (
        runtime,
        checkpoint,
    )


def test_deepseek_v4_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_v4")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert auto.check_and_get_model_definition("/tmp/deepseek-v4") is DeepSeekV4QModel


def test_deepseek_v4_module_tree_matches_v4_attention_and_fused_experts():
    layer_modules = DeepSeekV4QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=256),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.kv_proj" in flat_modules
    assert "self_attn.o_b_proj" in flat_modules
    # grouped projection must stay native and should not be part of quant blocks
    assert "self_attn.o_a_proj" not in flat_modules
    assert "mlp.experts.99.gate_proj" in flat_modules
    assert "mlp.experts.99.up_proj" in flat_modules
    assert "mlp.experts.99.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules


def test_deepseek_v4_preserves_integrated_mtp_namespace() -> None:
    assert DeepSeekV4QModel.out_of_model_tensors == {
        "prefixes": ["mtp", "vision", "aligner"],
        "suffixes": [".ffn.gate.bias_vl"],
        "tensors": [
            "image_start",
            "image_end",
            "image_newline",
            "image_pad",
            "layers.0.ffn.gate.bias",
            "layers.1.ffn.gate.bias",
            "layers.2.ffn.gate.bias",
        ],
    }


def test_deepseek_v4_configures_base_replay_store_with_resolved_path(
    tmp_path,
) -> None:
    harness = object.__new__(DeepSeekV4QModel)
    root = tmp_path / "nested" / ".." / "replay"

    harness.configure_base_replay_store(root, provenance={"plan_sha256": "abc"})

    assert harness._base_replay_store_root == str((tmp_path / "replay").resolve())
    assert harness._base_replay_store_provenance == {"plan_sha256": "abc"}


def test_deepseek_v4_target_input_capture_keeps_first_layer_lazy() -> None:
    first = nn.Linear(2, 2, device="meta")
    second = nn.Linear(2, 2, device="meta")
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(
        model=SimpleNamespace(layers=nn.ModuleList([first, second]))
    )

    assert (
        harness.prepare_input_capture_layer(
            first,
            module_path="model.layers.0",
            device=torch.device("cpu"),
        )
        is first
    )
    assert first.weight.device.type == "meta"

    try:
        harness.prepare_input_capture_layer(
            second,
            module_path="model.layers.1",
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "layer zero" in str(exc)
    else:
        raise AssertionError("nonzero target input-capture layer was accepted")


def test_deepseek_v4_pre_quantize_keeps_packed_target_layer_lazy(monkeypatch) -> None:
    first = nn.Linear(2, 2, device="meta")
    harness = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        model=SimpleNamespace(layers=nn.ModuleList([first]))
    )
    monkeypatch.setattr(
        harness,
        "shell_module_materialize",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("packed parent layer was eagerly materialized")
        ),
    )

    assert harness.pre_quantize(first) is first
    assert first.weight.device.type == "meta"


def test_auto_decoder_materializes_unquantized_linear_subclass_with_its_forward(
    monkeypatch,
) -> None:
    class GroupedLinear(nn.Linear):
        pass

    target = GroupedLinear(2, 4, bias=False, device="meta")
    named = SimpleNamespace(
        full_name="model.layers.0.self_attn.o_a_proj",
        state={},
    )
    harness = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(harness)
    harness.model = nn.Module()
    harness.model.proj = target
    harness.quantize_config = SimpleNamespace(
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)]
    )
    harness.auto_module_decoder_events = []
    harness.turtle_model = SimpleNamespace(
        checkpoint_tensors_for_submodule=lambda **_kwargs: {
            "weight": torch.zeros((4, 2), dtype=torch.float8_e4m3fn),
            "weight_scale": torch.ones((), dtype=torch.float32),
        }
    )
    decoded = GroupedLinear(2, 4, bias=False, dtype=torch.bfloat16)
    monkeypatch.setattr(harness, "_decoder_weight_format", lambda **_kwargs: "fp8")
    monkeypatch.setattr(
        harness,
        "_build_decoder_quant_source_module",
        lambda *_args, **_kwargs: decoded,
    )
    monkeypatch.setattr(
        harness,
        "_build_decoder_forward_module",
        lambda *, quant_source, device: quant_source.to(device),
    )
    monkeypatch.setattr(
        harness,
        "_replace_live_submodule",
        lambda _current, replacement: replacement,
    )

    replacement = harness._prepare_auto_decoder_forward_module(
        target_submodule=target,
        device=torch.device("cpu"),
        named_module=named,
    )

    assert isinstance(replacement, GroupedLinear)
    assert named.state["auto_module_decoder"]["target_dtype"] == torch.bfloat16
    assert named.state["auto_module_decoder_forward_mode"] == "decode"


def test_deepseek_v4_mtp_checkpoint_contract_is_exact_and_does_not_trust_nextn_count() -> (
    None
):
    config = _tiny_v4_config()
    assert config.num_nextn_predict_layers == 1
    keys = expected_deepseek_v4_mtp_checkpoint_keys(config)
    report = validate_deepseek_v4_mtp_checkpoint_keys(config, keys)

    assert report == {
        "block_count": 3,
        "target_layer_ids": [0, 1, 2],
        "routed_experts_per_block": 3,
        "tensor_count": len(keys),
    }
    assert "mtp.0.ffn.experts.0.w1.weight" in keys
    assert "mtp.2.ffn.experts.2.w3.scale" in keys
    assert "mtp.0.main_proj.weight" in keys
    assert "mtp.2.confidence_head.proj.weight" in keys

    config.vision_n_layers = 1
    vision_keys = expected_deepseek_v4_mtp_checkpoint_keys(config)
    assert "mtp.0.ffn.gate.bias_vl" in vision_keys
    assert "mtp.2.ffn.gate.bias_vl" in vision_keys
    validate_deepseek_v4_mtp_checkpoint_keys(config, vision_keys)
    config.vision_n_layers = 0

    missing = set(keys)
    missing.remove("mtp.1.ffn.experts.2.w2.scale")
    try:
        validate_deepseek_v4_mtp_checkpoint_keys(config, missing)
    except RuntimeError as exc:
        assert "missing=1" in str(exc)
    else:
        raise AssertionError("an incomplete MTP namespace was accepted")

    unexpected = set(keys)
    unexpected.add("mtp.3.ffn.gate.weight")
    try:
        validate_deepseek_v4_mtp_checkpoint_keys(config, unexpected)
    except RuntimeError as exc:
        assert "unexpected=1" in str(exc)
    else:
        raise AssertionError("an unknown MTP block was accepted")


def test_deepseek_v4_mtp_shell_is_defused_patched_and_fail_closed() -> None:
    shell = DeepSeekV4MTPAuxiliaryShell(_tiny_v4_config())

    assert shell.target_layer_ids == (0, 1, 2)
    assert shell.base_num_hidden_layers == 3
    assert shell.config.num_hidden_layers == 6
    assert shell.config.layer_types[-3:] == ["sliding_attention"] * 3
    assert shell.config.mlp_layer_types[-3:] == ["moe"] * 3
    assert len(shell.mtp) == 3
    assert shell.mtp[0].main_proj.in_features == 48
    assert shell.mtp[0].main_proj.out_features == 16
    assert shell.mtp[2].confidence_head.proj.in_features == 20
    assert shell.mtp[2].markov_head.markov_w1.weight.shape == (32, 4)

    for block in shell.mtp:
        assert block.self_attn.layer_type == "sliding_attention"
        assert len(block.mlp.experts) == 3
        assert hasattr(block.mlp.experts[0], "gate_proj")
        assert hasattr(block.mlp.experts[0], "up_proj")
        assert hasattr(block.mlp.experts[0], "down_proj")
        assert block.mlp.gate._gptqmodel_v4_fp32_router
        assert block.mlp.gate.weight.dtype is torch.bfloat16
        assert block.mlp.gate.e_score_correction_bias.dtype is torch.float32
        assert block.self_attn.sinks.dtype is torch.float32
        assert block.attn_hc.fn.dtype is torch.float32

    try:
        shell()
    except RuntimeError as exc:
        assert "must not be appended to target layers" in str(exc)
    else:
        raise AssertionError("generic MTP shell forward did not fail closed")


def test_deepseek_v4_mtp_replay_keeps_five_rows_joint_and_uses_target_lane_means() -> (
    None
):
    torch.manual_seed(0xD54)
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
        for block in shell.mtp:
            block.mlp.gate.e_score_correction_bias.zero_()
    embedding = torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    replay = DeepSeekV4MTPReplay(shell, embedding_weight=embedding)

    target_outputs = tuple(
        torch.randn(2, 4, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        for _ in range(3)
    )
    collapsed = tuple(
        DeepSeekV4MTPReplay.collapse_target_layer_output(value)
        for value in target_outputs
    )
    for source, tap in zip(target_outputs, collapsed):
        torch.testing.assert_close(tap, source.mean(dim=2))

    batch = DeepSeekV4MTPReplayBatch(
        target_taps=collapsed,
        anchor_token_ids=torch.tensor([7, 9]),
        main_position_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 5, 6]]),
        main_attention_mask=torch.tensor(
            [[True, True, True, True], [False, False, True, True]]
        ),
    )
    ffn_shapes = []
    result = replay.replay(
        batch,
        prepare_ffn=lambda block_index, _block, hidden, token_ids: ffn_shapes.append(
            (block_index, tuple(hidden.shape), tuple(token_ids.shape))
        ),
    )

    assert result.proposal_token_ids.tolist() == [
        [7, 31, 31, 31, 31],
        [9, 31, 31, 31, 31],
    ]
    assert result.proposal_position_ids.tolist() == [
        [4, 5, 6, 7, 8],
        [7, 8, 9, 10, 11],
    ]
    assert result.projected_main.shape == (2, 4, config.hidden_size)
    assert result.terminal_residual.shape == (
        2,
        5,
        config.hc_mult,
        config.hidden_size,
    )
    assert ffn_shapes == [
        (0, (2, 5, config.hidden_size), (2, 5)),
        (1, (2, 5, config.hidden_size), (2, 5)),
        (2, (2, 5, config.hidden_size), (2, 5)),
    ]
    assert len(result.routes) == 3
    for block_index, route in enumerate(result.routes):
        assert route.block_index == block_index
        assert route.logits.shape == (2, 5, config.n_routed_experts)
        assert route.weights.shape == (2, 5, config.num_experts_per_tok)
        assert route.indices.shape == (2, 5, config.num_experts_per_tok)
        assert route.logits.dtype is torch.float32
        assert route.weights.dtype is torch.float32

    projected_result = replay.replay(
        DeepSeekV4MTPReplayBatch(
            target_taps=None,
            projected_main=result.projected_main,
            anchor_token_ids=batch.anchor_token_ids,
            main_position_ids=batch.main_position_ids,
            main_attention_mask=batch.main_attention_mask,
        )
    )
    assert torch.equal(projected_result.terminal_residual, result.terminal_residual)
    for projected_route, tap_route in zip(projected_result.routes, result.routes):
        assert torch.equal(projected_route.indices, tap_route.indices)
        assert torch.equal(projected_route.weights, tap_route.weights)


def test_deepseek_v4_mtp_auxiliary_opens_without_target_shell(
    monkeypatch, tmp_path
) -> None:
    config = _tiny_v4_config()
    expected_keys = expected_deepseek_v4_mtp_checkpoint_keys(config)
    fake_turtle = SimpleNamespace(
        _weight_map=dict.fromkeys(expected_keys, "model.safetensors")
    )
    captured = {}

    from gptqmodel.utils.structure import LazyTurtle

    def maybe_create(**kwargs):
        captured.update(kwargs)
        return fake_turtle

    monkeypatch.setattr(LazyTurtle, "maybe_create", maybe_create)
    auxiliary = DeepSeekV4MTPAuxiliary.from_checkpoint(
        config=config,
        model_local_path=str(tmp_path),
    )

    assert isinstance(auxiliary.model, DeepSeekV4MTPAuxiliaryShell)
    assert auxiliary.turtle_model is fake_turtle
    assert auxiliary.checkpoint_contract == {
        "block_count": 3,
        "target_layer_ids": [0, 1, 2],
        "routed_experts_per_block": 3,
        "tensor_count": len(expected_keys),
    }
    assert captured["target_model"] is auxiliary.model
    assert captured["model_local_path"] == str(tmp_path)
    assert captured["module_tree"][0] == "mtp"


def test_deepseek_v4_mtp_quantization_adapter_builds_without_target_model() -> None:
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    embedding = torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    qcfg = EXL3Config(bits=3.0, device="cpu")

    adapter = DeepSeekV4MTPQuantizationModel.from_auxiliary(
        auxiliary=auxiliary,
        embedding_weight=embedding,
        quantize_config=qcfg,
        model_local_path="/tmp/deepseek-v4-test",
    )

    assert adapter.model is shell
    assert adapter.model_local_path == "/tmp/deepseek-v4-test"
    assert adapter.quantize_config.bits == 3.0
    assert qcfg.module_include is None
    assert adapter.quantize_config.module_is_included("mtp.1.mlp.experts.2.down_proj")
    assert not adapter.quantize_config.module_is_included(
        "mtp.1.mlp.shared_experts.down_proj"
    )


def test_deepseek_v4_mtp_post_quantize_preserves_lazy_terminal_passthroughs() -> None:
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    adapter = DeepSeekV4MTPQuantizationModel.from_auxiliary(
        auxiliary=auxiliary,
        embedding_weight=torch.randn(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        ),
        quantize_config=EXL3Config(bits=3.0, device="cpu"),
        model_local_path="/tmp/deepseek-v4-test",
    )
    terminal = shell.mtp[-1]
    passthrough = {
        name: terminal._modules[name].to(device="meta")
        for name in ("hc_head", "norm", "markov_head", "confidence_head")
    }
    tensor_passthrough = {}
    for path in adapter._PASSTHROUGH_TENSORS:
        owner_path, leaf = path.rsplit(".", 1)
        owner = terminal.get_submodule(owner_path)
        parameter = owner._parameters[leaf]
        expected = nn.Parameter(
            parameter.detach().to(device="meta"),
            requires_grad=parameter.requires_grad,
        )
        owner._parameters[leaf] = expected
        tensor_passthrough[path] = expected

    assert adapter.post_quantize(terminal) is terminal
    for name, expected in passthrough.items():
        assert terminal._modules[name] is expected
        assert all(
            tensor.is_meta for tensor in (*expected.parameters(), *expected.buffers())
        )
    for path, expected in tensor_passthrough.items():
        owner_path, leaf = path.rsplit(".", 1)
        assert terminal.get_submodule(owner_path)._parameters[leaf] is expected
    assert all(
        tensor.device.type == "cpu"
        for name, tensor in (
            tuple(terminal.named_parameters()) + tuple(terminal.named_buffers())
        )
        if name not in tensor_passthrough
        and not any(name == root or name.startswith(f"{root}.") for root in passthrough)
    )


def test_deepseek_v4_mtp_terminal_deferred_exl3_weight_can_finalize() -> None:
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    adapter = DeepSeekV4MTPQuantizationModel.from_auxiliary(
        auxiliary=auxiliary,
        embedding_weight=torch.randn(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        ),
        quantize_config=EXL3Config(bits=3.0, device="cpu"),
        model_local_path="/tmp/deepseek-v4-test",
    )
    terminal = shell.mtp[-1]
    for name in ("hc_head", "norm", "markov_head", "confidence_head"):
        terminal._modules[name].to(device="meta")

    linear = terminal.mlp.experts[0].gate_proj
    named = NamedModule(
        linear,
        "mlp.experts.0.gate_proj",
        "mtp.2.mlp.experts.0.gate_proj",
        2,
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    packed = {
        "trellis": torch.zeros(1),
        "suh": torch.zeros(1),
        "svh": torch.zeros(1),
    }
    named.state.update(packed)
    processor._stage_runtime_weight(
        module=named,
        out_tensors=packed,
        target_device=torch.device("cpu"),
    )
    processor.prepare_layer_post_quantize(
        model=adapter,
        layer_module=terminal,
        layer_index=2,
        processed_modules={named.name: named},
        is_lm_head_module=False,
    )

    assert adapter.post_quantize(terminal) is terminal
    assert linear.weight.device.type == "cpu"
    assert linear.weight.numel() == 0


def test_deepseek_v4_mtp_post_quantize_rejects_meta_block_body() -> None:
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    adapter = DeepSeekV4MTPQuantizationModel.from_auxiliary(
        auxiliary=auxiliary,
        embedding_weight=torch.randn(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        ),
        quantize_config=EXL3Config(bits=3.0, device="cpu"),
        model_local_path="/tmp/deepseek-v4-test",
    )
    terminal = shell.mtp[-1]
    terminal.self_attn.q_a_norm.weight = nn.Parameter(
        terminal.self_attn.q_a_norm.weight.detach().to(device="meta"),
        requires_grad=False,
    )

    with pytest.raises(RuntimeError, match="unexpected meta tensors"):
        adapter.post_quantize(terminal)


def test_deepseek_v4_mtp_quantization_adapter_replays_one_exact_block() -> None:
    torch.manual_seed(0xD55)
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
        for block in shell.mtp:
            block.mlp.gate.e_score_correction_bias.zero_()
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    embedding = torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    target = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(target)
    target.quantize_config = EXL3Config(bits=2.0, device="cpu")
    target.qlinear_kernel = None
    target.trust_remote_code = False
    target.model_local_path = "/tmp/deepseek-v4-test"

    adapter = DeepSeekV4MTPQuantizationModel.from_target_model(
        target,
        auxiliary=auxiliary,
        embedding_weight=embedding,
    )
    assert target.quantize_config.module_include is None
    assert adapter.quantize_config.module_is_included("mtp.2.mlp.experts.2.down_proj")
    assert not adapter.quantize_config.module_is_included(
        "mtp.2.mlp.shared_experts.down_proj"
    )
    assert adapter.extract_layers_node() == ["mtp"]
    modules = {
        name
        for block in adapter.simple_layer_modules(
            model_config=config,
            quantize_config=adapter.quantize_config,
        )
        for name in block
    }
    assert "mlp.experts.2.gate_proj" in modules
    assert "mlp.experts.2.up_proj" in modules
    assert "mlp.experts.2.down_proj" in modules
    assert not any(name.startswith("self_attn.") for name in modules)
    assert not any(name.startswith("mlp.shared_experts.") for name in modules)

    target.model = SimpleNamespace(config=config)
    target.quant_log = []
    adapter.quantized = True
    adapter.quant_log = [
        {
            "exl3_error_ledger_record": {
                "block_namespace": "mtp",
                "module": (f"mtp.{block}.mlp.experts.{expert}.{projection}"),
            }
        }
        for block in range(3)
        for expert in range(config.n_routed_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    target.attach_mtp_quantization_model(adapter)
    assert target._mtp_quantization_model_for_save is adapter
    assert len(target.quant_log) == 27

    batch = DeepSeekV4MTPReplayBatch(
        target_taps=None,
        projected_main=torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16),
        anchor_token_ids=torch.tensor([3, 7]),
        main_position_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 5, 6]]),
        main_attention_mask=torch.tensor(
            [[True, True, True, True], [False, False, True, True]]
        ),
    )
    prepared = adapter.prepare_dataset(
        [batch], calibration_dataset_sort=None, batch_size=1
    )
    assert prepared[0]["input_ids"].shape == (2, 5)
    assert prepared[0]["attention_mask"].all()

    class LazyReplay(Sequence):
        row_counts = [2]
        gptqmodel_calibration_summary = {
            "batch_count": 1,
            "input_ids_total_length": 10,
            "input_ids_max_length": 5,
            "total_calibration_tokens": 10,
        }

        def __init__(self):
            self.reads = 0

        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            self.reads += 1
            return batch

    lazy = LazyReplay()
    lazy_prepared = adapter.prepare_dataset(
        lazy, calibration_dataset_sort=None, batch_size=1
    )
    assert lazy.reads == 0
    lazy_cache = adapter.build_quantization_input_cache(
        layers=list(adapter.model.mtp),
        layer_names=[f"mtp.{index}" for index in range(3)],
        calibration_data=lazy_prepared,
        use_cache=False,
    )
    assert lazy.reads == 0
    lazy_inputs = lazy_cache.layer_inputs[0]
    lazy_kwargs = lazy_cache.layer_input_kwargs[0]
    lazy_positions = lazy_cache.position_ids[0]
    lazy_mask = lazy_cache.attention_masks[0]
    assert lazy.reads == 1
    assert lazy_cache.layer_inputs.row_counts == [2]

    reference = DeepSeekV4MTPReplay(shell, embedding_weight=embedding)
    state = reference.prepare_batch(batch)
    torch.testing.assert_close(lazy_inputs[0], state.residual)
    assert torch.equal(lazy_positions, state.proposal_position_ids)
    assert torch.equal(lazy_mask, prepared[0]["attention_mask"])
    assert torch.equal(
        lazy_kwargs["_gptqmodel_mtp_projected_main"], state.projected_main
    )
    expected, _ = reference.replay_block(0, state, capture_route=False)
    actual = adapter.run_input_capture(
        prepared[0], use_cache=False, data_device=torch.device("cpu")
    )
    torch.testing.assert_close(actual, expected)

    looper = SimpleNamespace(
        gptq_model=adapter,
        _batch_row_count=lambda value: int(value[0].shape[0]),
    )
    cache = StageInputsCapture(looper).cache_inputs(
        layers=list(adapter.model.mtp),
        layer_names=[f"mtp.{index}" for index in range(3)],
        calibration_data=prepared,
        use_cache=False,
    )
    assert len(cache.layer_inputs) == 1
    torch.testing.assert_close(cache.layer_inputs[0][0], state.residual)
    assert torch.equal(cache.position_ids[0], state.proposal_position_ids)
    assert torch.equal(cache.attention_masks[0], prepared[0]["attention_mask"])
    assert torch.equal(
        cache.layer_input_kwargs[0]["_gptqmodel_mtp_projected_main"],
        state.projected_main,
    )
    assert torch.equal(
        cache.layer_input_kwargs[0]["_gptqmodel_mtp_proposal_token_ids"],
        state.proposal_token_ids,
    )
    replay_kwargs = dict(cache.layer_input_kwargs[0])
    routes = []
    handles = [
        block.mlp.gate.register_forward_hook(
            lambda _module, _args, output: routes.append(output)
        )
        for block in adapter.model.mtp
    ]
    try:
        residual = cache.layer_inputs[0][0]
        for block in adapter.model.mtp:
            residual = block(
                residual,
                attention_mask=cache.attention_masks[0],
                position_ids=cache.position_ids[0],
                **replay_kwargs,
            )
    finally:
        for handle in handles:
            handle.remove()
    full_reference = reference.replay(batch)
    torch.testing.assert_close(residual, full_reference.terminal_residual)
    assert len(routes) == 3
    for output in routes:
        _, _, indices = output
        assert indices.shape == (2 * 5, config.num_experts_per_tok)


class _RecoveryExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False)
        self.up_proj = nn.Linear(4, 3, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)


class _RecoveryRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 4
        self.top_k = 6
        self.weight = nn.Parameter(torch.zeros(13, 4))
        self.register_buffer(
            "e_score_correction_bias",
            torch.arange(13, 0, -1, dtype=torch.float32),
        )
        self.score_fn = torch.sigmoid

    def forward(self, hidden_states: torch.Tensor):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(
            scores + self.e_score_correction_bias,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        weights = scores.gather(1, indices)
        return logits, weights, indices


class _RecoveryMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _RecoveryRouter()
        self.experts = nn.ModuleList([_RecoveryExpert() for _ in range(13)])
        self.act_fn = F.silu
        self.limit = 2.5


def _recovery_processor(task_names: tuple[str, ...], natural_count: int):
    return SimpleNamespace(
        tasks={
            task_name: {"route_evidence": {"expert_route_count": natural_count}}
            for task_name in task_names
        }
    )


def _recovery_capture(columns: int, natural_count: int):
    device = torch.device("cpu")
    return SimpleNamespace(
        columns=columns,
        nsamples=natural_count,
        H=None,
        _device_hessian_partials={
            device: torch.eye(columns, dtype=torch.float32).mul(natural_count)
        },
        _device_sample_counts={device: natural_count},
        _hessian_dirty=True,
        _final_hessian_device_hint=device,
    )


def _capture_linear_input(capture):
    def hook(_module, inputs, _output):
        values = inputs[0].reshape(-1, capture.columns).float()
        device = next(iter(capture._device_hessian_partials))
        capture._device_hessian_partials[device].add_(values.T @ values)
        count = int(values.shape[0])
        capture._device_sample_counts[device] += count
        capture.nsamples += count

    return hook


def test_deepseek_v4_recovery_uses_rank_seven_for_declared_expert_only() -> None:
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    block._gptqmodel_mtp_block_index = 0
    shell = nn.Module()
    shell.mtp = nn.ModuleList([block])
    adapter = object.__new__(DeepSeekV4MTPQuantizationModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        "mlp.experts.6.gate_proj": SimpleNamespace(
            full_name="mtp.0.mlp.experts.6.gate_proj"
        ),
        "mlp.experts.6.up_proj": SimpleNamespace(
            full_name="mtp.0.mlp.experts.6.up_proj"
        ),
    }
    task_names = tuple(sorted(subset))
    processor = _recovery_processor(task_names, natural_count=0)
    calls = []
    handles = []
    for expert_index, expert in enumerate(block.mlp.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            handles.append(
                getattr(expert, projection).register_forward_hook(
                    lambda _module,
                    _args,
                    _output,
                    key=(expert_index, projection): calls.append(key)
                )
            )
    pause_events = []
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, value: pause_events.append(value)
    )
    try:
        with adapter.zero_route_recovery_context(
            looper=looper,
            processor=processor,
            layer_module=block,
            subset=subset,
            task_names=task_names,
        ):
            ffn_input = torch.randn(1, 1024, 4)
            block._gptqmodel_mtp_zero_route_force(
                0,
                block,
                ffn_input,
                torch.zeros(1, 1024, dtype=torch.long),
            )
            block.mlp.gate(ffn_input)
            with block._gptqmodel_mtp_normal_mlp_capture_context():
                pass
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [(6, "gate_proj"), (6, "up_proj")]
    assert pause_events == [True, False]
    assert {
        task["zero_route_recovery_capture"]["candidate_rows_selected"]
        for task in processor.tasks.values()
    } == {1024}
    assert {
        task["zero_route_recovery_capture"]["candidate_rank_histogram"]["7"]
        for task in processor.tasks.values()
    } == {1024}
    assert not hasattr(block, "_gptqmodel_mtp_zero_route_force")
    assert not hasattr(block, "_gptqmodel_mtp_normal_mlp_capture_context")


def test_deepseek_v4_target_learned_router_uses_same_recovery_policy() -> None:
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    inner = nn.Module()
    inner.layers = nn.ModuleList([block])
    shell = nn.Module()
    shell.model = inner
    adapter = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        "mlp.experts.6.gate_proj": SimpleNamespace(
            full_name="model.layers.0.mlp.experts.6.gate_proj"
        ),
        "mlp.experts.6.up_proj": SimpleNamespace(
            full_name="model.layers.0.mlp.experts.6.up_proj"
        ),
    }
    task_names = tuple(sorted(subset))
    processor = _recovery_processor(task_names, natural_count=1014)
    processor._mask_tls = SimpleNamespace(value=torch.tensor([True] * 10 + [False] * 2))
    calls = []
    handles = []
    expert = block.mlp.experts[6]
    for projection in ("gate_proj", "up_proj"):
        handles.append(
            getattr(expert, projection).register_forward_hook(
                lambda _module, _args, _output, key=projection: calls.append(key)
            )
        )
    pause_events = []
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, value: pause_events.append(value)
    )
    try:
        with adapter.zero_route_recovery_context(
            looper=looper,
            processor=processor,
            layer_module=block,
            subset=subset,
            task_names=task_names,
        ):
            block.mlp.gate(torch.randn(1, 12, 4))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == ["gate_proj", "up_proj"]
    assert pause_events == [True, False]
    assert {
        task["zero_route_recovery_capture"]["candidate_rows_observed"]
        for task in processor.tasks.values()
    } == {10}
    assert {
        task["zero_route_recovery_capture"]["candidate_rows_selected"]
        for task in processor.tasks.values()
    } == {10}


def test_deepseek_v4_recovery_replays_rank_rows_from_capture_spool() -> None:
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    inner = nn.Module()
    inner.layers = nn.ModuleList([block])
    shell = nn.Module()
    shell.model = inner
    adapter = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        f"mlp.experts.6.{projection}": SimpleNamespace(
            full_name=f"model.layers.0.mlp.experts.6.{projection}"
        )
        for projection in ("gate_proj", "up_proj")
    }
    task_names = tuple(sorted(subset))
    processor = _recovery_processor(task_names, natural_count=1014)
    router_input = torch.randn(10, 4)
    candidate_indices = torch.full((10, 6), 12, dtype=torch.int64)
    candidate_indices[:, 0] = 6
    candidate_gaps = torch.arange(60, dtype=torch.float32).reshape(10, 6)

    class _Spool:
        phase = "gate-up"
        key = {"expected_batches": 1}
        committed_indices = frozenset({0})

        @staticmethod
        def load(batch_index):
            assert batch_index == 0
            return {
                "router_input": router_input,
                "candidate_indices": candidate_indices,
                "candidate_score_gaps": candidate_gaps,
            }, {}

    processor._active_capture_batch_spool = _Spool()
    calls = []
    handles = []
    expert = block.mlp.experts[6]
    handles.append(
        expert.gate_proj.register_forward_hook(
            lambda _module, _args, _output: calls.append("gate_proj")
        )
    )
    expert.up_proj = HookedLinear.from_linear(expert.up_proj)
    expert.up_proj.forward_hook = lambda _module, _args, _output: calls.append(
        "up_proj"
    )
    expert.up_proj.forward_hook_last = True
    try:
        with adapter.zero_route_recovery_context(
            looper=SimpleNamespace(_set_processor_hooks_paused=lambda *_args: None),
            processor=processor,
            layer_module=block,
            subset=subset,
            task_names=task_names,
        ) as replay_required:
            assert replay_required is False
    finally:
        for handle in handles:
            handle.remove()

    assert calls == ["gate_proj", "up_proj"]
    assert expert.up_proj.forward_hook_last is True
    assert {
        task["zero_route_recovery_capture"]["candidate_rows_selected"]
        for task in processor.tasks.values()
    } == {10}
    assert {
        task["zero_route_recovery_capture"]["candidate_rank_histogram"]["7"]
        for task in processor.tasks.values()
    } == {10}


def test_deepseek_v4_recovery_uses_identity_only_for_rank_shortfall() -> None:
    torch.manual_seed(0xD54)
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    block._gptqmodel_mtp_block_index = 0
    shell = nn.Module()
    shell.mtp = nn.ModuleList([block])
    adapter = object.__new__(DeepSeekV4MTPQuantizationModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        f"mlp.experts.6.{projection}": SimpleNamespace(
            full_name=f"mtp.0.mlp.experts.6.{projection}"
        )
        for projection in ("gate_proj", "up_proj")
    }
    task_names = tuple(sorted(subset))
    natural_count = 300
    processor = _recovery_processor(task_names, natural_count=natural_count)
    captures = {}
    handles = []
    expert = block.mlp.experts[6]
    for task_name in task_names:
        projection = task_name.rsplit(".", 1)[-1]
        capture = _recovery_capture(4, natural_count)
        capture.columns = np.int64(capture.columns)
        captures[projection] = capture
        processor.tasks[task_name]["capture"] = capture
        handles.append(
            getattr(expert, projection).register_forward_hook(
                _capture_linear_input(capture)
            )
        )
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, _value: None
    )
    candidates = torch.randn(1, 381, 4)
    try:
        with adapter.zero_route_recovery_context(
            looper=looper,
            processor=processor,
            layer_module=block,
            subset=subset,
            task_names=task_names,
        ):
            block._gptqmodel_mtp_zero_route_force(
                0,
                block,
                candidates,
                torch.zeros(1, 381, dtype=torch.long),
            )
            block.mlp.gate(candidates)
    finally:
        for handle in handles:
            handle.remove()

    identity_count = 1024 - natural_count - 381
    expected = (
        torch.eye(4, dtype=torch.float32).mul(natural_count + identity_count)
        + candidates.reshape(-1, 4).float().T @ candidates.reshape(-1, 4).float()
    )
    for capture in captures.values():
        assert capture.nsamples == 1024
        assert sum(capture._device_sample_counts.values()) == 1024
        torch.testing.assert_close(
            next(iter(capture._device_hessian_partials.values())),
            expected,
        )
    summaries = {
        task["zero_route_recovery_capture"]["recovery_mode"]
        for task in processor.tasks.values()
    }
    assert summaries == {"empirical-plus-identity-hessian"}
    assert {
        task["zero_route_recovery_capture"]["router_augmented_sample_count"]
        for task in processor.tasks.values()
    } == {381}
    assert {
        task["zero_route_recovery_capture"]["identity_calibration_count"]
        for task in processor.tasks.values()
    } == {identity_count}


def test_deepseek_v4_positive_route_without_near_rows_uses_identity_residual() -> None:
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    block._gptqmodel_mtp_block_index = 0
    shell = nn.Module()
    shell.mtp = nn.ModuleList([block])
    adapter = object.__new__(DeepSeekV4MTPQuantizationModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    task_name = "mlp.experts.12.gate_proj"
    subset = {task_name: SimpleNamespace(full_name="mtp.0.mlp.experts.12.gate_proj")}
    natural_count = 13
    capture = _recovery_capture(4, natural_count)
    processor = _recovery_processor((task_name,), natural_count=natural_count)
    processor.tasks[task_name]["capture"] = capture
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, _value: None
    )
    ffn_input = torch.randn(1, 32, 4)
    with adapter.zero_route_recovery_context(
        looper=looper,
        processor=processor,
        layer_module=block,
        subset=subset,
        task_names=(task_name,),
    ):
        block._gptqmodel_mtp_zero_route_force(
            0,
            block,
            ffn_input,
            torch.zeros(1, 32, dtype=torch.long),
        )
        block.mlp.gate(ffn_input)

    assert capture.nsamples == 1024
    torch.testing.assert_close(
        next(iter(capture._device_hessian_partials.values())),
        torch.eye(4, dtype=torch.float32).mul(1024),
    )
    summary = processor.tasks[task_name]["zero_route_recovery_capture"]
    assert summary["recovery_mode"] == "empirical-plus-identity-hessian"
    assert summary["router_augmented_sample_count"] == 0
    assert summary["identity_calibration_count"] == 1011
    assert summary["candidate_score_gap"] is None


def test_deepseek_v4_zero_route_down_recovery_uses_native_swiglu_input() -> None:
    torch.manual_seed(0xD0A4)
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    block.mlp.act_fn = torch.tanh
    block.mlp.limit = 0.25
    block._gptqmodel_mtp_block_index = 0
    shell = nn.Module()
    shell.mtp = nn.ModuleList([block])
    adapter = object.__new__(DeepSeekV4MTPQuantizationModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        "mlp.experts.6.down_proj": SimpleNamespace(
            full_name="mtp.0.mlp.experts.6.down_proj"
        )
    }
    expert_input = torch.randn(2, 5, 4)
    expert = block.mlp.experts[6]
    expected = torch.tanh(
        expert.gate_proj(expert_input.reshape(-1, 4)).clamp(max=0.25)
    ) * expert.up_proj(expert_input.reshape(-1, 4)).clamp(min=-0.25, max=0.25)
    observed = []
    handle = expert.down_proj.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, _value: None
    )
    task_names = ("mlp.experts.6.down_proj",)
    processor = _recovery_processor(task_names, natural_count=1014)
    try:
        with adapter.zero_route_recovery_context(
            looper=looper,
            processor=processor,
            layer_module=block,
            subset=subset,
            task_names=task_names,
        ):
            block._gptqmodel_mtp_zero_route_force(
                0,
                block,
                expert_input,
                torch.zeros(2, 5, dtype=torch.long),
            )
            block.mlp.gate(expert_input)
    finally:
        handle.remove()

    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected)


def test_deepseek_v4_true_zero_without_near_rows_uses_identity_hessian() -> None:
    block = nn.Module()
    block.mlp = _RecoveryMLP()
    block._gptqmodel_mtp_block_index = 0
    shell = nn.Module()
    shell.mtp = nn.ModuleList([block])
    adapter = object.__new__(DeepSeekV4MTPQuantizationModel)
    nn.Module.__init__(adapter)
    adapter.model = shell
    subset = {
        f"mlp.experts.12.{projection}": SimpleNamespace(
            full_name=f"mtp.0.mlp.experts.12.{projection}"
        )
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    task_names = tuple(sorted(subset))
    columns = {"gate_proj": 4, "up_proj": 4, "down_proj": 3}
    processor = SimpleNamespace(tasks={})
    for task_name in task_names:
        projection = task_name.rsplit(".", 1)[-1]
        processor.tasks[task_name] = {
            "route_evidence": {"expert_route_count": 0},
            "capture": SimpleNamespace(
                columns=columns[projection],
                nsamples=0,
                H=None,
                _device_hessian_partials={},
                _device_sample_counts={},
                _hessian_dirty=False,
                _final_hessian_device_hint=torch.device("cpu"),
            ),
        }
    looper = SimpleNamespace(
        _set_processor_hooks_paused=lambda _processor, _value: None
    )
    ffn_input = torch.randn(1, 1024, 4)
    with adapter.zero_route_recovery_context(
        looper=looper,
        processor=processor,
        layer_module=block,
        subset=subset,
        task_names=task_names,
    ):
        block._gptqmodel_mtp_zero_route_force(
            0,
            block,
            ffn_input,
            torch.zeros(1, 1024, dtype=torch.long),
        )
        block.mlp.gate(ffn_input)

    for task_name, task in processor.tasks.items():
        capture = task["capture"]
        assert capture.nsamples == 1024
        torch.testing.assert_close(
            capture.H,
            torch.eye(capture.columns, dtype=torch.float32) * 2.0,
        )
        assert task["zero_route_recovery_capture"] == {
            "recovery_mode": "identity-hessian",
            "router_augmented_sample_count": 0,
            "identity_calibration_count": 1024,
            "candidate_rows_observed": 0,
            "candidate_rows_selected": 0,
            "candidate_rank_histogram": {
                "7": 0,
                "8": 0,
                "9": 0,
                "10": 0,
                "11": 0,
                "12": 0,
            },
            "candidate_score_gap": None,
        }


def test_deepseek_v4_target_anchor_resolver_matches_native_fp32_greedy_head() -> None:
    torch.manual_seed(0xA4C40)
    config = _tiny_v4_config()
    hc_head = DeepseekV4HyperHead(config).to(dtype=torch.bfloat16)
    norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
        dtype=torch.bfloat16
    )
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(
        dtype=torch.bfloat16
    )
    resolver = DeepSeekV4TargetAnchorResolver(
        hc_head=hc_head,
        norm=norm,
        lm_head=lm_head,
        position_chunk_size=2,
        vocab_chunk_size=7,
    )
    raw = torch.randn(2, 4, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.arange(8).reshape(2, 4)
    attention_mask = torch.tensor(
        [[False, True, True, True], [True, True, True, False]]
    )
    decode_mask = torch.tensor([[False, True, False, True], [True, False, True, True]])
    position_ids = torch.tensor([[0, 3, 4, 5], [8, 9, 10, 0]])

    anchors = resolver(
        raw,
        input_ids,
        attention_mask,
        decode_mask,
        position_ids,
    )
    eligible = attention_mask & decode_mask
    selected = raw[eligible].unsqueeze(0)
    reference_hidden = norm(hc_head(selected)).squeeze(0)
    reference = F.linear(reference_hidden.float(), lm_head.weight.float()).argmax(-1)
    assert torch.equal(anchors[eligible], reference)
    assert torch.equal(anchors[~eligible], torch.full_like(anchors[~eligible], -1))


def test_deepseek_v4_target_tap_sink_receives_only_official_lane_means() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    events = []

    assert not harness.quantization_layer_output_required(
        layer_index=2, layer_name="model.layers.2", layer_count=3
    )
    harness.set_mtp_target_tap_sink(events.append)
    assert harness.quantization_layer_output_required(
        layer_index=2, layer_name="model.layers.2", layer_count=3
    )
    assert not harness.quantization_layer_output_required(
        layer_index=3, layer_name="model.layers.3", layer_count=3
    )

    raw = torch.randn(2, 5, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    harness.receive_quantization_layer_outputs(
        layer_index=2,
        layer_name="model.layers.2",
        layer_outputs=[[raw]],
        layer_input_kwargs=[{"input_ids": torch.arange(10).reshape(2, 5)}],
        position_ids=[positions],
        attention_masks=[mask],
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeepSeekV4MTPTargetTapEvent)
    assert event.layer_index == 2
    assert event.layer_name == "model.layers.2"
    assert event.raw_layer_outputs[0] is raw
    assert event.position_ids[0] is positions
    assert event.attention_masks[0] is mask
    torch.testing.assert_close(event.collapsed_target_taps[0], raw.mean(dim=2))


def test_deepseek_v4_mtp_output_boundary_prunes_superseded_capture(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture_root = tmp_path / "capture"
    family_join = {"recipe": "test"}
    capture_store = EXL3CaptureFrontierStore(
        capture_root,
        family_join=family_join,
    )
    for layer_index in (0, 1):
        module = f"mtp.{layer_index}.mlp.experts.0.gate_proj"
        subset = {
            "mlp.experts.0.gate_proj": SimpleNamespace(full_name=module),
        }
        capture_store.commit(
            layer_index=layer_index,
            subset_index=0,
            subset_total=1,
            subset=subset,
            states=[
                EXL3CaptureState(
                    module=module,
                    hessian=torch.eye(2, dtype=torch.float32),
                    sample_count=1,
                    route_evidence=None,
                )
            ],
        )
    layer_zero = next(capture_root.glob("layer-000000-*"))
    layer_one = next(capture_root.glob("layer-000001-*"))
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(capture_root))

    harness = object.__new__(DeepSeekV4MTPQuantizationModel)
    harness.configure_mtp_activation_store(
        str(tmp_path / "activations"),
        provenance={
            "plan_sha256": "a" * 64,
            "family_join": family_join,
        },
    )
    writer = harness.create_quantization_layer_output_writer(
        layer_index=0,
        expected_batches=1,
        progress_stage="Forward replay",
        apply_moe_config=False,
    )
    assert writer is not None
    writer.put(0, [torch.zeros(1, 2)])
    writer.finalize()

    assert not layer_zero.exists()
    assert layer_one.is_dir()


def test_deepseek_v4_target_tap_sink_rejects_uncollapsible_output() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    harness.set_mtp_target_tap_sink(lambda _event: None)
    try:
        harness.receive_quantization_layer_outputs(
            layer_index=2,
            layer_name="model.layers.2",
            layer_outputs=[[torch.zeros(2, 5, config.hidden_size)]],
            layer_input_kwargs=[{}],
            position_ids=[],
            attention_masks=[],
        )
    except RuntimeError as exc:
        assert "[batch, sequence, hc, hidden]" in str(exc)
    else:
        raise AssertionError("a rank-3 target boundary was accepted as a raw mHC tap")


def test_deepseek_v4_preserves_original_token_mask_metadata_without_forwarding_it() -> (
    None
):
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[0, 1, 1]])
    labels = torch.tensor([[-100, -100, 3]])
    harness.begin_input_capture_example(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        batch_device=torch.device("cpu"),
    )
    captured = harness.capture_first_layer_input_kwargs(
        args=(),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={"ordinary": torch.tensor(1)},
    )
    harness.end_input_capture_example()

    assert torch.equal(captured[MTP_CAPTURE_INPUT_IDS], input_ids)
    assert torch.equal(captured[MTP_CAPTURE_ATTENTION_MASK], attention_mask)
    assert captured[MTP_CAPTURE_DECODE_MASK].tolist() == [[False, True, False]]
    replay_kwargs = harness.prepare_layer_replay_kwargs(
        layer=nn.Identity(),
        layer_input=[torch.zeros(1, 3, config.hidden_size)],
        additional_inputs=dict(captured),
        target_device=torch.device("cpu"),
    )
    assert MTP_CAPTURE_INPUT_IDS not in replay_kwargs
    assert MTP_CAPTURE_ATTENTION_MASK not in replay_kwargs
    assert MTP_CAPTURE_DECODE_MASK not in replay_kwargs
    assert "ordinary" in replay_kwargs


def test_deepseek_v4_raw_text_calibration_marks_causal_replay_positions() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    harness.begin_input_capture_example(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        batch_device=torch.device("cpu"),
    )
    captured = harness.capture_first_layer_input_kwargs(
        args=(),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )
    harness.end_input_capture_example()

    assert captured[MTP_CAPTURE_DECODE_MASK].tolist() == [[True, True, False, False]]


class _LearnedRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 4
        self.top_k = 2
        self.routed_scaling_factor = 1.5
        self.score_fn = nn.Softplus()
        self.weight = nn.Parameter(
            torch.tensor(
                [[1, 2, 3, 4], [4, 3, 2, 1], [1, -1, 1, -1]],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.register_buffer("e_score_correction_bias", torch.tensor([0.1, -0.2, 0.3]))


class _HashRouter(_LearnedRouter):
    def __init__(self) -> None:
        super().__init__()
        del self.e_score_correction_bias
        self.register_buffer("tid2eid", torch.tensor([[0, 2], [1, 0]]))


class _Mlp(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate


class _Layer(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.mlp = _Mlp(gate)


class _RouterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Layer(_HashRouter()), _Layer(_LearnedRouter())]
        )


def test_deepseek_v4_router_patch_uses_fp32_without_promoting_stored_weights() -> None:
    model = _RouterModel()
    assert patch_deepseek_v4_router_precision(model) == 2
    assert patch_deepseek_v4_router_precision(model) == 2
    hidden = torch.tensor([[1.0, 0.5, -0.25, 2.0]], dtype=torch.bfloat16)

    learned = model.model.layers[1].mlp.gate
    logits, weights, indices = learned(hidden)
    expected_logits = F.linear(hidden.float(), learned.weight.float())
    expected_scores = learned.score_fn(expected_logits)
    expected_indices = torch.topk(
        expected_scores + learned.e_score_correction_bias,
        2,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = expected_scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(weights, expected_weights * 1.5)
    assert torch.equal(indices, expected_indices)
    assert learned.weight.dtype is torch.bfloat16
    assert logits.dtype is torch.float32
    assert weights.dtype is torch.float32

    hashed = model.model.layers[0].mlp.gate
    _, hash_weights, hash_indices = hashed(hidden, torch.tensor([[1]]))
    assert hash_indices.tolist() == [[1, 0]]
    assert hash_weights.dtype is torch.float32


def test_deepseek_v4_after_load_preserves_source_fp32_and_requires_router_coverage() -> (
    None
):
    model = DeepseekV4ForCausalLM(_tiny_v4_config()).to(dtype=torch.bfloat16)
    harness = object.__new__(DeepSeekV4QModel)
    assert DeepSeekV4QModel.after_model_load(harness, model) is model

    assert model.model.hc_head.hc_fn.dtype is torch.float32
    assert model.model.hc_head.hc_base.dtype is torch.float32
    assert model.model.hc_head.hc_scale.dtype is torch.float32
    assert model.model.norm.weight.dtype is torch.bfloat16
    assert model.lm_head.weight.dtype is torch.bfloat16
    for layer in model.model.layers:
        assert layer.attn_hc.fn.dtype is torch.float32
        assert layer.attn_hc.base.dtype is torch.float32
        assert layer.attn_hc.scale.dtype is torch.float32
        assert layer.ffn_hc.fn.dtype is torch.float32
        assert layer.ffn_hc.base.dtype is torch.float32
        assert layer.ffn_hc.scale.dtype is torch.float32
        assert layer.self_attn.sinks.dtype is torch.float32

    model.model.layers[-1].mlp.gate = nn.Identity()
    try:
        DeepSeekV4QModel.after_model_load(harness, model)
    except RuntimeError as exc:
        assert "router" in str(exc)
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing DeepSeek V4 router coverage was accepted")


def test_deepseek_v4_prefix_runtime_owns_exact_projector_anchor_and_embedding(
    monkeypatch,
) -> None:
    torch.manual_seed(0xA4C41)
    config = _tiny_v4_config()
    model = DeepseekV4ForCausalLM(config).to(dtype=torch.bfloat16)
    harness = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(harness)
    harness.model = DeepSeekV4QModel.after_model_load(harness, model)

    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )

    from gptqmodel.utils.structure import LazyTurtle

    harness.turtle_model = object.__new__(LazyTurtle)
    mtp_materialized = []
    target_materialized = []
    monkeypatch.setattr(
        harness,
        "build_mtp_auxiliary",
        lambda *, device="meta": auxiliary,
    )
    monkeypatch.setattr(
        harness,
        "materialize_mtp_replay_submodule",
        lambda _auxiliary, module, **_kwargs: mtp_materialized.append(module) or module,
    )
    monkeypatch.setattr(
        harness,
        "shell_module_materialize",
        lambda *, target_submodule, module_path, **_kwargs: (
            target_materialized.append(module_path) or target_submodule
        ),
    )

    runtime = harness.build_mtp_prefix_runtime(
        device="cpu",
        position_chunk_size=2,
        vocab_chunk_size=7,
    )
    assert isinstance(runtime, DeepSeekV4MTPPrefixRuntime)
    assert mtp_materialized == [shell.mtp[0].main_proj, shell.mtp[0].main_norm]
    assert target_materialized == [
        "model.hc_head",
        "model.norm",
        "lm_head",
        "model.embed_tokens",
    ]

    taps = tuple(
        torch.randn(2, 3, config.hidden_size, dtype=torch.bfloat16) for _ in range(3)
    )
    expected = shell.mtp[0].main_norm(shell.mtp[0].main_proj(torch.cat(taps, -1)))
    torch.testing.assert_close(runtime.project_target_taps(taps), expected)
    assert runtime.build_replay().embedding_weight is runtime.target_embedding.weight
    assert runtime.target_hc_head is not model.model.hc_head
    assert runtime.target_norm is not model.model.norm
    assert runtime.target_lm_head is not model.lm_head
    assert runtime.target_embedding is not model.model.embed_tokens

    for target in (
        model.model.hc_head,
        model.model.norm,
        model.lm_head,
        model.model.embed_tokens,
    ):
        target.to(device="meta")

    raw = torch.randn(2, 3, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.arange(6).reshape(2, 3)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    decode_mask = torch.tensor([[False, True, True], [True, False, True]])
    position_ids = torch.arange(3).expand(2, -1)
    anchors = runtime.anchor_resolver(
        raw, input_ids, attention_mask, decode_mask, position_ids
    )
    assert torch.all(anchors[decode_mask] >= 0)
    assert torch.all(anchors[~decode_mask] == -1)
