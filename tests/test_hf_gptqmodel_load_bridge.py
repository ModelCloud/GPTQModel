# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native GPTQModel loading bridge used by Transformers/Optimum."""

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, GPTQConfig
from transformers.models.laguna.configuration_laguna import LagunaConfig

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import (
    HFGPTQModelLoadContext,
    hf_gptqmodel_post_init_for_load,
    hf_gptqmodel_prepare_model_for_load,
)
from gptqmodel.utils.nf12_codec import encode as encode_nf12


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 32, bias=False, device="meta")
        self.g_proj = torch.nn.Linear(128, 48, bias=False, device="meta")


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Model(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            _name_or_path=str(checkpoint_path),
            dtype=torch.float16,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block()])


def _write_checkpoint(checkpoint_path, *, native=True, missing_module=False):
    quantizer = ["gptqmodel:test"] if native else ["optimum:test"]
    dynamic = None
    if native:
        dynamic = {
            r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 4, "group_size": 32},
            r"+:^model\.layers\.0\.self_attn\.g_proj$": {"bits": 4, "group_size": 32},
        }
    config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "format": "gptq",
        "quant_method": "gptq",
        "dynamic": dynamic,
        "meta": {"quantizer": quantizer},
    }
    (checkpoint_path / "quantize_config.json").write_text(json.dumps(config), encoding="utf-8")
    module_name = (
        "model.layers.0.self_attn.missing_proj" if missing_module else "model.layers.0.self_attn.q_proj"
    )
    save_file(
        {f"{module_name}.qweight": torch.zeros((16, 32), dtype=torch.int32)},
        checkpoint_path / "model.safetensors",
    )


def test_hf_load_bridge_uses_checkpoint_manifest_and_preserves_dense_module(tmp_path):
    _write_checkpoint(tmp_path)
    model = _Model(tmp_path)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    q_proj = model.model.layers[0].self_attn.q_proj
    g_proj = model.model.layers[0].self_attn.g_proj
    assert isinstance(q_proj, TorchLinear)
    assert q_proj.group_size == 32
    assert q_proj.qweight.device.type == "meta"
    # The manifest keeps g_proj dense even though a dynamic rule names it.
    assert isinstance(g_proj, torch.nn.Linear)
    assert g_proj.out_features == 48
    assert context.quantize_config.dynamic is not None


def test_hf_load_bridge_resolves_mixed_case_model_type_via_casefold(tmp_path, monkeypatch):
    """A published mixed-case model_type must select the registered definition."""

    import defuser

    from gptqmodel.models.definitions.rw import RwgQModel

    _write_checkpoint(tmp_path)
    model = _Model(tmp_path)
    model.config.model_type = "RefinedWeb"

    converted = []
    monkeypatch.setattr(defuser, "convert_model", lambda module, **kwargs: converted.append(module))
    # The declared defuser paths are the observable difference between the
    # registered RwgQModel and the BaseQModel fallback for this model type.
    monkeypatch.setattr(RwgQModel, "defuser_module_paths", ("model.layers.0",), raising=False)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    assert context is not None
    assert len(converted) == 2


def test_hf_load_bridge_sizes_nf12_escape_buffer_before_weight_assignment(tmp_path):
    """Checkpoint escape rows must fit before Transformers assigns its tensors."""

    config = {
        "bits": 12,
        "group_size": -1,
        "desc_act": False,
        "sym": True,
        "format": "nf12",
        "quant_method": "nf12",
        "meta": {"quantizer": ["gptqmodel:test"]},
    }
    (tmp_path / "quantize_config.json").write_text(json.dumps(config), encoding="utf-8")
    weight = torch.linspace(-1, 1, 32 * 128).reshape(32, 128).to(torch.bfloat16)
    packed, escapes, exp_base = encode_nf12(weight, exp_base=54)
    assert escapes.shape[0] > 0
    prefix = "model.layers.0.self_attn.q_proj"
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            f"{prefix}.nf12_packed": packed,
            f"{prefix}.nf12_escapes": escapes,
            f"{prefix}.nf12_exp_base": torch.tensor([exp_base], dtype=torch.uint8),
        },
        checkpoint,
    )
    # The selected checkpoint, rather than every safetensors file in the
    # directory, determines the buffer shape.
    save_file({f"{prefix}.nf12_escapes": escapes[:1]}, tmp_path / "stale.safetensors")

    model = _Model(tmp_path)
    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[checkpoint],
        device_map={"": "cpu"},
        backend=BACKEND.NF12_AD,
        dtype=torch.bfloat16,
    )
    module = model.model.layers[0].self_attn.q_proj
    assert context.quant_linear is type(module)
    assert module.nf12_escapes.is_meta
    for name, value in load_file(checkpoint).items():
        set_module_tensor_to_device(model, name, "cpu", value=value)

    assert module.nf12_escapes.shape == escapes.shape
    assert torch.equal(module.dequantize_weight().view(torch.int16), weight.view(torch.int16))


@pytest.mark.parametrize(
    ("quantizers", "expected"),
    [
        (["gptqmodel:7.8.0"], True),
        (["gptq-model:7.8.0"], True),
        # Optimum records gptqmodel too, because it delegates the math to GPTQModel.
        (["optimum:2.3.0", "gptqmodel:7.8.0"], False),
        (["optimum:2.3.0"], False),
    ],
)
def test_native_provenance_ignores_optimum_owned_checkpoints(tmp_path, quantizers, expected):
    from gptqmodel.utils.hf_load import is_native_gptqmodel_checkpoint  # noqa: PLC0415

    payload = {"bits": 4, "group_size": 128, "format": "gptq", "meta": {"quantizer": quantizers}}
    (tmp_path / "quantize_config.json").write_text(json.dumps(payload), encoding="utf-8")
    model = _Model(tmp_path)

    assert is_native_gptqmodel_checkpoint(model, [tmp_path / "quantize_config.json"]) is expected


def test_native_provenance_declines_unresolvable_hub_repo():
    from gptqmodel.utils.hf_load import is_native_gptqmodel_checkpoint  # noqa: PLC0415

    model = torch.nn.Module()
    model.config = SimpleNamespace(_name_or_path="ModelCloud/Does-Not-Exist")

    assert is_native_gptqmodel_checkpoint(model, None) is False


def test_hf_load_bridge_leaves_legacy_optimum_checkpoint_to_existing_path(tmp_path):
    _write_checkpoint(tmp_path, native=False)
    model = _Model(tmp_path)

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    assert context is None
    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)


def test_hf_load_bridge_fails_when_checkpoint_module_is_missing_after_conversion(tmp_path):
    _write_checkpoint(tmp_path, missing_module=True)

    with pytest.raises(ValueError, match="missing_proj"):
        hf_gptqmodel_prepare_model_for_load(
            _Model(tmp_path),
            checkpoint_files=[tmp_path / "model.safetensors"],
            device_map={"": "cpu"},
            backend=BACKEND.GPTQ_TORCH,
            dtype=torch.float16,
        )


def test_hf_load_bridge_defuses_laguna_experts_on_meta(tmp_path):
    config = LagunaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["full_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse"],
        num_attention_heads_per_layer=[2, 2],
    )
    config._name_or_path = str(tmp_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    quantize_config = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "format": "gptq",
        "quant_method": "gptq",
        "meta": {"quantizer": ["gptqmodel:test"]},
    }
    (tmp_path / "quantize_config.json").write_text(json.dumps(quantize_config), encoding="utf-8")
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": torch.zeros((8, 64), dtype=torch.int32),
            "model.layers.1.mlp.experts.0.gate_proj.qweight": torch.zeros((8, 32), dtype=torch.int32),
        },
        tmp_path / "model.safetensors",
    )

    context = hf_gptqmodel_prepare_model_for_load(
        model,
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
    )

    assert isinstance(model.model.layers[0].self_attn.q_proj, TorchLinear)
    assert isinstance(model.model.layers[0].self_attn.g_proj, torch.nn.Linear)
    assert model.model.layers[0].self_attn.g_proj.out_features == 2
    assert isinstance(model.model.layers[1].mlp.experts[0].gate_proj, TorchLinear)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert context.quant_linear is TorchLinear


def test_hf_load_post_init_restores_gptqmodel_config_fields(monkeypatch):
    # Local import keeps the module object needed by `monkeypatch.setattr`, which
    # matches how other tests here patch internals; importing at module scope
    # would bind the symbol instead of the module.
    import gptqmodel.utils.model as model_utils  # noqa: PLC0415

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        dynamic={r"+:^model\.layers\.0\.self_attn\.q_proj$": {"group_size": 32}},
        lm_head=True,
    )
    context = HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=TorchLinear)
    model = torch.nn.Module()
    model.config = SimpleNamespace(quantization_config=GPTQConfig(bits=4))

    monkeypatch.setattr(
        model_utils, "hf_convert_gptq_v1_to_v2_format", lambda model, **kwargs: (model, False)
    )
    monkeypatch.setattr(model_utils, "hf_gptqmodel_post_init", lambda model, **kwargs: model)

    assert hf_gptqmodel_post_init_for_load(model, context) is model
    assert model.config.quantization_config.dynamic == qcfg.dynamic
    assert model.config.quantization_config.lm_head is True
    assert model.config.quantization_config.method == qcfg.method
    assert model.config.quantization_config.pack_dtype == "int32"


class _RotationModel(torch.nn.Module):
    """One-layer shell with the ``mlp.down_proj`` the rotation path rewrites."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", _name_or_path=str(checkpoint_path))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_RotationLayer()])


class _RotationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False, device="meta")
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(128, 32, bias=False, device="meta")


def _write_rotation_checkpoint(checkpoint_path, rotation):
    config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "format": "gptq",
        "quant_method": "gptq",
        "rotation": rotation,
        "meta": {"quantizer": ["gptqmodel:7.8.0"]},
    }
    (checkpoint_path / "quantize_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": torch.zeros((8, 32), dtype=torch.int32),
            "model.layers.0.mlp.down_proj.qweight": torch.zeros((32, 32), dtype=torch.int32),
        },
        checkpoint_path / "model.safetensors",
    )


def test_hf_load_bridge_auto_backend_picks_rotation_kernel(tmp_path):
    _write_rotation_checkpoint(tmp_path, "hadamard")

    context = hf_gptqmodel_prepare_model_for_load(
        _RotationModel(tmp_path),
        checkpoint_files=[tmp_path / "model.safetensors"],
        device_map={"": "cpu"},
        backend=BACKEND.AUTO,
        dtype=torch.float16,
    )

    # AUTO must resolve to a rotation-aware kernel instead of raising or
    # defaulting to a backend that cannot apply the online transform.
    assert context.quant_linear is TorchLinear


def test_hf_load_bridge_rejects_rotation_on_incompatible_backend(tmp_path):
    _write_rotation_checkpoint(tmp_path, "hadamard")

    with pytest.raises(NotImplementedError, match="rotation"):
        hf_gptqmodel_prepare_model_for_load(
            _RotationModel(tmp_path),
            checkpoint_files=[tmp_path / "model.safetensors"],
            device_map={"": "cpu"},
            backend=BACKEND.MARLIN,
            dtype=torch.float16,
        )


def test_hf_load_post_init_restores_online_hadamard_state(monkeypatch):
    """A rotated checkpoint must leave the online transform armed on ``mlp.down_proj``.

    Metadata alone is not enough: without ``online_full_had`` the rotated weights
    consume untransformed activations and the forward result is silently wrong.
    This drives the real post-init hook, so dropping the rotation setup fails here.
    """

    qcfg = QuantizeConfig(bits=4, group_size=128, rotation="hadamard")
    context = HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=TorchLinear)

    model = _RotationModel("unused")
    model.model.layers[0].mlp.down_proj = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=32,
        pack_dtype=torch.int32,
        register_buffers=True,
        bias=False,
    )
    model.model.layers[0].self_attn.q_proj = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        pack_dtype=torch.int32,
        register_buffers=True,
        bias=False,
    )

    # Stub only the parts that need real checkpoint tensors; keep the rotation
    # setup that this test is about. Both helpers are imported inside the hook
    # from `gptqmodel.utils.model`, so patch them at their source.
    import gptqmodel.utils.model as model_utils

    monkeypatch.setattr(
        model_utils, "hf_convert_gptq_v1_to_v2_format", lambda model, **kwargs: (model, False)
    )
    monkeypatch.setattr(model_utils, "gptqmodel_post_init", lambda model, **kwargs: model)

    assert model.model.layers[0].mlp.down_proj.online_full_had is not True

    returned = hf_gptqmodel_post_init_for_load(model, context)

    assert returned is model
    assert model.model.layers[0].mlp.down_proj.online_full_had is True
    assert getattr(model.model.layers[0].mlp.down_proj, "K", None) is not None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("w4afp8", BACKEND.GPTQ_W4AFP8),
        ("w4a_nvfp4", BACKEND.GPTQ_W4A_NVFP4),
    ],
)
def test_hf_load_bridge_auto_backend_applies_saved_activation_policy(mode, expected):
    """A saved W4A policy must pin its kernel even when the caller asked for AUTO.

    W4AFP8/NVFP4 kernels have no AUTO-selection priority, so without this step an
    AUTO load installs ordinary GPTQ kernels and post-init then fails because the
    required W4A projections are absent.
    """

    from gptqmodel.utils.hf_load import _resolve_backend_for_checkpoint

    qcfg = QuantizeConfig(bits=4, group_size=128, activation={"mode": mode})

    assert _resolve_backend_for_checkpoint(qcfg, BACKEND.AUTO) is expected
    assert _resolve_backend_for_checkpoint(qcfg, expected) is expected


def test_hf_load_bridge_still_rejects_incompatible_activation_backend():
    from gptqmodel.utils.hf_load import _resolve_backend_for_checkpoint

    qcfg = QuantizeConfig(bits=4, group_size=128, activation={"mode": "w4afp8"})

    with pytest.raises(ValueError, match="requires backend"):
        _resolve_backend_for_checkpoint(qcfg, BACKEND.GPTQ_TRITON)


def test_hf_load_post_init_opts_paged_attention_out_of_cuda_graphs(monkeypatch):
    """Paged Flash Attention is not CUDA-graph capture safe.

    The normal loader disables graph capture for paged models so
    `generate_batch` / `init_continuous_batching` work without a caller-side
    monkey patch. The bridge must do the same after loading.
    """

    from transformers import ContinuousBatchingConfig, PretrainedConfig

    qcfg = QuantizeConfig(bits=4, group_size=128)
    context = HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=TorchLinear)

    model = torch.nn.Module()
    model.config = PretrainedConfig()
    model.config._attn_implementation = "flash_attention_2|paged"
    model.model = torch.nn.Module()
    model.quantize_config = qcfg

    import gptqmodel.utils.model as model_utils

    monkeypatch.setattr(
        model_utils, "hf_convert_gptq_v1_to_v2_format", lambda model, **kwargs: (model, False)
    )
    monkeypatch.setattr(model_utils, "gptqmodel_post_init", lambda model, **kwargs: model)

    hf_gptqmodel_post_init_for_load(model, context)

    cb_config = model.generation_config.continuous_batching_config
    assert isinstance(cb_config, ContinuousBatchingConfig)
    assert cb_config.use_cuda_graph == (False, False)


def test_hf_load_post_init_leaves_non_paged_attention_untouched(monkeypatch):
    """A non-paged model must not gain a continuous-batching config."""

    from transformers import PretrainedConfig

    qcfg = QuantizeConfig(bits=4, group_size=128)
    context = HFGPTQModelLoadContext(quantize_config=qcfg, quant_linear=TorchLinear)

    model = torch.nn.Module()
    model.config = PretrainedConfig()
    model.config._attn_implementation = "sdpa"
    model.quantize_config = qcfg

    import gptqmodel.utils.model as model_utils

    monkeypatch.setattr(
        model_utils, "hf_convert_gptq_v1_to_v2_format", lambda model, **kwargs: (model, False)
    )
    monkeypatch.setattr(model_utils, "gptqmodel_post_init", lambda model, **kwargs: model)

    hf_gptqmodel_post_init_for_load(model, context)

    assert getattr(model, "generation_config", None) is None
