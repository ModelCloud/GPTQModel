# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import PreTrainedTokenizerFast

import gptqmodel.models.base as base_module
import gptqmodel.models.loader as loader_module
import gptqmodel.quantization.dtype as dtype_module
import gptqmodel.quantization.nf12 as nf12_quantization
from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization import AutoModuleDecoderConfig, QuantizeConfig
from gptqmodel.quantization.dtype import (
    dequantize_f4_e2m1,
    dequantize_fp8,
    device_supports_native_fp8,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils import auto_dtype_cast
from gptqmodel.utils.device_telemetry import (
    clear_device_telemetry_records,
    device_telemetry_scope,
    get_device_telemetry_records,
)
from gptqmodel.utils.structure import LazyTurtle


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except Exception:
    nvfp4_quantize = None


class _LinearWrapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)


def _write_index(path: Path, shard_name: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard_name)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def _write_minimal_tokenizer(path: Path, vocab_size: int) -> None:
    backend = Tokenizer(
        WordLevel({f"w{i}": i for i in range(vocab_size)}, unk_token="w0")
    )
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="w0",
        pad_token="w0",
        bos_token="w1",
        eos_token="w2",
    ).save_pretrained(str(path))


def _build_qwen3_fp8_source(path: Path) -> None:
    """Save the Qwen/Qwen3-0.6B-FP8 checkpoint layout for two layers.

    Key names, dtypes and projection shapes follow the official hub
    ``config.json`` and safetensors header. Only the layer count and the
    vocabulary are truncated so the fixture stays small; values are synthetic.
    """

    vocab = 256
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": vocab,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    _write_minimal_tokenizer(path, vocab)

    hidden, intermediate, heads, kv_heads, head_dim = 1024, 3072, 16, 8, 128
    generator = torch.Generator().manual_seed(6139)

    def fp8_weight(shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.02).to(
            torch.float8_e4m3fn
        )

    def bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.02

    state = {
        "model.embed_tokens.weight": bf16((vocab, hidden)),
        "lm_head.weight": bf16((vocab, hidden)),
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        state[prefix + "post_attention_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        state[prefix + "self_attn.q_norm.weight"] = torch.ones(
            head_dim, dtype=torch.bfloat16
        )
        state[prefix + "self_attn.k_norm.weight"] = torch.ones(
            head_dim, dtype=torch.bfloat16
        )
        projections = (
            ("q_proj", heads * head_dim, hidden),
            ("k_proj", kv_heads * head_dim, hidden),
            ("v_proj", kv_heads * head_dim, hidden),
            ("o_proj", hidden, heads * head_dim),
        )
        for name, out_features, in_features in projections:
            state[prefix + f"self_attn.{name}.weight"] = fp8_weight(
                (out_features, in_features)
            )
            state[prefix + f"self_attn.{name}.weight_scale_inv"] = torch.ones(
                (out_features // 128, in_features // 128), dtype=torch.bfloat16
            )
        mlp = (
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        )
        for name, out_features, in_features in mlp:
            state[prefix + f"mlp.{name}.weight"] = fp8_weight(
                (out_features, in_features)
            )
            state[prefix + f"mlp.{name}.weight_scale_inv"] = torch.ones(
                (out_features // 128, in_features // 128), dtype=torch.bfloat16
            )

    save_file(state, str(path / "model.safetensors"))
    _write_index(path, "model.safetensors", list(state))


def _build_gpt_oss_mxfp4_source(path: Path) -> None:
    """Save the openai/gpt-oss-20b MXFP4 checkpoint layout for two layers.

    Key names, dtypes and projection shapes follow the official hub
    ``config.json`` and safetensors header. Only the layer count, expert count
    and vocabulary are truncated; values are synthetic.
    """

    vocab, experts = 256, 4
    config = {
        "architectures": ["GptOssForCausalLM"],
        "attention_bias": True,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "experts_per_token": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 2880,
        "initial_context_length": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 2880,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 131072,
        "model_type": "gpt_oss",
        "num_attention_heads": 64,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "num_local_experts": experts,
        "output_router_logits": False,
        "pad_token_id": 0,
        "quantization_config": {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
            "quant_method": "mxfp4",
        },
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
        },
        "rope_theta": 150000,
        "router_aux_loss_coef": 0.9,
        "sliding_window": 128,
        "swiglu_limit": 7.0,
        "tie_word_embeddings": False,
        "transformers_version": "4.55.0.dev0",
        "use_cache": True,
        "vocab_size": vocab,
    }
    (path / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    _write_minimal_tokenizer(path, vocab)

    hidden, intermediate, heads, kv_heads, head_dim = 2880, 2880, 64, 8, 64
    generator = torch.Generator().manual_seed(6139)

    def bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.02

    def blocks(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.full(shape, 0x21, dtype=torch.uint8)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        # E8M0 exponent 128 is the value 2.0 for every block.
        return torch.full(shape, 128, dtype=torch.uint8)

    state = {
        "model.embed_tokens.weight": bf16((vocab, hidden)),
        "lm_head.weight": bf16((vocab, hidden)),
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        state[prefix + "post_attention_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        state[prefix + "self_attn.sinks"] = bf16((heads,))
        attention = (
            ("q_proj", heads * head_dim, hidden),
            ("k_proj", kv_heads * head_dim, hidden),
            ("v_proj", kv_heads * head_dim, hidden),
            ("o_proj", hidden, heads * head_dim),
        )
        for name, out_features, in_features in attention:
            state[prefix + f"self_attn.{name}.weight"] = bf16(
                (out_features, in_features)
            )
            state[prefix + f"self_attn.{name}.bias"] = bf16((out_features,))
        state[prefix + "mlp.router.weight"] = bf16((experts, hidden))
        state[prefix + "mlp.router.bias"] = bf16((experts,))
        gate_up = 2 * intermediate
        state[prefix + "mlp.experts.gate_up_proj_blocks"] = blocks(
            (experts, gate_up, hidden // 32, 16)
        )
        state[prefix + "mlp.experts.gate_up_proj_scales"] = scales(
            (experts, gate_up, hidden // 32)
        )
        state[prefix + "mlp.experts.gate_up_proj_bias"] = bf16((experts, gate_up))
        state[prefix + "mlp.experts.down_proj_blocks"] = blocks(
            (experts, hidden, intermediate // 32, 16)
        )
        state[prefix + "mlp.experts.down_proj_scales"] = scales(
            (experts, hidden, intermediate // 32)
        )
        state[prefix + "mlp.experts.down_proj_bias"] = bf16((experts, hidden))

    save_file(state, str(path / "model.safetensors"))
    _write_index(path, "model.safetensors", list(state))


def test_native_floatx_source_configuration_injects_decoder_and_preserves_lazy_weights(monkeypatch):
    source_config = SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        }
    )
    qcfg = QuantizeConfig(bits=4, offload_to_disk=False)

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: True)

    source_format = loader_module.configure_native_floatx_source_quantization(
        source_config,
        qcfg,
        device=torch.device("cuda:0"),
    )

    assert source_format == "nvfp4"
    assert qcfg.offload_to_disk is True
    assert qcfg._native_floatx_forward_plan["source_format"] == "nvfp4"
    assert qcfg._native_floatx_forward_plan["bf16_validated"] is True
    decoder = next(item for item in qcfg.preprocessors if isinstance(item, AutoModuleDecoderConfig))
    assert decoder.target_dtype is torch.bfloat16


def test_native_floatx_source_configuration_fails_before_model_load_without_bf16(monkeypatch):
    source_config = SimpleNamespace(quantization_config={"quant_algo": "FP8"})
    qcfg = QuantizeConfig(bits=4, offload_to_disk=False)

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: False)

    with pytest.raises(EnvironmentError, match="BF16 linear support"):
        loader_module.configure_native_floatx_source_quantization(
            source_config,
            qcfg,
            device=torch.device("cuda:0"),
        )

    assert qcfg.offload_to_disk is False
    assert not any(isinstance(item, AutoModuleDecoderConfig) for item in qcfg.preprocessors)


def test_native_floatx_source_configuration_preserves_fp32_hessian(monkeypatch):
    source_config = SimpleNamespace(quantization_config={"quant_algo": "FP8"})
    qcfg = QuantizeConfig(bits=4)
    qcfg.hessian.staging_dtype = torch.bfloat16

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: True)

    loader_module.configure_native_floatx_source_quantization(
        source_config,
        qcfg,
        device=torch.device("cuda:0"),
    )

    assert qcfg.hessian.staging_dtype is torch.float32
    assert qcfg._native_floatx_forward_plan["hessian_accumulation_dtype"] == "float32"


def test_native_floatx_source_format_reads_modelopt_sidecar(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8"}}),
        encoding="utf-8",
    )

    assert loader_module.native_floatx_source_format(
        SimpleNamespace(quantization_config=None),
        model_local_path=str(tmp_path),
    ) == "fp8"


def test_native_floatx_source_format_reads_hf_fp8_quant_method():
    """Qwen's native FP8 export uses quant_method rather than quant_algo."""
    assert loader_module.native_floatx_source_format(
        SimpleNamespace(quantization_config={"quant_method": "fp8"}),
    ) == "fp8"


@pytest.mark.parametrize("label", ["mxfp8", "mxfp4", "nvfp8", "mixed-mxfp4-mxfp8"])
def test_native_floatx_source_format_distinguishes_scale_layouts(label):
    expected = "mixed-mx" if label.startswith("mixed") else label
    assert loader_module.native_floatx_source_format(
        SimpleNamespace(quantization_config={"format": label}),
    ) == expected


def test_native_floatx_source_format_reads_quark_global_and_layer_specs():
    config = SimpleNamespace(quantization_config={
        "quant_method": "quark_online",
        "global_quant_config": {"weight": {
            "dtype": "fp4", "group_size": 32, "scale_format": "e8m0",
        }},
        "layer_quant_config": {"*experts*": {"weight": {
            "dtype": "fp8_e5m2", "group_size": 32, "scale_format": "e8m0",
        }}},
    })
    assert loader_module.native_floatx_source_format(config) == "mixed-mx"


@pytest.mark.parametrize("label", ["mxfp8", "mxfp4", "nvfp8"])
def test_mx_and_nvfp8_sources_use_dense_decoder_until_kernel_is_validated(label, monkeypatch):
    qcfg = QuantizeConfig(bits=4, offload_to_disk=False)
    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: True)
    source_format = loader_module.configure_native_floatx_source_quantization(
        SimpleNamespace(quantization_config={"format": label}), qcfg, device="cuda:0",
    )
    assert source_format == label
    assert qcfg._native_floatx_forward_plan["native_validated"] is False
    assert qcfg._native_floatx_forward_plan["mode"] == "decode"


@pytest.mark.parametrize("label,weight,scale,expected", [
    (
        "mxfp8",
        torch.ones((1, 32), dtype=torch.float32).to(torch.float8_e5m2),
        torch.tensor([[128]], dtype=torch.uint8),
        torch.full((1, 32), 2.0, dtype=torch.bfloat16),
    ),
    (
        "mxfp4",
        torch.full((1, 16), 0x21, dtype=torch.int8),
        torch.tensor([[128]], dtype=torch.uint8),
        torch.tensor([[1.0, 2.0] * 16], dtype=torch.bfloat16),
    ),
])
def test_module_decoder_reconstructs_mx_weight(label, weight, scale, expected):
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = _LinearWrapper(32, 1)
    harness.model.config = SimpleNamespace(quantization_config={"format": label})
    decoded = harness._build_decoder_quant_source_module(
        harness.model.linear,
        checkpoint_tensors={"weight": weight, "weight_scale": scale},
        target_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(decoded.weight, expected)


def test_module_decoder_reconstructs_quark_mxfp4_sibling_scale():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = _LinearWrapper(32, 1)
    harness.model.config = SimpleNamespace(quantization_config={
        "quant_method": "quark_online",
        "global_quant_config": {"weight": {
            "dtype": "fp4", "group_size": 32, "scale_format": "e8m0",
        }},
    })
    decoded = harness._build_decoder_quant_source_module(
        harness.model.linear,
        checkpoint_tensors={
            "weight": torch.full((1, 16), 0x21, dtype=torch.int8),
            "scale": torch.tensor([[128]], dtype=torch.uint8),
        },
        target_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(decoded.weight, torch.tensor([[1.0, 2.0] * 16], dtype=torch.bfloat16))


@pytest.mark.parametrize("weight_format", ["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("secondary_layout", ["global", "padded"])
def test_module_decoder_clips_padded_mx_scales_before_expansion(weight_format, secondary_layout):
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = _LinearWrapper(64, 2)
    harness.model.config = SimpleNamespace(quantization_config={
        "format": "mxfp4" if weight_format == "mxfp4" else "mxfp8",
        "weight_block_size": [1, 32],
    })
    if weight_format == "mxfp4":
        # Independent E2M1 oracle: each row exercises every positive and negative code.
        values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
        packed_bytes = [code | ((15 - code) << 4) for code in range(16)] * 2
        weight = torch.tensor(packed_bytes, dtype=torch.uint8).repeat(2, 1)
        unscaled = torch.tensor(
            [item for code in range(16) for item in (values[code], values[15 - code])] * 2,
            dtype=torch.float32,
        ).repeat(2, 1)
    else:
        fp8_dtype = torch.float8_e4m3fn if weight_format.endswith("e4m3") else torch.float8_e5m2
        unscaled = torch.tensor([-1.5, 0.5, 2.0, 4.0] * 16, dtype=torch.float32).repeat(2, 1)
        weight = unscaled.to(fp8_dtype)
    # The last two columns and rows are padding. The 255 byte must never reach the decoder.
    scale = torch.tensor([
        [127, 128, 255, 255],
        [129, 126, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
    ], dtype=torch.uint8)
    global_scale = torch.tensor(0.3, dtype=torch.float32)
    if secondary_layout == "padded":
        scale_2 = torch.full((4, 4), 99.0)
        scale_2[:2, :2] = global_scale
    else:
        scale_2 = global_scale
    oracle_scale = torch.tensor([[1.0, 2.0], [4.0, 0.5]], dtype=torch.float64).repeat_interleave(32, dim=1)
    expected = (unscaled.to(torch.float64) * oracle_scale * global_scale.to(torch.float64)).to(torch.bfloat16)

    decoded = harness._build_decoder_quant_source_module(
        nn.Linear(64, 2, bias=False),
        checkpoint_tensors={"weight": weight, "weight_scale": scale, "weight_scale_2": scale_2},
        target_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(decoded.weight, expected, rtol=0, atol=0)
    if torch.cuda.is_available():
        inputs = torch.arange(64, device="cuda:0", dtype=torch.bfloat16).reshape(1, 64) / 64
        actual = decoded.to("cuda:0")(inputs)
        oracle = torch.nn.functional.linear(inputs, expected.to("cuda:0"))
        torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


def test_auto_module_decoder_config_exposes_validated_policies():
    config = AutoModuleDecoderConfig(
        passthrough_forward_policy="NATIVE",
        passthrough_save_policy="decode",
    )

    assert config.passthrough_forward_policy == "native"
    assert config.to_dict()["passthrough_save_policy"] == "decode"

    with pytest.raises(ValueError, match="passthrough_forward_policy"):
        AutoModuleDecoderConfig(passthrough_forward_policy="invalid")


def test_fp4_decoder_has_torch_only_fallback(monkeypatch):
    monkeypatch.setattr(dtype_module, "unpack_uint4", None)
    monkeypatch.setattr(dtype_module, "f4_unpacked_to_f32", None)
    packed = torch.tensor([[0x10, 0x98]], dtype=torch.uint8)

    decoded = dequantize_f4_e2m1(packed, axis=None, target_dtype=torch.bfloat16)

    expected = torch.tensor([[0.0, 0.5, 0.0, -0.5]], dtype=torch.bfloat16)
    torch.testing.assert_close(decoded, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_defers_fp8_quant_source_until_quantization(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale_inv = torch.linspace(2.0, 3.75, steps=source_model.linear.out_features, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
        "quant_source_decode_policy": "defer_until_quantization",
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    clear_device_telemetry_records()
    with device_telemetry_scope(True):
        prepared = base_module.BaseQModel.shell_module_materialize(
            harness,
            target_submodule=shell_model.linear,
            device=torch.device("cpu"),
            role="forward",
            named_module=named,
        )

    assert isinstance(prepared, TorchFP8Linear)
    assert isinstance(shell_model.linear, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state
    expected = dequantize_fp8(
        weight_fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    with device_telemetry_scope(True):
        quant_source = base_module.BaseQModel.shell_module_materialize(
            harness,
            target_submodule=prepared,
            device=torch.device("cpu"),
            role="quant_source",
            named_module=named,
        )
    assert isinstance(quant_source, nn.Linear)
    assert quant_source.weight.device.type == "cpu"
    assert quant_source.weight.dtype == torch.bfloat16
    torch.testing.assert_close(quant_source.weight, expected)
    # The native forward wrapper and the dense quant source decode the same
    # checkpoint tensor, so they must not disagree about the scale semantics.
    forward_weight = prepared.dequantize_weight(device="cpu", dtype=torch.float32).T
    torch.testing.assert_close(forward_weight, expected.float(), rtol=1e-2, atol=1e-8)
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "native"
    assert harness.auto_module_decoder_events[0]["deferred_quant_source_decode"] is True
    assert {record["event"] for record in get_device_telemetry_records()} == {
        "floatx_forward_plan",
        "floatx_native_forward_used",
        "floatx_deferred_quant_source_decode",
    }


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
@pytest.mark.parametrize(
    ("semantics", "scale_inv_value"),
    [("multiplier", 0.01), ("inverse", 100.0)],
    ids=["deepseek_multiplier", "native_inverse"],
)
def test_fp8_forward_and_quant_source_share_the_declared_scale_semantics(
    tmp_path, semantics, scale_inv_value
):
    """Both FP8 decode paths must apply one resolved checkpoint convention.

    HF/DeepSeek checkpoints multiply ``weight_scale_inv``; a native GPT-QModel
    inverse scale divides. Reading the two paths independently made them differ
    by 10^4 for the same tensor.
    """

    model_dir = tmp_path / f"fp8_{semantics}"
    model_dir.mkdir()

    torch.manual_seed(7)
    weight_fp8 = (torch.randn(8, 16) * 0.02).to(torch.float8_e4m3fn)
    scale_inv = torch.full((8,), scale_inv_value, dtype=torch.float32)
    shard_name = "model.safetensors"
    save_file(
        {"linear.weight": weight_fp8, "linear.weight_scale_inv": scale_inv},
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(
            _experts_implementation=None,
            quantization_config={
                "quant_method": "fp8",
                "format": "float8_e4m3fn",
                "weight_scale_semantics": semantics,
            },
        ),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    checkpoint_tensors = turtle.checkpoint_tensors_for_submodule(
        target_model=shell_model,
        target_submodule=shell_model.linear,
        recurse=False,
    )
    forward_module = base_module.BaseQModel._build_fp8_forward_module(
        harness,
        target_submodule=shell_model.linear,
        checkpoint_tensors=checkpoint_tensors,
        device=torch.device("cpu"),
        target_dtype=torch.bfloat16,
    )
    quant_source = base_module.BaseQModel._build_decoder_quant_source_module(
        harness,
        shell_model.linear,
        checkpoint_tensors=checkpoint_tensors,
        target_dtype=torch.bfloat16,
        module_name="linear",
    )

    assert isinstance(forward_module, TorchFP8Linear)
    assert forward_module.weight_scale_semantics == semantics
    # Independent FP64 oracle: no call into either decode path.
    source = weight_fp8.double()
    if semantics == "multiplier":
        expected = source * scale_inv.double()[:, None]
    else:
        expected = source / scale_inv.double()[:, None]

    forward_weight = forward_module.dequantize_weight(device="cpu", dtype=torch.float32).T
    torch.testing.assert_close(forward_weight.double(), expected, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        quant_source.weight.double(),
        expected,
        rtol=1e-2,
        atol=0,
    )
    torch.testing.assert_close(
        forward_weight.double(),
        quant_source.weight.double(),
        rtol=1e-2,
        atol=0,
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_decodes_when_passthrough_forward_policy_is_decode(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source_decode"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale_inv = torch.linspace(2.0, 3.75, steps=source_model.linear.out_features, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
        "passthrough_forward_policy": "decode",
        "passthrough_save_policy": "decode",
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    expected = dequantize_fp8(
        weight_fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, nn.Linear)
    assert not isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "decode"
    torch.testing.assert_close(prepared.weight, expected)
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "decode"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_quant_source_swaps_back_to_dense_module(tmp_path, monkeypatch):
    source_model = _LinearWrapper(8, 4).eval()
    model_dir = tmp_path / "fp8_source"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu(),
            "linear.weight_scale_inv": torch.ones(4, dtype=torch.float32),
            "linear.bias": source_model.linear.bias.detach().cpu(),
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(8, 4).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    forward_module = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )
    named.module = forward_module

    quant_source = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=forward_module,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )

    assert isinstance(quant_source, nn.Linear)
    assert isinstance(shell_model.linear, nn.Linear)
    torch.testing.assert_close(
        quant_source.weight,
        named.state["quant_source_module"].weight,
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_builds_fp8_wrapper_from_weight_scale_metadata(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source_scale"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale = torch.full((), 0.5, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale": scale,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    assert isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state
    torch.testing.assert_close(prepared.weight_scale_inv, torch.reciprocal(scale).to(torch.float32))
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "native"


@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_fp4_forward_module_keeps_stored_bias_width():
    """The FP4 wrapper must receive the checkpoint bias at its stored width.

    ``TorchFP4Linear`` selects its wide epilogue from the stored bias dtype, so
    narrowing the bias to the activation dtype while the wrapper is built turns
    an FP32 ``-65536`` into FP16 ``-inf`` before the epilogue can apply it.
    """

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(config=SimpleNamespace(quantization_config=None))

    target = nn.Linear(32, 8, bias=True)
    packed = torch.zeros((8, 16), dtype=torch.uint8)
    scale = torch.ones((8, 2), dtype=torch.float32)
    bias = torch.full((8,), -65536.0, dtype=torch.float32)

    prepared = base_module.BaseQModel._build_fp4_forward_module(
        harness,
        target_submodule=target,
        checkpoint_tensors={"weight": packed, "weight_scale": scale, "bias": bias},
        device=torch.device("cpu"),
        target_dtype=torch.float16,
    )

    assert isinstance(prepared, TorchFP4Linear)
    assert prepared.bias.dtype is torch.float32
    assert torch.isfinite(prepared.bias).all()
    assert float(prepared.bias[0].item()) == -65536.0


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_shell_materialize_forward_decodes_fp4_source_to_dense_module(tmp_path):
    source_model = _LinearWrapper(16, 4).eval()
    model_dir = tmp_path / "fp4_source"
    model_dir.mkdir()

    scales, packed = nvfp4_quantize(source_model.linear.weight.detach().to(torch.float32), block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed_float4.cpu(),
            "linear.weight_scale": scales.cpu(),
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 4).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    expected = dequantize_f4_e2m1(
        packed_float4.cpu(),
        scale=scales.cpu(),
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, nn.Linear)
    assert not isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "decode"
    torch.testing.assert_close(prepared.weight, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(named.state["quant_source_module"].weight, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="E8M0 dtype not available")
def test_shell_materialize_forward_decodes_signed_packed_fp4_with_direct_scale(tmp_path, monkeypatch):
    model_dir = tmp_path / "deepseek_v4_style_fp4_source"
    model_dir.mkdir()

    codes = torch.tensor(list(range(16)) * 4, dtype=torch.uint8).view(2, 32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    scales = torch.tensor([[127], [128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed,
            "linear.scale": scales,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.scale"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = nn.Module()
    shell_model.config = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
        }
    )
    shell_model.linear = nn.Linear(32, 2, bias=False, device="meta")

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    # Exercise the TorchAO-free reference required by CPython 3.14t images.
    monkeypatch.setattr(dtype_module, "_load_floatx_cpu_ops", lambda: None)
    monkeypatch.setattr(dtype_module, "unpack_uint4", None)
    monkeypatch.setattr(dtype_module, "f4_unpacked_to_f32", None)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.bfloat16,
    )
    expected = codebook[codes.to(torch.long)]
    expected[1].mul_(2.0)

    assert isinstance(prepared, nn.Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "decode"
    torch.testing.assert_close(prepared.weight, expected)
    torch.testing.assert_close(named.state["quant_source_module"].weight, expected)


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_shell_materialize_forward_builds_fp4_wrapper_when_native_supported(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 4).eval()
    model_dir = tmp_path / "fp4_native_source"
    model_dir.mkdir()

    scales, packed = nvfp4_quantize(source_model.linear.weight.detach().to(torch.float32), block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed_float4.cpu(),
            "linear.weight_scale": scales.cpu(),
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(
            _experts_implementation=None,
            quantization_config={"format": "nvfp4"},
        ),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 4).eval()
    shell_model.config = SimpleNamespace(quantization_config={"format": "nvfp4"})
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_native_fp4", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    dequantize_f4_e2m1(
        packed_float4.cpu(),
        scale=scales.cpu(),
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, TorchFP4Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state


def test_configure_modelopt_runtime_rejects_modelopt_input_activation_quantization():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            },
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = None

    with pytest.raises(ValueError, match="activation quantized|activation scales"):
        base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_allows_native_source_requantization():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            }
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = None
    harness.quantize_config = SimpleNamespace(preprocessors=[AutoModuleDecoderConfig()])

    base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_rejects_checkpoint_activation_scale_metadata():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "modelopt"}),
        state_dict=lambda: {},
    )
    harness.turtle_model = base_module.LazyTurtle.__new__(base_module.LazyTurtle)
    harness.turtle_model._weight_map = {"model.layers.0.self_attn.q_proj.input_scale": "model.safetensors"}

    with pytest.raises(ValueError, match="activation quantized|activation scales"):
        base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_allows_weight_only_modelopt_metadata():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            },
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = base_module.LazyTurtle.__new__(base_module.LazyTurtle)
    harness.turtle_model._weight_map = {"model.layers.0.self_attn.q_proj.weight_scale": "model.safetensors"}

    base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_gptq_prefers_quant_source_module_when_present():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    task = GPTQ(named)

    assert task.module is quant_source


def test_module_looper_materializes_deferred_native_source_before_gptq():
    """The native Hessian wrapper must never become the GPTQ source module."""

    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state.update(
        {
            "auto_module_decoder": {"code": "auto_module_decoder"},
            "auto_module_decoder_forward_mode": "native",
        }
    )

    calls = []

    class _Model:
        def shell_module_materialize(self, **kwargs):
            calls.append(kwargs["role"])
            kwargs["named_module"].state["quant_source_module"] = quant_source
            return quant_source

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = _Model()
    looper._assign_quant_device_for_module = lambda named_module, fallback_device: fallback_device

    ModuleLooper._prepare_named_module_for_quantization(
        looper,
        SimpleNamespace(tasks={}),
        named,
        torch.device("cpu"),
    )

    assert calls == ["quant_source"]
    assert named.module is quant_source


def test_awq_resolve_quant_source_module_prefers_dense_source():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    resolved = AWQProcessor.resolve_quant_source_module(named)

    assert resolved is quant_source


@pytest.mark.parametrize(
    ("forward_required", "expect_packed"),
    [(False, False), (True, True)],
    ids=["planning_processor", "quantizing_processor"],
)
def test_auto_nf12_forward_pack_respects_processor_forward_policy(
    monkeypatch, forward_required, expect_packed
):
    """A no-forward planning processor must not hand NF12 wrappers to the quantizer."""

    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(
        nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None)
    )

    linear = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) * 0.02)
    named = NamedModule(linear, name="proj", full_name="model.proj", layer_index=0)

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = nn.Module()
    harness.model.add_module("model", nn.Module())
    harness.model.model.add_module("proj", linear)
    harness.quantize_config = QuantizeConfig(
        bits=4, group_size=32, offload_to_disk=False
    )
    harness.lm_head = None
    harness.auto_dtype_cast_events = []

    class _Processor:
        forward = SimpleNamespace(
            required=forward_required, allow_auto_dtype_cast=True
        )
        tasks = {}

        def preprocess(self, module):
            module.state["planned"] = True

        def is_skipped(self, module):
            return False

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = harness
    looper.input_embeddings_name = None
    looper.output_embeddings_name = None
    looper.layer_module_coverage = SimpleNamespace(
        record_hit=lambda *_args: None,
        record_miss=lambda *_args: None,
        record_undeclared=lambda *_args: None,
    )

    subset = ModuleLooper.create_named_modules(
        looper,
        module=harness.model,
        full={"proj": named},
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="model",
        names=["proj"],
        processor=_Processor(),
        fallback=False,
    )

    assert named.state["planned"] is True
    assert subset["proj"] is named
    packed = isinstance(named.module, auto_dtype_cast.AutoNF12HookedLinear)
    assert packed is expect_packed
    assert bool(harness.auto_dtype_cast_events) is expect_packed


def test_mxfp4_fused_expert_checkpoint_tensors_resolve_blocks_scales_and_bias(tmp_path):
    """Defused GPT-OSS MXFP4 experts read their fused blocks/scales/bias (issue #6139)."""

    experts, hidden, intermediate = 2, 64, 32
    gate_up_blocks = torch.full(
        (experts, 2 * intermediate, hidden // 32, 16), 0x21, dtype=torch.uint8
    )
    gate_up_scales = torch.full(
        (experts, 2 * intermediate, hidden // 32), 128, dtype=torch.uint8
    )
    gate_up_bias = torch.arange(
        experts * 2 * intermediate, dtype=torch.bfloat16
    ).reshape(experts, 2 * intermediate)
    down_blocks = torch.full(
        (experts, hidden, intermediate // 32, 16), 0x21, dtype=torch.uint8
    )
    down_scales = torch.full(
        (experts, hidden, intermediate // 32), 128, dtype=torch.uint8
    )
    down_bias = torch.arange(
        experts * hidden, dtype=torch.bfloat16
    ).reshape(experts, hidden)
    state = {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": gate_up_blocks,
        "model.layers.0.mlp.experts.gate_up_proj_scales": gate_up_scales,
        "model.layers.0.mlp.experts.gate_up_proj_bias": gate_up_bias,
        "model.layers.0.mlp.experts.down_proj_blocks": down_blocks,
        "model.layers.0.mlp.experts.down_proj_scales": down_scales,
        "model.layers.0.mlp.experts.down_proj_bias": down_bias,
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    _write_index(tmp_path, "model.safetensors", list(state))

    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    for index in range(experts):
        container = nn.Module()
        container.gate_proj = nn.Linear(hidden, intermediate, bias=True)
        container.up_proj = nn.Linear(hidden, intermediate, bias=True)
        container.down_proj = nn.Linear(intermediate, hidden, bias=True)
        layer.mlp.experts.add_module(str(index), container)
    model.model.layers = nn.ModuleList([layer])
    for name, parameter in list(model.named_parameters()):
        parent_path, leaf = name.rsplit(".", 1)
        setattr(
            model.get_submodule(parent_path),
            leaf,
            nn.Parameter(
                torch.empty_like(parameter, device="meta"), requires_grad=False
            ),
        )

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(tmp_path),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    expert = 1
    projections = (
        ("gate_proj", gate_up_blocks, gate_up_scales, gate_up_bias, 0),
        ("up_proj", gate_up_blocks, gate_up_scales, gate_up_bias, 1),
        ("down_proj", down_blocks, down_scales, down_bias, None),
    )
    resolved = {}
    for name, blocks, scales, bias, split in projections:
        module_path = f"model.layers.0.mlp.experts.{expert}.{name}"
        tensors = turtle.checkpoint_tensors_for_submodule(
            target_model=model,
            target_submodule=model.get_submodule(module_path),
            recurse=False,
            module_path=module_path,
        )
        expected_blocks, expected_scales, expected_bias = (
            blocks[expert],
            scales[expert],
            bias[expert],
        )
        if split is not None:
            expected_blocks = expected_blocks.chunk(2, dim=0)[split]
            expected_scales = expected_scales.chunk(2, dim=0)[split]
            expected_bias = expected_bias.chunk(2, dim=0)[split]
        assert torch.equal(
            tensors["weight"],
            expected_blocks.reshape(expected_blocks.shape[0], -1),
        )
        assert torch.equal(tensors["weight_scale"], expected_scales)
        assert torch.equal(tensors["bias"], expected_bias)
        resolved[name] = tensors

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.model.config = SimpleNamespace(
        quantization_config={"quant_method": "mxfp4"}
    )
    gate_path = f"model.layers.0.mlp.experts.{expert}.gate_proj"
    decoded = harness._build_decoder_quant_source_module(
        model.get_submodule(gate_path),
        checkpoint_tensors=resolved["gate_proj"],
        target_dtype=torch.bfloat16,
    )
    # Every packed byte 0x21 decodes to (0.5, 1.0) and E8M0 128 is 2.0.
    expected_dense = torch.tensor(
        [[1.0, 2.0] * (hidden // 2)] * intermediate, dtype=torch.bfloat16
    )
    torch.testing.assert_close(decoded.weight, expected_dense)
    torch.testing.assert_close(decoded.bias, gate_up_bias[expert, :intermediate])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layout", ["qwen3_fp8", "gpt_oss_mxfp4"])
def test_floatx_source_quantizes_with_default_auto_nf12(tmp_path, monkeypatch, layout):
    """AUTO_NF12 must not replace the dense linear a floatx quantizer consumes."""

    device = torch.device("cuda", torch.cuda.current_device())
    if not auto_dtype_cast.nf12_supported_device(device):
        pytest.skip("requires a device with AUTO_NF12 support")
    if layout == "qwen3_fp8" and device_supports_native_fp8(
        device, require_validation=True
    ):
        pytest.skip("native FP8 is validated; this fixture covers the decode path")

    model_dir = tmp_path / layout
    model_dir.mkdir()
    if layout == "qwen3_fp8":
        _build_qwen3_fp8_source(model_dir)
    else:
        _build_gpt_oss_mxfp4_source(model_dir)

    preprocessed = []
    original_preprocess = GPTQProcessor.preprocess

    def spy(self, module, *args, **kwargs):
        if module.full_name == "model.layers.0.self_attn.q_proj":
            preprocessed.append(type(module.module))
        return original_preprocess(self, module, *args, **kwargs)

    monkeypatch.setattr(GPTQProcessor, "preprocess", spy)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=32),
        device=str(device),
    )
    rows = [
        list(range(2 + index * 16, 18 + index * 16)) for index in range(4)
    ]
    model.quantize(
        CalibrationConfig(dataset=rows, batch_size=2), backend=BACKEND.TORCH
    )

    # The NF12 wrapper is a calibration stand-in: the GPTQ processor must
    # receive the dense linear source that it owns.
    assert preprocessed
    assert issubclass(preprocessed[0], nn.Linear)
    assert not any(
        issubclass(module_type, auto_dtype_cast.AutoNF12HookedLinear)
        for module_type in preprocessed
    )
    # NF12 is still used for the Hessian forward of the dense source.
    assert any(
        event["module"] == "model.layers.0.self_attn.q_proj"
        for event in model.auto_dtype_cast_events
    )

    logged = {str(row["module"]) for row in model.quant_log}
    assert "self_attn.q_proj" in logged
    if layout == "gpt_oss_mxfp4":
        # The fused MXFP4 experts must decode from _blocks/_scales, not stay
        # on meta, so they reach the quantizer like any dense projection.
        assert "mlp.experts.0.gate_proj" in logged
        assert "mlp.experts.0.down_proj" in logged
