# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU GPTQ acceptance using generated mixed-format MiMo checkpoints."""

import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

import gptqmodel.models.loader as loader_module
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models.definitions.mimo_v2 import _compatible_mask
from gptqmodel.quantization.config import (
    AutoModuleDecoderConfig,
    AWQConfig,
    ExpertsRoutingOverride,
    MoEConfig,
    QuantizeConfig,
)
from gptqmodel.quantization.gptq import GPTQ


@pytest.fixture
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make_source(
    path: Path, tp: int, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.nn.Module, dict]:
    path.mkdir()
    for file in (Path(__file__).parent / "fixtures/mimo_v2").glob("*.py"):
        shutil.copyfile(file, path / file.name)
    config = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        },
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2 * tp,
        "num_key_value_heads": tp,
        "head_dim": 192,
        "v_head_dim": 128,
        "swa_num_attention_heads": 2 * tp,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 192,
        "swa_v_head_dim": 128,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "layernorm_epsilon": 1e-6,
        "initializer_range": 0.02,
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "partial_rotary_factor": 0.334,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.334,
        },
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 1],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": None,
        "attention_projection_layout": "fused_qkv",
        "attention_value_scale": 0.707 if tp == 4 else 0.612,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "add_swa_attention_sink_bias": True,
        "sliding_window": 8,
        "sliding_window_size": 8,
        "use_cache": True,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "ignored_layers": [
                f"model.layers.{layer}.self_attn.o_proj" for layer in range(2)
            ],
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                **{f"t{i}": i for i in range(4, 128)},
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    ).save_pretrained(path)
    hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        hf_config,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).eval()
    modeling = __import__(type(model).__module__, fromlist=["create_causal_mask"])
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        setattr(modeling, name, _compatible_mask(getattr(modeling, name)))
    torch.manual_seed(123)
    weights, scales, dense = {}, {}, {}
    for name, parameter in model.state_dict().items():
        value = torch.randn(parameter.shape).mul_(0.02).bfloat16()
        if "norm.weight" in name:
            value.fill_(1)
        dense[name] = value
        if ".mlp.experts." in name and name.endswith(".weight"):
            packed = torch.randint(
                0, 256, (value.shape[0], value.shape[1] // 2), dtype=torch.uint8
            )
            weights[name] = packed
            scales[name + "_scale"] = torch.full(
                (value.shape[0], value.shape[1] // 32), 120, dtype=torch.uint8
            )
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
            dense[name] = (table[codes.long()] / 128).bfloat16()
        elif name.endswith("qkv_proj.weight"):
            layer = model.model.layers[int(name.split(".")[2])]
            attention = layer.self_attn
            sizes = (
                attention.q_size // tp,
                attention.k_size // tp,
                attention.v_size // tp,
            )
            raw = torch.randint(-4, 5, value.shape).to(torch.float8_e4m3fn)
            weights[name] = raw
            local_scale_rows = (sum(sizes) + 127) // 128
            scales[name + "_scale_inv"] = torch.cat(
                [
                    torch.full((local_scale_rows, 1), (rank + 1) / 256)
                    for rank in range(tp)
                ]
            )
            parts = [
                (shard.float() * ((rank + 1) / 256)).split(sizes)
                for rank, shard in enumerate(raw.chunk(tp))
            ]
            dense[name] = torch.cat([p[i] for i in range(3) for p in parts]).bfloat16()
        elif name.startswith("model.layers.0.mlp.") and name.endswith(".weight"):
            raw = torch.randint(-4, 5, value.shape).to(torch.float8_e4m3fn)
            weights[name] = raw
            scales[name + "_scale_inv"] = torch.full((1, 1), 1 / 128)
            dense[name] = (raw.float() / 128).bfloat16()
        else:
            weights[name] = value
    auxiliary = {
        "model.mtp.layers.0.proj.weight": torch.ones(8, 8).to(torch.float8_e4m3fn),
        "model.mtp.layers.0.proj.weight_scale_inv": torch.full((1, 1), 0.25),
    }
    weights.update(auxiliary)
    save_file(weights, path / "weights.safetensors")
    save_file(scales, path / "scales.safetensors")
    mapping = {
        **dict.fromkeys(weights, "weights.safetensors"),
        **dict.fromkeys(scales, "scales.safetensors"),
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    model.load_state_dict(dense)
    return model, {
        **auxiliary,
        "visual.weight": weights["visual.weight"],
        "audio_encoder.weight": weights["audio_encoder.weight"],
        "model.layers.0.mlp.up_proj.weight": dense["model.layers.0.mlp.up_proj.weight"],
    }


@pytest.mark.parametrize(
    "tp,group_size,dtype,explicit_decoder",
    [
        (4, 64, "auto", True),
        (8, 128, "auto", False),
        (4, 64, torch.float16, False),
        (4, 64, torch.float16, True),
        (4, 64, torch.bfloat16, False),
    ],
)
def test_mimo_cpu_gptq_roundtrip(
    tmp_path: Path,
    tp: int,
    group_size: int,
    dtype: str | torch.dtype,
    explicit_decoder: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_threads: None,
) -> None:
    # The lazy loader's virtual pool requires two CPU workers, even on CI VMs.
    monkeypatch.setenv("GPTQMODEL_CPU_WORKERS", "2")
    if dtype == "auto":
        # A host accelerator must not override the explicit CPU quantization device.
        select_device = loader_module.auto_select_device
        monkeypatch.setattr(
            loader_module,
            "auto_select_device",
            lambda device, backend: (
                loader_module.DEVICE.MPS
                if device is None else select_device(device, backend)
            ),
        )
    source, output = tmp_path / "source", tmp_path / "quantized"
    execution_dtype = torch.bfloat16 if dtype == "auto" else dtype
    baseline, preserved = make_source(source, tp, dtype=execution_dtype)
    config = QuantizeConfig(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        device="cpu",
        offload_to_disk=True,
        offload_to_disk_path=str(tmp_path / "offload"),
        preprocessors=(
            [AutoModuleDecoderConfig(target_dtype=execution_dtype)]
            if explicit_decoder
            else []
        ),
        dynamic={r"-:^model\.layers\.0\.mlp\.up_proj$": {}},
        fallback=None,
        moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all")),
    )
    model = GPTQModel.load(
        str(source),
        quantize_config=config,
        backend=BACKEND.TORCH,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    decoders = [
        item
        for item in config.preprocessors
        if isinstance(item, AutoModuleDecoderConfig)
    ]
    assert len(decoders) == 1
    assert decoders[0].target_dtype == execution_dtype
    assert model.turtle_model is not None
    model.pre_quantize_generate_hook_start()
    for name, module in model.model.named_children():
        model.shell_module_materialize(
            target_submodule=module, device=torch.device("cpu"), module_path=name
        )
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    with torch.no_grad():
        expected = baseline(input_ids=ids, use_cache=False).logits
        actual = model.model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del model, baseline
    model = GPTQModel.load(
        str(source),
        quantize_config=config,
        backend=BACKEND.TORCH,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    assert model.model.model.layers[0].self_attn.qkv_proj.weight.is_meta
    captured_batches = []
    cache_inputs = StageInputsCapture.cache_inputs

    def record_capture(
        stage: StageInputsCapture, *args: object, **kwargs: object
    ) -> InputCache:
        first_layer = stage.gptq_model.model.model.layers[0]
        assert first_layer.self_attn.qkv_proj.weight.is_meta
        result = cache_inputs(stage, *args, **kwargs)
        assert first_layer.self_attn.qkv_proj.weight.is_meta
        assert all(torch.isfinite(batch[0]).all() for batch in result.layer_inputs)
        assert all(batch[0].dtype == execution_dtype for batch in result.layer_inputs)
        captured_batches.append(len(result.layer_inputs))
        return result

    monkeypatch.setattr(StageInputsCapture, "cache_inputs", record_capture)
    quantized_modules = []
    quantize = GPTQ.quantize

    def reject_fallback(*args: object, **kwargs: object) -> None:
        pytest.fail("MiMo acceptance must not use weight-only fallback")

    def record_quantize(task: GPTQ, *args: object, **kwargs: object) -> tuple:
        assert task.fallback is None
        assert task.qcfg.mock_quantization is False
        assert task.nsamples > 0
        result = quantize(task, *args, **kwargs)
        assert isinstance(result[5], (int, float)) and math.isfinite(result[5])
        assert result[5] != 999999999
        quantized_modules.append(task._named_module.full_name)
        return result

    monkeypatch.setattr(GPTQ, "_fallback_quantize", reject_fallback)
    monkeypatch.setattr(GPTQ, "quantize", record_quantize)
    calibration = [torch.randint(4, 128, (1, 16)).tolist()[0] for _ in range(4)]
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
        calibration_data_min_length=1,
    )
    assert captured_batches == [len(calibration)]
    assert model.quant_log
    model.save(str(output))
    saved = json.loads((output / "config.json").read_text())
    assert saved["quantization_config"]["quant_method"] == "gptq"
    source_quantization = json.loads((source / "config.json").read_text())[
        "quantization_config"
    ]
    assert saved["mimo_mtp_source_quantization_config"] == source_quantization
    saved_tensors = {}
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as reader:
            saved_tensors.update({key: reader.get_tensor(key) for key in reader.keys()})
    for key, value in preserved.items():
        if not key.startswith("model.mtp."):
            value = value.to(execution_dtype)
        assert torch.equal(
            saved_tensors[key].view(torch.uint8), value.view(torch.uint8)
        )
    expected_quantized = (
        {
            f"model.layers.{layer}.self_attn.{projection}.qweight"
            for layer in range(2)
            for projection in ("qkv_proj", "o_proj")
        }
        | {
            f"model.layers.0.mlp.{projection}.qweight"
            for projection in ("gate_proj", "down_proj")
        }
        | {
            f"model.layers.1.mlp.experts.{expert}.{projection}.qweight"
            for expert in range(4)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
    )
    assert len(quantized_modules) == len(expected_quantized) == 18
    assert {name + ".qweight" for name in quantized_modules} == expected_quantized
    assert {key for key in saved_tensors if key.endswith(".qweight")} == (
        expected_quantized
    )
    assert not any(
        "weight_scale" in key
        for key in saved_tensors
        if key.startswith("model.layers.")
    )
    reloaded = GPTQModel.load(
        str(output),
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        trust_remote_code=True,
        dtype=execution_dtype,
        attn_implementation="eager",
    )
    assert (
        reloaded.model.config.mimo_mtp_source_quantization_config == source_quantization
    )
    with torch.no_grad():
        full = reloaded.model(input_ids=ids, use_cache=False).logits
        prefill = reloaded.model(input_ids=ids[:, :-1], use_cache=True)
        step = reloaded.model(
            input_ids=ids[:, -1:],
            use_cache=True,
            past_key_values=prefill.past_key_values,
        ).logits
    assert torch.isfinite(full).all()
    torch.testing.assert_close(step, full[:, -1:], atol=0.02, rtol=0.02)
    generated = reloaded.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=None,
    )
    assert generated.shape == (1, 12)
    torch.save(full, tmp_path / "expected_logits.pt")
    script = """
import sys
from pathlib import Path
import torch
from gptqmodel import GPTQModel, BACKEND

torch.set_num_threads(2)
output = Path(sys.argv[1])
model = GPTQModel.load(
    str(output), backend=BACKEND.GPTQ_TORCH, device="cpu",
    trust_remote_code=True, attn_implementation="eager",
    dtype=getattr(torch, sys.argv[2]),
)
ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
with torch.no_grad():
    actual = model.model(input_ids=ids, use_cache=False).logits
expected = torch.load(output.parent / "expected_logits.pt", weights_only=True)
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            script,
            str(output),
            str(execution_dtype).removeprefix("torch."),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "dtype,decoder_dtype,error",
    [
        (torch.float32, None, "requires dtype=torch.float16 or dtype=torch.bfloat16"),
        (torch.float16, torch.bfloat16, "must match the model dtype"),
        (torch.bfloat16, torch.float16, "must match the model dtype"),
    ],
)
def test_mimo_quantization_rejects_incompatible_dtypes_before_construction(
    tmp_path: Path,
    dtype: torch.dtype,
    decoder_dtype: torch.dtype | None,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    cpu_threads: None,
) -> None:
    source = tmp_path / "source"
    make_source(source, 4)
    config = QuantizeConfig(
        bits=4,
        device="cpu",
        offload_to_disk_path=str(tmp_path / "offload"),
        preprocessors=(
            [AutoModuleDecoderConfig(target_dtype=decoder_dtype)]
            if decoder_dtype is not None
            else []
        ),
    )

    def reject_construction(*args: object, **kwargs: object) -> None:
        pytest.fail("Incompatible MiMo dtypes must fail before tokenizer/model load")

    monkeypatch.setattr(loader_module, "load_hf_tokenizer", reject_construction)
    with pytest.raises(ValueError, match=error):
        GPTQModel.load(
            str(source),
            quantize_config=config,
            backend=BACKEND.TORCH,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="eager",
        )


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("explicit_decoder", [False, True])
def test_mimo_rejects_awq_before_construction(
    tmp_path: Path,
    tp: int,
    explicit_decoder: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_threads: None,
) -> None:
    source = tmp_path / "source"
    make_source(source, tp)
    config = AWQConfig(
        bits=4,
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path=str(tmp_path / "offload"),
        preprocessors=[AutoModuleDecoderConfig()] if explicit_decoder else [],
    )
    preprocessors = list(config.preprocessors)

    def reject_construction(*args: object, **kwargs: object) -> None:
        pytest.fail("MiMo AWQ must fail before tokenizer/model load")

    monkeypatch.setattr(loader_module, "load_hf_tokenizer", reject_construction)
    with pytest.raises(ValueError, match="MiMo mixed-source AWQ.*use GPTQ"):
        GPTQModel.load(
            str(source),
            quantize_config=config,
            backend=BACKEND.TORCH,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    assert config.preprocessors == preprocessors
    assert config.offload_to_disk is False
    assert not hasattr(config, "_native_floatx_forward_plan")
