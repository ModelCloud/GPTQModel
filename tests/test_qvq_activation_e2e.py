# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end QVQ P32/A8 quantize, save, reload, and CUDA inference coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.nn_modules.qvq_fp8_cache import QVQFP8DynamicCache
from gptqmodel.quantization import FORMAT, QVQConfig
from gptqmodel.utils.qvq_cuda import qvq_cuda_supported

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(
        not qvq_cuda_supported(),
        reason="requires an NVIDIA CUDA device supported by QVQ",
    ),
]

_CALIBRATION_TEXTS = [
    "tiny qvq calibration sample one exercises attention and feed forward projections",
    "tiny qvq calibration sample two targets dynamic fp8 activation quantization",
    "another deterministic calibration row keeps the complete model lifecycle covered",
] * 2


def _build_tiny_llama_fixture(model_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    torch.manual_seed(41)
    LlamaForCausalLM(config).save_pretrained(model_dir)
    return fast_tokenizer


def _calibration_dataset(tokenizer) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        for text in _CALIBRATION_TEXTS
        for encoded in (tokenizer(text, return_tensors="pt"),)
    ]


def test_qvq_p32_a8_quantize_save_reload_and_native_inference(tmp_path: Path):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip(
            "native QVQ E4M3 input requires NVIDIA compute capability 8.9 or newer"
        )

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir()
    tokenizer = _build_tiny_llama_fixture(model_dir)
    calibration = _calibration_dataset(tokenizer)
    quantize_config = QVQConfig(
        bits=3.5,
        format="v2b2-g32",
        rounding="block_ldlq",
        activation={"kernel_mode": "require"},
        device="cuda:0",
        offload_to_disk=False,
    )

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    quant_log = model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.QVQ,
        calibration_data_min_length=1,
    )
    replay_stats = [
        row["fp8_target_replay"]
        for rows in quant_log.values()
        for row in rows
        if isinstance(row, dict)
        and row.get("fp8_target_replay") is not None
    ]
    assert len(replay_stats) == 7
    assert all(
        stats["operand_dtype"] == "float8_e4m3fn"
        and stats["native_first_executed"] > 0
        and stats["native_second_executed"] > 0
        for stats in replay_stats
    )

    encoded = tokenizer("tiny qvq calibration sample", return_tensors="pt").to("cuda:0")
    model.model.to("cuda:0").eval()
    with torch.inference_mode():
        in_memory_logits = model.model(**encoded).logits.float()
    model.save(quantized_dir)
    del model
    torch.cuda.empty_cache()

    saved_config = json.loads(
        (quantized_dir / "config.json").read_text(encoding="utf-8")
    )
    metadata = saved_config["quantization_config"]
    assert metadata["method"] == "qvq"
    assert metadata["bits"] == 3.5
    assert metadata["format"] == FORMAT.QVQ_V2B2_P32.value
    assert metadata["activation"] == {
        "bits": 8,
        "format": "float8_e4m3fn",
        "kernel_mode": "require",
        "replay_max_rows": 2048,
        "replay_passes": 1,
        "replay_validation_fraction": 0.125,
        "scale_method": "dynamic_per_token",
        "target": "p32_operand",
    }

    tensor_keys = set()
    for tensor_file in quantized_dir.glob("*.safetensors"):
        with safe_open(str(tensor_file), framework="pt") as handle:
            tensor_keys.update(handle.keys())
    assert any(key.endswith(".trellis") for key in tensor_keys)
    assert any(key.endswith(".bank_ids") for key in tensor_keys)
    assert any(key.endswith(".bank_alt_id") for key in tensor_keys)

    reloaded = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    qvq_layers = [
        module for module in reloaded.model.modules() if isinstance(module, QVQLinear)
    ]
    assert len(qvq_layers) == 7
    assert all(
        layer.v2b2_p32 and layer.activation is not None
        for layer in qvq_layers
    )
    assert (
        reloaded.quantize_config.activation
        == quantize_config.activation
    )

    reloaded.model.eval()
    with torch.inference_mode():
        native_output = reloaded.model(**encoded)
        native_logits = native_output.logits.float()
    assert isinstance(native_output.past_key_values, QVQFP8DynamicCache)
    cache_telemetry = native_output.past_key_values.telemetry()
    assert cache_telemetry["all_payloads_fp8"] is True
    assert cache_telemetry["no_full_precision_residual"] is True
    assert cache_telemetry["initialized_layer_count"] == 1
    assert cache_telemetry["payload_dtypes"] == ["torch.float8_e4m3fn"]
    # The dynamic CUDA cache allocates one full page. Its physical ratio is
    # therefore the logical E4M3+row-scale ratio times capacity/utilized rows.
    assert cache_telemetry["storage_ratio_vs_dense"] == pytest.approx(
        0.625
        * cache_telemetry["capacities"][0]
        / cache_telemetry["sequence_lengths"][0]
    )
    kernel_telemetry = [layer.qvq_fp8_kernel_telemetry() for layer in qvq_layers]
    assert all(
        telemetry["requested"] > 0
        and telemetry["requested"] == telemetry["executed"]
        and telemetry["fallback"] == 0
        and telemetry["rejected"] == 0
        for telemetry in kernel_telemetry
    )
    torch.testing.assert_close(native_logits, in_memory_logits, atol=2e-3, rtol=0)
    generated = reloaded.generate(
        **encoded,
        min_new_tokens=2,
        max_new_tokens=2,
        do_sample=False,
        return_dict_in_generate=True,
    )
    assert generated.sequences.shape[-1] == encoded["input_ids"].shape[-1] + 2
    assert isinstance(generated.past_key_values, QVQFP8DynamicCache)
    generated_cache_telemetry = generated.past_key_values.telemetry()
    assert generated_cache_telemetry["all_payloads_fp8"] is True
    assert generated_cache_telemetry["sequence_lengths"] == [
        generated.sequences.shape[-1] - 1
    ]
