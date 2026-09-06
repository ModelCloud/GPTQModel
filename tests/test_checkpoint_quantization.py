# SPDX-License-Identifier: Apache-2.0
"""Real two-layer quantization equivalence; no replay of completed layers."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from gptqmodel import BACKEND, CheckpointConfig, GPTQModel, QuantizeConfig
from gptqmodel.looper.checkpoint import CheckpointExtension, CheckpointStopped
from gptqmodel.looper.gptq_checkpoint import GPTQCheckpointAdapter
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


def assert_identical_tensors(expected, actual):
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        assert (
            tensor.shape == actual[name].shape and tensor.dtype == actual[name].dtype
        ), name
        # Compare exact element bits, including signed zero, not a tolerance
        # or dtype-coercing numerical comparison.
        assert torch.equal(
            tensor.contiguous().reshape(-1).view(torch.uint8),
            actual[name].contiguous().reshape(-1).view(torch.uint8),
        ), name


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
@pytest.mark.parametrize(
    "mode",
    ["kill-before", "kill-after", "error", "int", "kill-hessian", "kill-hessian-early"],
)
def test_subprocess_recovery(tmp_path, family, mode, device="cpu"):
    torch.manual_seed(1234)
    native = tmp_path / "native"
    args = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**args))
    else:
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **args,
                moe_intermediate_size=32,
                num_experts=4,
                num_experts_per_tok=2,
            )
        )
    model.save_pretrained(native)
    _build_local_tokenizer(native)
    driver = Path(__file__).with_name("checkpoint_quantization_driver.py")
    audit = mode.startswith("kill-hessian")

    def run(mode, expected):
        output = tmp_path / mode
        result = subprocess.run(
            [
                sys.executable,
                str(driver),
                "--model",
                str(native),
                "--checkpoint",
                str(tmp_path / "checkpoints"),
                "--output",
                str(output),
                "--mode",
                mode,
                "--device",
                device,
                *(["--audit-hessians"] if audit else []),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == expected, (
            result.stdout.decode()[-10000:] + result.stderr.decode()[-10000:]
        )
        return result, output

    reference, baseline_path = run("baseline", 0)
    interrupted, _ = run(
        mode, 74 if mode == "error" else 75 if mode == "int" else -signal.SIGKILL
    )
    if audit:
        root = tmp_path / "checkpoints"
        current = json.loads((root / "CURRENT").read_text())
        manifest = json.loads(
            (root / "objects" / current["generations"][0]).read_text()
        )
        assert manifest["cursor"] == 1  # Layer 1's partial work never became resumable.
    result, resumed_path = run("run", 0)
    assert b"CHECKPOINT_BOUNDARY 0" not in result.stdout
    assert b"CHECKPOINT_BOUNDARY 1" in result.stdout
    baseline = load_file(baseline_path / "model.safetensors")
    resumed = load_file(resumed_path / "model.safetensors")
    assert_identical_tensors(baseline, resumed)
    if audit:

        def events(output):
            decoder = json.JSONDecoder()
            return [
                decoder.raw_decode(line.split("HESSIAN_AUDIT ", 1)[1])[0]
                for line in output.decode().splitlines()
                if "HESSIAN_AUDIT " in line
            ]

        expected = [
            event for event in events(reference.stdout) if ".layers.1." in event["name"]
        ]
        actual = events(result.stdout)
        assert actual and all(".layers.1." in event["name"] for event in actual)
        # Compare the full set/multiplicity of task initializations and input
        # batches, plus sample counts and exact pre-quantization Hessian hashes.
        canonical = lambda records: sorted(
            json.dumps(event, sort_keys=True) for event in records
        )
        assert canonical(actual) == canonical(expected)
        killed = [
            event for event in events(interrupted.stdout) if event["event"] == "kill"
        ]
        assert len(killed) == 1 and killed[0]["fwd_counter"] == 2
        assert killed[0]["nsamples"] > 0 and killed[0]["partial_hashes"]
        assert any(
            event["event"] == "final" and event["name"] == killed[0]["name"]
            for event in actual
        )
        if mode == "kill-hessian":
            assert any(
                event["event"] == "final" and ".layers.1." in event["name"]
                for event in events(interrupted.stdout)
            ), "kill must follow already-quantized subsets"
        assert any(
            event["event"] == "fresh"
            and event["name"] == killed[0]["name"]
            and event["nsamples"] == event["fwd_counter"] == 0
            for event in actual
        )


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
def test_quantized_tensors_equal_after_resume(tmp_path, monkeypatch, family):
    torch.manual_seed(1234)
    native = tmp_path / "native"
    config_args = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if family == "llama":
        native_model = LlamaForCausalLM(LlamaConfig(**config_args))
    else:
        native_model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **config_args,
                moe_intermediate_size=32,
                num_experts=4,
                num_experts_per_tok=2,
            )
        )
    native_model.save_pretrained(native)
    tokenizer = _build_local_tokenizer(native)
    calibration = _build_calibration_dataset(tokenizer)

    def run(output, checkpoint):
        torch.manual_seed(6789)
        config = QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cpu",
            offload_to_disk=True,
            offload_to_disk_path=str(tmp_path / f"offload-{output}"),
            moe=MoEConfig(routing=ExpertsRoutingOverride())
            if family == "qwen3_moe"
            else None,
        )
        model = GPTQModel.load(
            str(native), quantize_config=config, backend=BACKEND.TORCH
        )
        model.quantize(
            calibration,
            batch_size=1,
            backend=BACKEND.TORCH,
            calibration_data_min_length=1,
            checkpoint=checkpoint,
        )
        model.save(tmp_path / output)
        return load_file(tmp_path / output / "model.safetensors")

    baseline = run("baseline", None)
    original = CheckpointExtension.on_boundary

    def stop(extension, boundary):
        if boundary.step.index == 0:
            signal.raise_signal(signal.SIGTERM)
        original(extension, boundary)

    checkpoint = CheckpointConfig(tmp_path / "checkpoints")
    with monkeypatch.context() as patch:
        patch.setattr(CheckpointExtension, "on_boundary", stop)
        with pytest.raises(CheckpointStopped):
            run("interrupted", checkpoint)
    restored = []
    original_restore = GPTQCheckpointAdapter.restore

    def restore(adapter, state, artifacts):
        original_restore(adapter, state, artifacts)
        restored.extend(artifacts)
        assert all(
            tensor.is_meta
            for name in artifacts
            for tensor in adapter.model.model.get_submodule(name).state_dict().values()
        )

    monkeypatch.setattr(GPTQCheckpointAdapter, "restore", restore)
    resumed = run("resumed", checkpoint)
    assert restored and all("layers.0." in name for name in restored)
    completed = run("completed", checkpoint)
    assert_identical_tensors(baseline, resumed)
    assert_identical_tensors(baseline, completed)
