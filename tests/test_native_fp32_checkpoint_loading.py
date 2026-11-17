# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native dtype-policy boundaries alongside real packed checkpoint state."""

import json

import pytest
import torch
from accelerate.utils.offload import load_offloaded_weight
from safetensors.torch import load_file, save_file
from torch import nn

from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


class PolicyModel(nn.Module):
    _keep_in_fp32_modules_strict = [
        r"(^|\.)gate\.weight$", r"(^|\.)moe_statics(\.|$)", r"^packed.*", r"^exl3.*",
    ]
    _keep_in_fp32_modules = [r"(^|\.)legacy(\.|$)"]

    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        for name in ("gate", "not_gate", "gate_extra", "legacy", "projection"):
            setattr(self.block, name, nn.Linear(2, 2, bias=True))
        for name in ("moe_statics", "moe_statics_extra"):
            part = nn.Module()
            part.register_buffer("correction", torch.tensor([1.0003, -2.0007]))
            part.register_buffer("count", torch.tensor([7], dtype=torch.int64))
            part.register_buffer("scratch", torch.tensor([0.123456]), persistent=False)
            setattr(self.block, name, part)
        self.packed = AwqTorchLinear(
            bits=4, group_size=16, desc_act=False, sym=True,
            in_features=32, out_features=32, bias=False, register_buffers=True,
        )
        self.exl3 = ExllamaV3TorchLinear.from_tensors(
            in_features=128, out_features=128, name="exl3",
            tensors={"trellis": torch.zeros((8, 8, 64), dtype=torch.int16),
                     "suh": torch.full((128,), 0.12347, dtype=torch.float16),
                     "svh": torch.full((128,), -1.2345, dtype=torch.float16)},
        )
        with torch.no_grad():
            for param in self.parameters():
                param.copy_(torch.tensor([1.0003, -2.0007, 0.12345678, 123.98765]).repeat(2)[:param.numel()].reshape_as(param))
            self.packed.scales.fill_(0.12347)

    def tie_weights(self):
        pass


def write_checkpoint(path, *, sharded=False, prefix="", source_dtype=torch.float32):
    source = PolicyModel()
    values = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    for name, value in values.items():
        if value.is_floating_point() and not name.startswith(("packed.", "exl3.")):
            values[name] = value.to(source_dtype)
    stored = {prefix + name: value for name, value in values.items()}
    if not sharded:
        target = path / "model.safetensors"
        save_file(stored, target, metadata={"format": "pt"})
    else:
        names = list(stored)
        mapping = {}
        for index, subset in enumerate((names[::2], names[1::2])):
            filename = f"model-{index}.safetensors"
            save_file({name: stored[name] for name in subset}, path / filename, metadata={"format": "pt"})
            mapping.update(dict.fromkeys(subset, filename))
        target = path / "model.safetensors.index.json"
        target.write_text(json.dumps({"weight_map": mapping}))
    return str(target), values


def expected_dtype(name, value, dtype):
    if not value.is_floating_point():
        return value.dtype
    if name.startswith(("packed.", "exl3.")):
        return torch.float32 if dtype == torch.float32 else value.dtype
    if name == "block.gate.weight" or name == "block.moe_statics.correction":
        return torch.float32
    if dtype == torch.float16 and name.startswith("block.legacy."):
        return torch.float32
    return dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("prefix", ["", "upstream_"])
@pytest.mark.parametrize("offload_state_dict", [False, True])
def test_native_policy_restores_exact_dense_values_without_widening_packed_state(
    tmp_path, dtype, sharded, prefix, offload_state_dict,
):
    checkpoint, values = write_checkpoint(tmp_path, sharded=sharded, prefix=prefix)
    model = PolicyModel().to(dtype=dtype)
    # Quantization metadata follows its own checkpoint dtype, even if a native
    # policy's broad regex also matches the packed module.
    kwargs = {"key_converter": lambda name: name.removeprefix(prefix)} if prefix else {}
    load_checkpoint_in_model_then_tie_weights(
        model, checkpoint=checkpoint, dtype=dtype, device_map={"": "cpu"},
        offload_state_dict=offload_state_dict, **kwargs,
    )
    assert model.state_dict().keys() == values.keys()
    for name, value in values.items():
        torch.testing.assert_close(model.state_dict()[name], value.to(expected_dtype(name, value, dtype)),
                                   rtol=0, atol=0, msg=name)
    assert model.block.moe_statics.scratch.dtype == dtype
    assert not list(tmp_path.glob("gptqmodel-key-remap-*"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offload_buffers", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.float32])
def test_native_policy_disk_offload_keeps_payload_and_index_consistent(tmp_path, dtype, offload_buffers, source_dtype):
    checkpoint, values = write_checkpoint(tmp_path, source_dtype=source_dtype, sharded=True, prefix="upstream_")
    model = PolicyModel().to(dtype=dtype)
    folder = tmp_path / "offloaded"
    load_checkpoint_in_model_then_tie_weights(
        model, checkpoint=checkpoint, dtype=dtype,
        device_map={"block.gate": "disk", "block.moe_statics": "disk", "": "cpu"},
        offload_folder=folder, offload_buffers=offload_buffers,
        key_converter=lambda name: name.removeprefix("upstream_"),
    )
    index = json.loads((folder / "index.json").read_text())
    for name in ("block.gate.weight", "block.moe_statics.correction"):
        module_name, _, leaf = name.rpartition(".")
        slot = getattr(model.get_submodule(module_name), leaf)
        if name.endswith("correction") and not offload_buffers:
            torch.testing.assert_close(slot, values[name].float(), rtol=0, atol=0)
        else:
            assert slot.device.type == "meta" and slot.dtype == torch.float32
            assert index[name]["dtype"] == "float32"
            actual = load_offloaded_weight(folder / (name + ".dat"), index[name])
            torch.testing.assert_close(actual, values[name].float(), rtol=0, atol=0)
    # Updating native slots must keep unrelated offload metadata and bytes.
    assert "block.gate.bias" in index
    actual_bias = load_offloaded_weight(folder / "block.gate.bias.dat", index["block.gate.bias"])
    torch.testing.assert_close(actual_bias, values["block.gate.bias"], rtol=0, atol=0)
    assert not list(tmp_path.glob("gptqmodel-key-remap-*"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_policy_rejects_invalid_regex_then_retries_same_model(tmp_path, dtype):
    checkpoint, values = write_checkpoint(tmp_path)
    model = PolicyModel().to(dtype=dtype)
    model._keep_in_fp32_modules_strict = ["["]
    with pytest.raises(ValueError, match="native FP32"):
        load_checkpoint_in_model_then_tie_weights(model, checkpoint=checkpoint, dtype=dtype, device_map={"": "cpu"})
    model._keep_in_fp32_modules_strict = PolicyModel._keep_in_fp32_modules_strict
    load_checkpoint_in_model_then_tie_weights(model, checkpoint=checkpoint, dtype=dtype, device_map={"": "cpu"})
    torch.testing.assert_close(model.block.gate.weight, values["block.gate.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_policy_matches_transformers_unanchored_regex_semantics(tmp_path, dtype):
    checkpoint, values = write_checkpoint(tmp_path)
    model = PolicyModel().to(dtype=dtype)
    # Native globs are regex search (dot is a wildcard), NOT segment or suffix
    # equality. Here an unanchored rule deliberately includes not_gate.weight.
    model._keep_in_fp32_modules_strict = ["gate.weight"]
    model._keep_in_fp32_modules = None
    load_checkpoint_in_model_then_tie_weights(model, checkpoint=checkpoint, dtype=dtype, device_map={"": "cpu"})
    for name in ("block.gate.weight", "block.not_gate.weight"):
        module_name, _, leaf = name.rpartition(".")
        torch.testing.assert_close(getattr(model.get_submodule(module_name), leaf), values[name], rtol=0, atol=0)
    assert model.block.gate_extra.weight.dtype == dtype


def test_native_policy_handles_positional_load_and_existing_fp16_override(tmp_path):
    checkpoint, values = write_checkpoint(tmp_path, prefix="upstream_")
    model = PolicyModel().half()
    load_checkpoint_in_model_then_tie_weights(
        model, checkpoint, {"": "cpu"}, None, torch.float16,
        keep_in_fp32_modules=["block.projection"], key_converter=lambda name: name.removeprefix("upstream_"),
    )
    torch.testing.assert_close(model.block.gate.weight, values["block.gate.weight"], rtol=0, atol=0)
    torch.testing.assert_close(model.block.projection.weight, values["block.projection.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_policy_missing_tensor_cleans_remap_then_retries_same_model(tmp_path, dtype):
    checkpoint, values = write_checkpoint(tmp_path, prefix="upstream_")
    stored = load_file(checkpoint)
    broken = dict(stored)
    del broken["upstream_block.moe_statics.correction"]
    save_file(broken, checkpoint, metadata={"format": "pt"})
    model = PolicyModel().to(dtype=dtype)
    kwargs = {"dtype": dtype, "device_map": {"": "cpu"},
              "key_converter": lambda name: name.removeprefix("upstream_")}
    with pytest.raises(RuntimeError, match="missing from checkpoint.*block.moe_statics.correction"):
        load_checkpoint_in_model_then_tie_weights(model, checkpoint=checkpoint, **kwargs)
    assert not list(tmp_path.glob("gptqmodel-key-remap-*"))
    save_file(stored, checkpoint, metadata={"format": "pt"})
    load_checkpoint_in_model_then_tie_weights(model, checkpoint=checkpoint, **kwargs)
    for name, value in values.items():
        torch.testing.assert_close(model.state_dict()[name], value.to(expected_dtype(name, value, dtype)),
                                   rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_awq_reload_preserves_nonzero_native_router_bias_after_failure(tmp_path, dtype):
    # The independently owned native ERNIE qualification supplies the actual
    # checkpoint/tokenizer fixture, not a mocked model or loader.
    from test_ernie4_5_moe_local_port import checkpoint

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import AWQConfig

    source = tmp_path / "source"
    reference, _ = checkpoint(source, num_hidden_layers=2, intermediate_size=32, moe_intermediate_size=16)
    bias_name = "model.layers.1.mlp.gate.moe_statics.e_score_correction_bias"
    bias = reference.state_dict()[bias_name]
    bias.copy_(torch.tensor([0.001234567, -0.1234567, 0.2345678]))
    reference.save_pretrained(source)
    model = GPTQModel.load(
        str(source), quantize_config=AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=False),
        backend=BACKEND.TORCH, dtype=dtype, attn_implementation="eager",
    )
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]])
    model.quantize([{"input_ids": ids, "attention_mask": torch.ones_like(ids)}],
                   batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    torch.testing.assert_close(model.model.state_dict()[bias_name], bias, rtol=0, atol=0)
    expected = {name: value.clone() for name, value in model.model.state_dict().items()}
    with torch.inference_mode():
        logits = model.model(ids).logits
    destination = tmp_path / "packed"
    model.save(str(destination))
    payloads = list(destination.glob("*.safetensors"))
    assert len(payloads) == 1
    payload = payloads[0]
    stored = load_file(payload)
    verified = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                             dtype=dtype, attn_implementation="eager")
    torch.testing.assert_close(verified.model.state_dict()[bias_name], bias, rtol=0, atol=0)
    torch.testing.assert_close(verified.model.state_dict()["model.layers.1.mlp.gate.weight"],
                               reference.state_dict()["model.layers.1.mlp.gate.weight"], rtol=0, atol=0)
    broken = dict(stored)
    del broken[bias_name]
    save_file(broken, payload, metadata={"format": "pt"})
    default_dtype = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="missing from checkpoint.*e_score_correction_bias"):
        GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                      dtype=dtype, attn_implementation="eager")
    assert torch.get_default_dtype() == default_dtype
    assert not list(destination.glob("gptqmodel-key-remap-*"))
    save_file(stored, payload, metadata={"format": "pt"})
    restored = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                             dtype=dtype, attn_implementation="eager")
    assert restored.model.state_dict().keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0, msg=name)
    with torch.inference_mode():
        torch.testing.assert_close(restored.model(ids).logits, logits, rtol=1e-3, atol=1e-3)
