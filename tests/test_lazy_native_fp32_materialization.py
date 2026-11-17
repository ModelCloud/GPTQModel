# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native precision must survive checkpoint-backed AWQ materialization."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.klear import KlearQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from tests.test_klear_local_port import tiny_config
from tests.test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("offload", [True, False], ids=["lazy", "direct"])
def test_public_awq_native_fp32_pack_and_two_reloads(tmp_path, dtype, offload):
    torch.manual_seed(57)
    config = tiny_config(
        vocab_size=128, hidden_size=32, intermediate_size=32, moe_intermediate_size=32,
        num_hidden_layers=2, num_experts_per_tok=3, mlp_only_layers=[0], max_position_embeddings=256,
    )
    source = KlearQModel.loader.from_config(config, dtype=dtype).eval()
    bias_name = "model.layers.1.mlp.expert_bias"
    bias = torch.tensor([0.00391319, -0.002332191, 0.001397211], dtype=torch.float32)
    assert not torch.equal(bias, bias.to(dtype).float())
    with torch.no_grad():
        # Checkpoint truth must not depend on a shell constructor's default dtype.
        source.model.layers[1].mlp.expert_bias = bias.clone()
        source.model.layers[1].mlp.gate.weight.normal_(std=0.5)
    native = tmp_path / "native"
    source.save_pretrained(native, safe_serialization=True)
    torch.testing.assert_close(load_file(native / "model.safetensors")[bias_name], bias, rtol=0, atol=0)
    tokenizer = _build_local_tokenizer(native)
    calibration = _build_calibration_dataset(tokenizer)
    config = AWQConfig(
        bits=4, group_size=16, device="cpu", fallback=None, offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    model = GPTQModel.load(str(native), quantize_config=config, backend=BACKEND.TORCH,
                          dtype=dtype, attn_implementation="eager", trust_remote_code=False)
    assert (model.turtle_model is not None) is offload
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name: module for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 7 + 4 + 3 * 3 + 3
    actual = model.model.state_dict()[bias_name]
    torch.testing.assert_close(actual, bias, rtol=0, atol=0, msg=f"{bias_name} after quantize: {actual}")
    ids = calibration[0]["input_ids"]
    with torch.inference_mode():
        logits = model.model(ids, use_cache=False).logits
        assert torch.isfinite(logits).all()
    previous = {name: value.clone() for name, value in model.model.state_dict().items()}
    for attempt in range(2):
        destination = tmp_path / f"awq-{attempt}"
        model.save(destination)
        payload = next(destination.glob("*.safetensors"))
        stored = load_file(payload)
        # AWQ checkpoints omit the runtime-only qzeros layout markers.
        assert set(stored) == {name for name in previous if not name.endswith(".qzeros_format")}
        for name, value in stored.items():
            torch.testing.assert_close(value, previous[name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(stored[bias_name], bias, rtol=0, atol=0)
        assert not list(destination.glob("*.bin")) and not list(destination.glob("*.pt"))
        if attempt == 0:
            save_file({name: value for name, value in stored.items() if name != bias_name}, payload,
                      metadata={"format": "pt"})
            before = torch.get_default_dtype()
            with pytest.raises(RuntimeError, match="missing from checkpoint.*expert_bias"):
                GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=dtype,
                               attn_implementation="eager", trust_remote_code=False)
            assert torch.get_default_dtype() == before
            assert not list(destination.glob("gptqmodel-key-remap-*"))
            save_file(stored, payload, metadata={"format": "pt"})
        model = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=dtype,
                              attn_implementation="eager", trust_remote_code=False)
        assert set(model.model.state_dict()) == set(previous)
        for name, value in model.model.state_dict().items():
            torch.testing.assert_close(value, previous[name], rtol=0, atol=0, msg=name)
        with torch.inference_mode():
            torch.testing.assert_close(model.model(ids, use_cache=False).logits, logits, rtol=0, atol=0)
