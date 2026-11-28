# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resume keeps the AWQ scale dtype of the packed module (#6153).

``AwqGEMMLinear.pack`` stores ``scales`` at the source model width (fp16 or
bf16), while ``restore_packed_module`` rebuilt the template at the kernel's fp16
default. Resuming a stopped BF16 AWQ run then failed the tensor-schema
comparison for every module, so checkpointed AWQ was unusable on BF16 models
(BF16 is the default dtype of most current checkpoints). The GPTQ path was not
affected because GPTQ scales are always stored as FP16.

These tests drive the real adapter capture and restore path with a synthetic
packed AWQ bundle.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.gptq_checkpoint import QuantizationCheckpointAdapter
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.quantization.config import QuantizeConfig


def _awq_module(config, *, scales_dtype):
    module = AwqGEMMLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=896,
        out_features=896,
        bias=True,
        format=config.format,
        register_buffers=True,
    )
    with torch.no_grad():
        module.qweight.random_()
        module.qzeros.random_()
        module.scales = module.scales.to(scales_dtype)
        module.bias = module.bias.to(scales_dtype)
    return module


def _adapter(model, config, offload):
    adapter = QuantizationCheckpointAdapter()
    adapter.model = SimpleNamespace(
        model=model,
        quantize_config=config,
        qlinear_kernel=AwqGEMMLinear,
        lm_head="lm_head",
    )
    adapter.execution = SimpleNamespace(
        execution_state_dict=dict, load_execution_state_dict=lambda state: None
    )
    adapter.processors = ()
    adapter.processor = SimpleNamespace(lock=threading.Lock())
    adapter.shared_state = {}
    adapter.offload = offload
    adapter._artifacts = {}
    return adapter


def _capture(adapter):
    state, payloads = adapter.capture()
    artifacts = {}
    for name, payload in payloads.items():
        path = adapter.offload / "bundles" / f"{name}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[name] = path
    return state, artifacts


def _config():
    return QuantizeConfig(bits=4, group_size=128, format="gemm", desc_act=False)


def test_restore_preserves_a_bf16_awq_scale_dtype(tmp_path):
    config = _config()
    model = torch.nn.ModuleDict({"proj": _awq_module(config, scales_dtype=torch.bfloat16)})
    adapter = _adapter(model, config, tmp_path)

    state, artifacts = _capture(adapter)

    # This raised ``checkpoint tensor schema differs: proj`` before the fix.
    adapter.restore(state, artifacts)

    assert state["modules"]["proj"]["scales_dtype"] == "torch.bfloat16"
    assert model.proj.scales.dtype is torch.bfloat16
    index = (tmp_path / "proj" / "index.json").read_text()
    assert '"dtype": "bfloat16"' in index


def test_fp16_awq_scales_need_no_schema_entry(tmp_path):
    """FP16 is the restore template default, so the config stays unchanged."""

    config = _config()
    model = torch.nn.ModuleDict({"proj": _awq_module(config, scales_dtype=torch.float16)})
    adapter = _adapter(model, config, tmp_path)

    state, artifacts = _capture(adapter)

    assert "scales_dtype" not in state["modules"]["proj"]
    adapter.restore(state, artifacts)
    assert model.proj.scales.dtype is torch.float16


def test_restore_still_rejects_a_genuinely_different_scale_dtype(tmp_path):
    config = _config()
    model = torch.nn.ModuleDict({"proj": _awq_module(config, scales_dtype=torch.bfloat16)})
    adapter = _adapter(model, config, tmp_path)

    state, artifacts = _capture(adapter)
    # Corrupt the bundle: the recorded bf16 schema no longer matches the file.
    bundle = Path(artifacts["proj"])
    tensors = load_file(str(bundle))
    tensors["scales"] = tensors["scales"].to(torch.float32)
    save_file(tensors, str(bundle), metadata={"format": "pt"})

    with pytest.raises(CheckpointError, match="checkpoint tensor schema differs: proj"):
        adapter.restore(state, artifacts)
