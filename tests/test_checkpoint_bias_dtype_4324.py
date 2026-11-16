# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resume keeps the bias dtype of the quantized module (#4324).

``restore_packed_module`` rebuilds a packed layer on the meta device. The
kernel allocates its bias placeholder at fp16 unless told otherwise, while a
bf16/fp32 model keeps a bias at the model width. The restore schema check then
rejected every such checkpoint. These tests drive the real adapter capture and
restore path with a synthetic packed bundle.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.gptq_checkpoint import QuantizationCheckpointAdapter
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig


def _packed_module(config, *, bias_dtype):
    module = TorchLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=True,
        format=config.format,
        register_buffers=True,
    )
    with torch.no_grad():
        module.bias = module.bias.to(bias_dtype)
        module.qweight.random_()
        module.qzeros.random_()
        module.scales.random_()
    return module


def _adapter(model, config, offload):
    adapter = QuantizationCheckpointAdapter()
    adapter.model = SimpleNamespace(
        model=model,
        quantize_config=config,
        qlinear_kernel=TorchLinear,
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


def _capture(adapter, tmp_path):
    state, payloads = adapter.capture()
    artifacts = {}
    for name, payload in payloads.items():
        path = tmp_path / "bundles" / f"{name}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[name] = path
    return state, artifacts


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_restore_preserves_non_fp16_bias_dtype(tmp_path, bias_dtype):
    config = QuantizeConfig(bits=4, group_size=32, format="gptq", desc_act=False)
    module = _packed_module(config, bias_dtype=bias_dtype)
    model = torch.nn.ModuleDict({"proj": module})
    adapter = _adapter(model, config, tmp_path)

    state, artifacts = _capture(adapter, tmp_path)

    # This raised ``checkpoint tensor schema differs: proj`` before the fix.
    adapter.restore(state, artifacts)

    assert state["modules"]["proj"]["bias_dtype"] == str(bias_dtype)
    assert model.proj.bias.dtype == bias_dtype
    index = (tmp_path / "proj" / "index.json").read_text()
    assert f'"dtype": "{str(bias_dtype).removeprefix("torch.")}"' in index


def test_restore_still_rejects_a_genuinely_different_bias_dtype(tmp_path):
    config = QuantizeConfig(bits=4, group_size=32, format="gptq", desc_act=False)
    module = _packed_module(config, bias_dtype=torch.bfloat16)
    model = torch.nn.ModuleDict({"proj": module})
    adapter = _adapter(model, config, tmp_path)

    state, artifacts = _capture(adapter, tmp_path)
    # Corrupt the bundle: the recorded bf16 schema no longer matches the file.
    bundle = Path(artifacts["proj"])
    tensors = load_file(str(bundle))
    tensors["bias"] = tensors["bias"].to(torch.float32)
    save_file(tensors, str(bundle), metadata={"format": "pt"})

    with pytest.raises(CheckpointError, match="checkpoint tensor schema differs: proj"):
        adapter.restore(state, artifacts)
