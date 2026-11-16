# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Re-saving a loaded AWQ GEMV / GEMV_FAST checkpoint must resolve its shell (#4305).

``BaseQModel.save_quantized`` rebuilds a zero-filled export shell from the
checkpoint config and then loads the checkpoint into it. The shell never runs a
forward pass, so it must not require a CPU-runnable kernel. AWQ GEMV /
GEMV_FAST only ship accelerator kernels, and the old ``AUTO + CPU`` selection
raised ``ValueError: No valid quant linear`` and aborted every re-save.

These tests pin the layout-only selection for those formats, prove the packed
buffers still materialize on CPU, and keep the CPU-runnable GEMM/GPTQ selection
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import writer as writer_module
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.writer import _export_shell_device
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import make_quant


class _Stop(Exception):
    """Abort the shell build once it has asked for its kernel."""


def _awq_config(fmt: FORMAT, group_size: int) -> QuantizeConfig:
    qcfg = QuantizeConfig(method="awq", format=fmt, bits=4, group_size=group_size)
    if fmt == FORMAT.GEMV_FAST:
        # The quantize path pins this; a saved checkpoint carries it too.
        qcfg.pack_dtype = torch.int16
    return qcfg


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (FORMAT.GEMM, DEVICE.CPU),
        (FORMAT.GEMV, None),
        (FORMAT.GEMV_FAST, None),
    ],
)
def test_export_shell_device_follows_cpu_kernel_availability(fmt, expected):
    assert _export_shell_device(_awq_config(fmt, 128)) == expected


def test_export_shell_device_keeps_cpu_for_gptq():
    qcfg = QuantizeConfig(method="gptq", format=FORMAT.GPTQ_V2, bits=4, group_size=128)
    assert _export_shell_device(qcfg) == DEVICE.CPU


@pytest.mark.parametrize(
    ("fmt", "group_size", "expected_cls", "expected_pack_dtype"),
    [
        (FORMAT.GEMM, 128, "AwqTorchLinear", torch.int32),
        (FORMAT.GEMV, 64, "AwqGEMVLinear", torch.int32),
        (FORMAT.GEMV_FAST, 128, "AwqGEMVFastLinear", torch.int16),
    ],
)
def test_export_shell_materializes_packed_buffers_on_cpu(
    fmt, group_size, expected_cls, expected_pack_dtype
):
    """The exact selection the save path performs, on a CPU-only host."""

    qcfg = _awq_config(fmt, group_size)
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(256, 128, bias=False)

    kernel = make_quant(
        model,
        qcfg=qcfg,
        quant_result={"q_proj": {}},
        backend=BACKEND.AUTO,
        lm_head_name="lm_head",
        pack=True,
        device=_export_shell_device(qcfg),
    )

    assert kernel.__name__ == expected_cls
    layer = model.q_proj
    assert type(layer).__name__ == expected_cls
    buffers = dict(layer.named_buffers())
    assert {"qweight", "qzeros", "scales"} <= set(buffers)
    assert buffers["qweight"].dtype == expected_pack_dtype
    assert buffers["qweight"].dim() == 2
    assert layer.qweight.device.type == "cpu"


def _write_tiny_config(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "tiny-checkpoint"
    checkpoint.mkdir()
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize("fmt", [FORMAT.GEMM, FORMAT.GEMV, FORMAT.GEMV_FAST])
def test_save_shell_asks_for_the_layout_device(tmp_path, monkeypatch, fmt):
    """``get_model_with_quantize`` must route the shell through the helper."""

    checkpoint = _write_tiny_config(tmp_path)
    qcfg = _awq_config(fmt, 128)
    observed = {}

    def _from_config(config, **_kwargs):
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM(config)

    def _make_quant(model, **kwargs):
        observed.update(kwargs)
        raise _Stop

    monkeypatch.setattr(writer_module, "make_quant", _make_quant)
    owner = SimpleNamespace(
        trust_remote_code=False,
        loader=SimpleNamespace(from_config=_from_config),
        lm_head="lm_head",
        get_base_modules=lambda model: [],
        simple_layer_modules=lambda config, qcfg: [
            [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        ],
        extract_layers_node=lambda: ["model.layers"],
        should_quantize_module=lambda model, name, module, qcfg: True,
    )

    with pytest.raises(_Stop):
        BaseQModel.get_model_with_quantize(
            owner, qcfg=qcfg, model_id_or_path=str(checkpoint)
        )

    assert observed["quant_result"], "the shell must have collected its modules"
    assert observed["device"] == _export_shell_device(qcfg)
    assert observed["backend"] == BACKEND.AUTO
    assert observed["pack"] is True
