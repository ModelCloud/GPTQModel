# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pack-time shape contracts for AWQ and QQQ (issue #4567).

The model-wide packing kernel is chosen before any module shape is known, but
``AwqGEMMLinear`` only packs ``out_features % 64 == 0`` and QQQ's kernels tile
both axes. A width the *format* supports (FalconMamba/Mamba ``x_proj``
8192->288) therefore failed inside ``create_quant_module`` after the whole
calibration had run. AWQ now falls back to the portable ``AwqTorchLinear`` of the
same checkpoint format; QQQ has no portable alternative and fails on the first
offending module instead of the last.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QQQConfig, QuantizeConfig


# The Mamba `x_proj` width from the issue: 8192 -> 288 (dt_rank + 2*d_state).
PARTIAL_OUT_FEATURES = 288
SUPPORTED_OUT_FEATURES = 512


def _awq_processor(model_kernel, *, bits: int = 4, group_size: int = 128) -> AWQProcessor:
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
    )
    gptq_model = SimpleNamespace(
        qlinear_kernel=model_kernel,
        lm_head="lm_head",
        quantize_config=qcfg,
    )
    return AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=gptq_model,
        model=None,
        calculate_w_wq_diff=False,
    )


def test_awq_falls_back_to_portable_kernel_for_partial_output_tile():
    processor = _awq_processor(AwqGEMMLinear)

    resolved = processor._resolve_qlinear_kernel(
        "model.layers.0.mixer.x_proj",
        in_features=8192,
        out_features=PARTIAL_OUT_FEATURES,
    )

    assert resolved is AwqTorchLinear


def test_awq_keeps_native_kernel_for_a_supported_width():
    processor = _awq_processor(AwqGEMMLinear)

    resolved = processor._resolve_qlinear_kernel(
        "model.layers.0.mlp.down_proj",
        in_features=8192,
        out_features=SUPPORTED_OUT_FEATURES,
    )

    assert resolved is AwqGEMMLinear


def test_awq_kernel_choice_is_unchanged_without_a_shape():
    """Callers that only know the module name keep the old behaviour."""

    processor = _awq_processor(AwqGEMMLinear)

    assert processor._resolve_qlinear_kernel("model.layers.0.mlp.down_proj") is AwqGEMMLinear


def test_awq_keeps_the_requested_kernel_when_no_kernel_fits():
    """A width the portable kernel also rejects must surface its own error."""

    processor = _awq_processor(AwqGEMMLinear)

    resolved = processor._resolve_qlinear_kernel(
        "model.layers.0.proj",
        in_features=8192,
        out_features=100,  # not divisible by AwqTorchLinear's 8 either
    )

    assert resolved is AwqGEMMLinear


def _build_tiny_qwen3(directory: Path) -> None:
    from tests.test_tiny_moe_quant_smoke import _build_local_tokenizer

    config = Qwen3Config(
        num_hidden_layers=1,
        hidden_size=96,
        intermediate_size=192,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=24,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3ForCausalLM(config).save_pretrained(directory)
    _build_local_tokenizer(directory)


def test_awq_pack_step_resolves_the_kernel_from_the_module_geometry(tmp_path, monkeypatch):
    """The pack step must pass real in/out features, not just the module name.

    Without the geometry the fallback above can never trigger, which is exactly
    how the shape error escaped to ``create_quant_module``.
    """

    from tests.test_tiny_moe_quant_smoke import _build_calibration_dataset

    _build_tiny_qwen3(tmp_path / "native")

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
        offload_to_disk=False,
    )

    seen: list[tuple[str, int, int]] = []
    real = AWQProcessor._resolve_qlinear_kernel

    def spy(self, module_name=None, *, resolved_cfg=None, in_features=None, out_features=None):
        if in_features is not None and out_features is not None:
            seen.append((module_name, in_features, out_features))
        return real(
            self,
            module_name,
            resolved_cfg=resolved_cfg,
            in_features=in_features,
            out_features=out_features,
        )

    monkeypatch.setattr(AWQProcessor, "_resolve_qlinear_kernel", spy)

    model = GPTQModel.load(
        str(tmp_path / "native"), quantize_config=qcfg, backend=BACKEND.TORCH
    )
    model.quantize(
        _build_calibration_dataset(model.tokenizer),
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )

    # `q_proj` is 96 -> 96: the width the CUDA GEMM kernel rejects.
    assert (
        "model.layers.0.self_attn.q_proj",
        96,
        96,
    ) in seen
    packed = {
        name: module
        for name, module in model.model.named_modules()
        if isinstance(module, (AwqTorchLinear, AwqGEMMLinear))
    }
    assert packed, "expected AWQ to pack at least one module"


def _qqq_processor() -> QQQProcessor:
    qcfg = QQQConfig(bits=4, group_size=32, desc_act=False, device="cpu")
    return QQQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        calculate_w_wq_diff=False,
        backend=BACKEND.QQQ_TORCH,
    )


def _named(linear: nn.Linear, full_name: str) -> NamedModule:
    return NamedModule(
        linear,
        name=full_name.rsplit(".", 1)[-1],
        full_name=full_name,
        layer_index=0,
    )


@pytest.mark.parametrize(
    "in_features,out_features",
    [(8192, PARTIAL_OUT_FEATURES), (256, 288), (96, 96)],
    ids=["x_proj_8192x288", "256x288", "96x96"],
)
def test_qqq_rejects_an_unpackable_module_before_calibration(in_features, out_features):
    processor = _qqq_processor()
    module = _named(
        nn.Linear(in_features, out_features, bias=False),
        "model.layers.0.mixer.x_proj",
    )

    with pytest.raises(ValueError, match=r"QQQ cannot pack module `model\.layers\.0\.mixer\.x_proj`"):
        processor.preprocess(module)

    # The rejection happens before any calibration task is created.
    assert processor.tasks == {}


@pytest.mark.parametrize(
    "in_features,out_features",
    [(256, 128), (128, 256), (64, 128)],
    ids=["256x128", "128x256", "64x128"],
)
def test_qqq_accepts_a_supported_tile(in_features, out_features):
    processor = _qqq_processor()
    module = _named(
        nn.Linear(in_features, out_features, bias=False),
        "model.layers.0.mlp.down_proj",
    )

    processor.preprocess(module)

    assert "down_proj" in processor.tasks


def test_qqq_error_names_the_kernel_and_the_alternatives():
    processor = _qqq_processor()
    module = _named(
        nn.Linear(8192, PARTIAL_OUT_FEATURES, bias=False),
        "model.layers.0.mixer.x_proj",
    )

    with pytest.raises(ValueError) as excinfo:
        processor.preprocess(module)

    message = str(excinfo.value)
    assert QQQTorchLinear.__name__ in message
    assert "8192 -> 288" in message
    assert "`-:` dynamic rule" in message
