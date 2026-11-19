# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled/failed AWQ or QQQ ``quantize()`` must leave runnable kernels (#4947).

#4893 materialized the layers a cancelled run never reached, and #4829/#4807
kept the offload root of the layers it did finish. The failure branch never
rehomed the finished layers' kernels, so a run packed with an accelerator-only
kernel (``AwqGEMMLinear``, ``QQQLinear``) left CUDA-only modules on a CPU model
and the next forward raised ``Could not run ... with arguments from the 'CPU'
backend``. The success path rehomes them in ``rehome_quantized_kernel``.

The failure branch now rehomes best effort on the device the rest of the model
occupies. Finished layers of a disk-offloaded run are meta shells, so the
accelerate hook's captured bound ``forward`` is rebound as part of the class
swap; otherwise the hook would keep calling the old kernel.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, CheckpointConfig, GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.base import _kernel_supports_device
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import AWQConfig, QQQConfig
from gptqmodel.quantization.config import AutoDtypeCast


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="AWQ GEMM and QQQ pack with CUDA-only kernels; this contract needs a CUDA host.",
)

NUM_LAYERS = 4
CANCEL_AT = 2
PROJECTIONS_PER_LAYER = 7  # q, k, v, o, gate, up, down
CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


def _build_source(tmp_path: Path, *, hidden_size: int, intermediate_size: int) -> Path:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).eval().save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    return source


def _quantize_config(method: str):
    if method == "awq":
        # AWQ GEMM needs out_features % 64 == 0, so keep the tiny model's
        # projections wide enough to select the CUDA-only kernel.
        return AWQConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cuda:0",
            offload_to_disk=True,
            auto_dtype_cast=AutoDtypeCast.OFF,
        )
    return QQQConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        device="cuda:0",
        offload_to_disk=True,
        auto_dtype_cast=AutoDtypeCast.OFF,
    )


def _load_qmodel(source: Path, method: str):
    return GPTQModel.load(
        str(source), quantize_config=_quantize_config(method), dtype=torch.float16
    )


def _cancel_hook(error, layer_idx: int = CANCEL_AT):
    def callback(**kwargs):
        if kwargs.get("stage") == "forward_start" and kwargs.get("layer_idx") == layer_idx:
            if error is StopMainLoop:
                return StopMainLoop
            raise error("injected mid-run failure")
        return None

    return callback


def _run_calibration(qmodel, *, checkpoint=None):
    kwargs = {"checkpoint": checkpoint} if checkpoint is not None else {}
    return qmodel.quantize(
        CalibrationConfig(dataset=CALIBRATION, batch_size=1, min_length=1),
        backend=BACKEND.AUTO,
        **kwargs,
    )


def _assert_plain_forward_still_works(model) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert output.logits.shape == (1, 4, 32)
    assert torch.isfinite(output.logits).all()


def _finished_quant_modules(qmodel):
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    return [
        module
        for module in qmodel.model.modules()
        if isinstance(module, BaseQuantLinear)
    ]


def _force_cuda_qqq_kernel(monkeypatch) -> None:
    """Select ``QQQLinear`` even when the native extension cannot build here.

    The regression only needs the CUDA-only class to be selected and packed;
    after the fix the layers are rehomed to the portable ``QQQTorchLinear``,
    so the native kernel never has to execute.
    """

    if QQQLinear.validate_once()[0]:
        return
    monkeypatch.setattr(QQQLinear, "validate_once", classmethod(lambda cls: (True, None)))


METHODS = [
    pytest.param("awq", 64, 128, id="awq"),
    pytest.param("qqq", 128, 256, id="qqq"),
]


@pytest.mark.parametrize(("method", "hidden_size", "intermediate_size"), METHODS)
@pytest.mark.parametrize("error", [StopMainLoop, RuntimeError], ids=["cancel", "failure"])
def test_failed_run_rehomes_finished_layers(
    tmp_path, monkeypatch, method, hidden_size, intermediate_size, error
):
    if method == "qqq":
        _force_cuda_qqq_kernel(monkeypatch)

    source = _build_source(
        tmp_path, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    qmodel = _load_qmodel(source, method)
    assert qmodel.quantize_config.offload_to_disk is True

    qmodel.subset_callback = _cancel_hook(error)
    with pytest.raises(error):
        _run_calibration(qmodel)

    finished = _finished_quant_modules(qmodel)
    assert len(finished) == CANCEL_AT * PROJECTIONS_PER_LAYER, [type(m).__name__ for m in finished]

    model_device = qmodel.model.get_input_embeddings().weight.device
    portable = AwqTorchLinear if method == "awq" else QQQTorchLinear
    for module in finished:
        # The packed layers must be interpreted by a kernel the model's device
        # can actually execute, not the CUDA-only calibration kernel.
        assert type(module) is portable, type(module).__name__
        assert _kernel_supports_device(type(module), model_device)
    assert qmodel.qlinear_kernel is portable

    _assert_plain_forward_still_works(qmodel.model)


@pytest.mark.parametrize(("method", "hidden_size", "intermediate_size"), METHODS)
def test_cancel_at_last_layer_leaves_a_runnable_model(
    tmp_path, monkeypatch, method, hidden_size, intermediate_size
):
    """No unreached layer to materialize; the finished layers still need rehoming."""

    if method == "qqq":
        _force_cuda_qqq_kernel(monkeypatch)

    source = _build_source(
        tmp_path, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    qmodel = _load_qmodel(source, method)
    qmodel.subset_callback = _cancel_hook(StopMainLoop, layer_idx=NUM_LAYERS - 1)

    with pytest.raises(StopMainLoop):
        _run_calibration(qmodel)

    portable = AwqTorchLinear if method == "awq" else QQQTorchLinear
    finished = _finished_quant_modules(qmodel)
    assert finished
    assert all(type(module) is portable for module in finished)
    _assert_plain_forward_still_works(qmodel.model)


@pytest.mark.parametrize(("method", "hidden_size", "intermediate_size"), METHODS)
def test_cancelled_run_can_still_resume(
    tmp_path, monkeypatch, method, hidden_size, intermediate_size
):
    if method == "qqq":
        _force_cuda_qqq_kernel(monkeypatch)

    source = _build_source(
        tmp_path, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    checkpoint = CheckpointConfig(tmp_path / "checkpoints")

    qmodel = _load_qmodel(source, method)
    qmodel.subset_callback = _cancel_hook(StopMainLoop)
    with pytest.raises(StopMainLoop):
        _run_calibration(qmodel, checkpoint=checkpoint)

    resumed = _load_qmodel(source, method)
    _run_calibration(resumed, checkpoint=CheckpointConfig(tmp_path / "checkpoints"))
    _assert_plain_forward_still_works(resumed.model)
