# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled default-offload ``quantize()`` must materialize unreached layers (#4860).

With the default ``offload_to_disk=True`` the model is loaded as a lazy meta
shell. ``#4829`` kept the per-run offload root for the layers that finished
before a cancel, but the failure branch never filled the layers the loop never
reached: they stayed on meta with no offload hook, so the next forward failed on
the first unreached layer with ``Tensor on device cpu is not on the expected
device meta!``.

The failure branch now materializes those layers best effort (the success path
already did) without touching the completed layers, which stay readable through
the kept offload root.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import (
    BACKEND,
    CalibrationConfig,
    CheckpointConfig,
    GPTQModel,
    QuantizeConfig,
)
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.quantization.config import AutoDtypeCast
from gptqmodel.utils.offload import _owns_meta_tensor, is_meta_module


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]
NUM_LAYERS = 4
CANCEL_AT = 2


def _build_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    return source


def _load_qmodel(source: Path):
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device=DEVICE,
        # Default is True; pin it so the test cannot silently stop covering the
        # default disk-offload path.
        offload_to_disk=True,
        auto_dtype_cast=AutoDtypeCast.OFF,
    )
    return GPTQModel.load(str(source), quantize_config=quantize_config, dtype=torch.float16)


def _cancel_hook(error):
    def callback(**kwargs):
        if kwargs.get("stage") == "forward_start" and kwargs.get("layer_idx") == CANCEL_AT:
            if error is StopMainLoop:
                return StopMainLoop
            raise error("injected mid-run failure")
        return None

    return callback


def _run_calibration(qmodel, *, checkpoint=None):
    kwargs = {"checkpoint": checkpoint} if checkpoint is not None else {}
    return qmodel.quantize(
        CalibrationConfig(dataset=CALIBRATION, batch_size=1, min_length=1),
        backend=BACKEND.GPTQ_TORCH,
        **kwargs,
    )


def _disk_offloaded_modules(model) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if is_meta_module(module) and getattr(module, "_hf_hook", None) is not None
    ]


def _assert_plain_forward_still_works(model) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert output.logits.shape == (1, 4, 32)


@pytest.mark.parametrize("error", [StopMainLoop, RuntimeError], ids=["cancel", "failure"])
def test_cancelled_default_offload_materializes_unreached_layers(tmp_path, error):
    source = _build_source(tmp_path)
    qmodel = _load_qmodel(source)
    assert qmodel.quantize_config.offload_to_disk is True

    qmodel.subset_callback = _cancel_hook(error)
    with pytest.raises(error):
        _run_calibration(qmodel)

    model = qmodel.model
    layers = model.model.layers
    assert len(layers) == NUM_LAYERS

    # Layers the loop never reached must no longer own meta tensors.
    for index in range(CANCEL_AT + 1, NUM_LAYERS):
        unresolved = [
            name
            for name, module in layers[index].named_modules()
            if _owns_meta_tensor(module)
        ]
        assert not unresolved, (
            f"layer {index} was never reached by the cancelled run but still "
            f"owns meta tensors: {unresolved}"
        )

    # The completed layers keep the #4829 contract: disk-offloaded shells whose
    # bundles still exist in the kept per-run root.
    offloaded = _disk_offloaded_modules(model)
    assert offloaded, "expected the completed layers to stay disk-offloaded"
    assert all(
        int(name.split(".")[2]) <= CANCEL_AT for name in offloaded
    ), f"unexpected offload bundle for an unreached layer: {offloaded}"
    run_root = getattr(qmodel, "_private_offload_run_root", None)
    assert run_root is not None and Path(run_root).is_dir()
    for name in offloaded:
        assert (Path(run_root) / name / "module.safetensors").is_file(), (
            f"offload bundle for completed module `{name}` is missing"
        )

    _assert_plain_forward_still_works(model)


def test_cancelled_default_offload_resume_still_works(tmp_path):
    source = _build_source(tmp_path)
    checkpoint = CheckpointConfig(tmp_path / "checkpoints")

    qmodel = _load_qmodel(source)
    qmodel.subset_callback = _cancel_hook(StopMainLoop)
    with pytest.raises(StopMainLoop):
        _run_calibration(qmodel, checkpoint=checkpoint)

    # A fresh model resumes the checkpointed layers of the cancelled run.
    resumed = _load_qmodel(source)
    _run_calibration(resumed, checkpoint=CheckpointConfig(tmp_path / "checkpoints"))
    _assert_plain_forward_still_works(resumed.model)
