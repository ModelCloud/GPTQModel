# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled/failed ``quantize()`` must keep finished disk-offloaded layers (#4807).

With the default ``offload_to_disk=True`` every completed layer is packed,
written to the run's private offload directory, and replaced by a meta shell
behind an accelerate disk-offload hook. #4580 restored the in-flight layer and
the base modules, but the failure branch then deleted the run directory while
those shells still pointed into it. The next forward failed with
``FileNotFoundError`` and the completed layers were unrecoverable.

The failure branch must not delete a directory that a live module still reads
from. A cancelled run keeps the directory, which is already owned by the
model-lifetime finalizer installed by ``bind_private_offload_root``, so the
completed layers stay loadable and the next plain forward works.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast
from gptqmodel.utils.offload import is_meta_module


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


def _build_qmodel(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
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
    return LlamaQModel(
        model=model,
        quantized=False,
        tokenizer=tokenizer,
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device=DEVICE,
            # Default is True; pin it so this test cannot silently stop
            # covering the disk-offload failure branch.
            offload_to_disk=True,
            offload_to_disk_path=str(tmp_path / "offload"),
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


def _disk_offloaded_modules(model) -> list[str]:
    """Names of modules that still read their weights from a disk bundle."""

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
def test_cancelled_or_failed_quantize_keeps_completed_disk_offloaded_layers(tmp_path, error):
    qmodel = _build_qmodel(tmp_path)

    def cancel(**kwargs):
        # Cancel on layer 1's forward: layer 0 is already packed, finalized and
        # written to the run's offload directory by the time we get here.
        if kwargs["stage"] == "forward_start" and kwargs["layer_idx"] == 1:
            return StopMainLoop
        return None

    def fail(**kwargs):
        if kwargs["stage"] == "forward_start" and kwargs["layer_idx"] == 1:
            raise RuntimeError("injected mid-run failure")
        return None

    qmodel.subset_callback = cancel if error is StopMainLoop else fail

    with pytest.raises(error):
        qmodel.quantize(
            CalibrationConfig(
                dataset=CALIBRATION,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.GPTQ_TORCH,
        )

    model = qmodel.model
    offloaded = _disk_offloaded_modules(model)
    assert offloaded, "expected the completed layer to be disk-offloaded after the cancel"

    # Either the root survived, or the model no longer points at it. Deleting
    # the root while a shell still references it is the bug: on unmodified main
    # `_private_offload_run_root` is None and every bundle is gone.
    run_root = getattr(qmodel, "_private_offload_run_root", None)
    if run_root is None or not Path(run_root).is_dir():
        assert not offloaded, (
            "the per-run offload root was deleted while modules still read from it: "
            f"{offloaded[:8]}"
        )
    else:
        for name in offloaded:
            assert (Path(run_root) / name / "module.safetensors").is_file(), (
                f"offload bundle for completed layer `{name}` is missing"
            )

    # The finished layers must still be reconstructible: a plain forward reads
    # them back through their hooks instead of raising FileNotFoundError.
    _assert_plain_forward_still_works(model)
