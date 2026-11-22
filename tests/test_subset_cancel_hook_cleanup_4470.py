# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled or failed subset forward must release its capture hooks (#4470).

``StageSubset`` installs a forward hook on every module of the subset and flips
``forward_hook_last`` on the final one. The cleanup ran only on the success
path, so a ``StopMainLoop`` from the ``forward_start`` callback (or any forward
exception) left the hooks installed: the closures retained the processor state
and the next ordinary forward raised ``StopForward``.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast


CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


def _build_qmodel(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
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
            device="cpu",
            offload_to_disk=True,
            offload_to_disk_path=str(tmp_path / "offload"),
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


def _leaked_capture_hooks(model) -> list[tuple[str, str]]:
    leaked: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        if getattr(module, "forward_hook", None) is not None:
            leaked.append((name, "forward_hook"))
        if getattr(module, "forward_hook_last", False):
            leaked.append((name, "forward_hook_last"))
        for attribute in ("_forward_hooks", "_forward_pre_hooks"):
            for _ in getattr(module, attribute, {}).values():
                leaked.append((name, attribute))
    return leaked


def _assert_plain_forward_still_works(model) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert output.logits.shape == (1, 4, 32)


@pytest.mark.parametrize("error", [StopMainLoop, RuntimeError], ids=["stop", "failure"])
def test_cancelled_subset_releases_capture_hooks(tmp_path, error):
    qmodel = _build_qmodel(tmp_path)

    def cancel(**kwargs):
        if kwargs["stage"] == "forward_start":
            return StopMainLoop
        return None

    def fail(**kwargs):
        if kwargs["stage"] == "forward_start":
            raise RuntimeError("callback failure")
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

    assert _leaked_capture_hooks(qmodel.model) == []
    _assert_plain_forward_still_works(qmodel.model)


def test_forward_exception_releases_capture_hooks(tmp_path, monkeypatch):
    qmodel = _build_qmodel(tmp_path)
    state = {"armed": False}

    original_forward = HookedLinear.forward

    def exploding_forward(self, x):
        if state["armed"]:
            raise RuntimeError("forward failure")
        return original_forward(self, x)

    def arm(**kwargs):
        if kwargs["stage"] == "forward_start":
            state["armed"] = True
        return None

    qmodel.subset_callback = arm
    monkeypatch.setattr(HookedLinear, "forward", exploding_forward)

    with pytest.raises(RuntimeError, match="forward failure"):
        qmodel.quantize(
            CalibrationConfig(
                dataset=CALIBRATION,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.GPTQ_TORCH,
        )

    state["armed"] = False
    assert _leaked_capture_hooks(qmodel.model) == []
    _assert_plain_forward_still_works(qmodel.model)
