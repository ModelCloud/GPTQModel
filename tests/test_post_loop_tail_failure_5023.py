# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A failure in the post-loop tail must still leave a runnable model (#5023).

Both loopers guard ``_loop_impl`` with a best-effort failure cleanup, but the
tail that turns the lazy offloaded shell back into an in-session model used to
run *after* that guard. A Ctrl-C, an OOM or an I/O error landing there (for RTN
about 15% of the run, and ``restore_disk_offloaded_modules`` reads every
quantized layer back from disk) skipped the cleanup entirely, so
``embed_tokens``/``norm``/``rotary_emb`` stayed on meta and every later
in-session forward failed with a meta-tensor error.

Each case injects one failure at the first call of a tail helper, which is what
an interrupt delivered at that point does; later calls (the recovery) run the
real helper.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import (
    AutoDtypeCast,
    CalibrationConfig,
    QuantizeConfig,
    RTNConfig,
)


DEVICE = "cpu"
NUM_LAYERS = 4
CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


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


def _load_qmodel(source: Path, *, weight_only: bool = False):
    config_cls = RTNConfig if weight_only else QuantizeConfig
    return GPTQModel.load(
        str(source),
        quantize_config=config_cls(
            bits=4,
            group_size=32,
            desc_act=False,
            device=DEVICE,
            # The default; pinned so the test cannot silently stop covering the
            # disk-offload shell.
            offload_to_disk=True,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        dtype=torch.float16,
    )


def _assert_forward_is_runnable(qmodel) -> None:
    qmodel.model.eval()
    with torch.inference_mode():
        output = qmodel.model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert output.logits.shape == (1, 4, 32)
    assert torch.isfinite(output.logits).all()


def _fail_first_call(monkeypatch, target, attribute, exception):
    """Raise ``exception`` at the first call, then run the real helper."""

    original = getattr(target, attribute)
    state = {"calls": 0}

    def wrapper(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise exception("injected post-loop failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attribute, wrapper)
    return state


@pytest.mark.parametrize(
    "attribute",
    [
        "restore_disk_offloaded_modules",
        "replace_inference_parameters",
        "materialize_remaining_meta_modules",
    ],
)
def test_weight_only_tail_failure_leaves_a_runnable_model(tmp_path, monkeypatch, attribute):
    import gptqmodel.looper.weight_only_looper as weight_only_looper

    state = _fail_first_call(
        monkeypatch, weight_only_looper, attribute, KeyboardInterrupt
    )

    qmodel = _load_qmodel(_build_source(tmp_path), weight_only=True)
    with pytest.raises(KeyboardInterrupt):
        qmodel.quantize(None)

    assert state["calls"] >= 1
    _assert_forward_is_runnable(qmodel)


@pytest.mark.parametrize(
    ("attribute", "exception"),
    [
        ("restore_disk_offloaded_modules", KeyboardInterrupt),
        ("materialize_remaining_meta_modules", RuntimeError),
        ("restore_offloaded_modules", RuntimeError),
    ],
)
def test_gptq_tail_failure_leaves_a_runnable_model(tmp_path, monkeypatch, attribute, exception):
    import gptqmodel.looper.module_looper as module_looper

    state = _fail_first_call(monkeypatch, module_looper, attribute, exception)

    qmodel = _load_qmodel(_build_source(tmp_path))
    with pytest.raises(exception):
        qmodel.quantize(
            CalibrationConfig(dataset=CALIBRATION, batch_size=1, min_length=1),
            backend=BACKEND.GPTQ_TORCH,
        )

    assert state["calls"] >= 1
    _assert_forward_is_runnable(qmodel)
