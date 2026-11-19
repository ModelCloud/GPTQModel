# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5026: no caller pins the task Hessian across ``quantize()``.

``GPTQProcessor.process`` bound the task's Hessian to a local (``g_h``) for a
device assert and then called ``g.quantize()`` with that local still alive.
GPTAQ/FOEM hand their accumulator to the quantizer (``H = self.H; del self.H``)
and rebind ``H`` to the activation-order permuted copy, so the caller's local
was the only remaining reference to the original unpermuted fp32 ``[C, C]``
Hessian and kept it allocated for the whole quantize -- including the peak,
where ``H``, ``dXXT``, the inverse and the native-delta product are all live.

The nodes below take a ``weakref`` to ``task.H`` before ``process()`` and, on
entry to ``hessian_inverse`` (which every GPTQ/GPTAQ/FOEM quantize calls), scan
the frames *above* the quantizer for a local that still holds that tensor. A
caller local is exactly the defect; the quantizer's own handle is fine (it
releases it) and lives in its own frames, which the scan skips.

A failed ``quantize()`` must release the accumulator too; otherwise the task
pins it until the processor is torn down.
"""

from __future__ import annotations

import sys
import weakref

import pytest
import torch
import transformers

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, GPTQConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ
from tests.tiny_remote_models import build_wordlevel_tokenizer


ROW_COUNT = 8
ROW_LENGTH = 48


def _tiny_model(model_dir):
    """Build a 2-layer Qwen2 checkpoint with a local word-level tokenizer."""

    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)
    return model_dir


def _rows(length=ROW_LENGTH):
    generator = torch.Generator().manual_seed(5026)
    return [
        {"input_ids": torch.randint(4, 128, (length,), generator=generator).tolist()}
        for _ in range(ROW_COUNT)
    ]


def _quantize_config(kind):
    if kind == "gptaq":
        return GPTQConfig(bits=4, group_size=32, gptaq=GPTAQConfig())
    if kind == "foem":
        return GPTQConfig(bits=4, group_size=32, foem=FOEMConfig(alpha=0.25))
    return GPTQConfig(bits=4, group_size=32)


class _HessianProbe:
    """Tracks the pre-quantize ``task.H`` across the ``process()`` handoff."""

    def __init__(self) -> None:
        # module name -> weakref to the Hessian the task held before process()
        self.pending: dict[str, weakref.ReferenceType] = {}
        # module name -> caller locals that still held it during quantize()
        self.callers: dict[str, list[str]] = {}
        # modules whose Hessian handoff was actually inspected
        self.checked: list[str] = []


def _install_probe(monkeypatch) -> _HessianProbe:
    probe = _HessianProbe()
    original_process = GPTQProcessor.process
    original_inverse = GPTQ.hessian_inverse

    def spy_process(self, module, *args, **kwargs):
        task = self.tasks.get(module.name)
        hessian = getattr(task, "H", None)
        if hessian is not None:
            probe.pending[module.name] = weakref.ref(hessian)
        # Drop every strong reference this frame holds: the probe must not
        # itself keep the tensor alive.
        del hessian, task
        return original_process(self, module, *args, **kwargs)

    def spy_inverse(self, hessian, *args, **kwargs):
        ref = probe.pending.get(self.name)
        if ref is not None and self.name not in probe.checked:
            probe.checked.append(self.name)
            original = ref()
            if original is not None:
                # Skip the quantizer's own frames: locate the `quantize` frame
                # that owns this task, then look only above it.
                boundary = None
                frame = sys._getframe().f_back
                while frame is not None:
                    if (
                        frame.f_code.co_name == "quantize"
                        and frame.f_locals.get("self") is self
                    ):
                        boundary = frame
                        break
                    frame = frame.f_back
                frame = boundary.f_back if boundary is not None else sys._getframe().f_back
                holders = []
                while frame is not None:
                    for name, value in frame.f_locals.items():
                        if value is original:
                            holders.append(f"{frame.f_code.co_name}():{name}")
                    frame = frame.f_back
                probe.callers[self.name] = holders
            del original
        return original_inverse(self, hessian, *args, **kwargs)

    monkeypatch.setattr(GPTQProcessor, "process", spy_process)
    monkeypatch.setattr(GPTQ, "hessian_inverse", spy_inverse)
    return probe


@pytest.mark.parametrize("kind", ["gptaq", "foem", "gptq"])
def test_no_caller_pins_the_task_hessian_across_quantize(tmp_path, monkeypatch, kind):
    model_dir = _tiny_model(tmp_path / "tiny")
    probe = _install_probe(monkeypatch)

    model = GPTQModel.load(
        str(model_dir), quantize_config=_quantize_config(kind), dtype=torch.float16
    )
    model.quantize(CalibrationConfig(dataset=_rows(), batch_size=1))

    assert probe.checked, "no module reached hessian_inverse with a task Hessian"
    pinned = {name: holders for name, holders in probe.callers.items() if holders}
    assert not pinned, (
        f"{kind}: a caller still held the task Hessian while quantize() ran "
        f"{pinned}; the original [C, C] accumulator stays allocated for the "
        "whole quantize and its peak"
    )


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_failed_quantize_releases_the_hessian(tmp_path, monkeypatch, kind):
    model_dir = _tiny_model(tmp_path / "tiny")
    _install_probe(monkeypatch)

    # GPTAQ/FOEM override ``quantize``, so patch the class the task uses.
    quantizer_cls = GPTAQ if kind == "gptaq" else FOEM
    original_quantize = quantizer_cls.quantize
    state = {"task": None, "ref": None, "fired": False}

    def failing_quantize(self, *args, **kwargs):
        if not state["fired"] and getattr(self, "H", None) is not None:
            state["fired"] = True
            state["task"] = self
            state["ref"] = weakref.ref(self.H)
            raise RuntimeError("injected Hessian failure")
        return original_quantize(self, *args, **kwargs)

    monkeypatch.setattr(quantizer_cls, "quantize", failing_quantize)

    model = GPTQModel.load(
        str(model_dir), quantize_config=_quantize_config(kind), dtype=torch.float16
    )
    with pytest.raises(RuntimeError, match="injected Hessian failure"):
        model.quantize(CalibrationConfig(dataset=_rows(), batch_size=1))

    assert state["fired"], "the injected failure never ran"
    assert state["task"].H is None, (
        "a failed quantize left the task Hessian attached; it stays allocated "
        "until the processor is torn down"
    )
    assert state["task"]._hessian_state_released, (
        "a failed quantize must release the Hessian accumulator state"
    )
    assert state["ref"]() is None, "the original Hessian storage survived the failed run"
