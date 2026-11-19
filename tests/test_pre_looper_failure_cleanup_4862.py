# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A ``quantize()`` failure before the looper must not leak Device-SMI threads (#4862).

#4823 closed the processors' Device-SMI handles in the failure branch of the
loopers. A ``quantize()`` that fails before any looper exists -- a calibration
dataset that is empty, too short, or malformed, or a later processor in the
chain failing -- aborted ``LoopProcessor.__init__`` after it had already opened
one ``device_smi.Device`` per accelerator plus the CPU. Each handle owns a
5 Hz ``device-smi-metrics-*`` sampler thread, so every retry added threads for
the life of the process.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import gptqmodel.models.base as base_module
from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.adapter.adapter import Lora
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.quantization_run_plan import (
    ProcessorStep,
)
from gptqmodel.models.quantization_run_plan import (
    processor_class as real_processor_class,
)
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast


SMI_THREAD_PREFIX = "device-smi"
CALIBRATION = [{"input_ids": list(range(1, 17)), "attention_mask": [1] * 16}]


def _smi_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(SMI_THREAD_PREFIX)
    ]


def _new_smi_threads(baseline_idents: set[int | None]) -> list[threading.Thread]:
    return [thread for thread in _smi_threads() if thread.ident not in baseline_idents]


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` so a sampler thread that is still stopping is not a flake."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


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
            device="cpu",
            offload_to_disk=False,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


def _assert_no_new_smi_threads(baseline: set[int | None]) -> None:
    assert _wait_for(lambda: not _new_smi_threads(baseline)), [
        thread.name for thread in _new_smi_threads(baseline)
    ]


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        # #4985 moved the empty check into the shared calibration preparer.
        pytest.param([], "calibration dataset is empty", id="empty"),
        pytest.param(
            [{"input_ids": [1, 2], "attention_mask": [1, 1]}],
            "removed all samples",
            id="too-short",
        ),
        pytest.param([123], "unsupported calibration example type", id="bad-row"),
    ],
)
def test_calibration_rejected_before_the_looper_leaves_no_smi_threads(
    tmp_path, dataset, message
):
    """Calibration is prepared once before any processor or looper is built."""

    baseline = {thread.ident for thread in _smi_threads()}
    qmodel = _build_qmodel(tmp_path)

    with pytest.raises(ValueError, match=message):
        qmodel.quantize(
            CalibrationConfig(dataset=dataset, batch_size=1),
            backend=BACKEND.GPTQ_TORCH,
        )

    _assert_no_new_smi_threads(baseline)


def test_later_processor_failure_releases_the_processors_already_built(
    tmp_path, monkeypatch
):
    """GPTQ + EoRA: the EoRA processor fails, so the built GPTQ one is released."""

    baseline = {thread.ident for thread in _smi_threads()}
    qmodel = _build_qmodel(tmp_path)

    released: list[str] = []
    original_release = LoopProcessor.release_failed_run

    def recording_release(self) -> None:
        released.append(type(self).__name__)
        original_release(self)

    monkeypatch.setattr(LoopProcessor, "release_failed_run", recording_release)

    # #4985 validates the calibration and adapter rows before any processor is
    # built, so a malformed `adapter_dataset` no longer reaches the prologue
    # release. Fail the second processor's construction instead: the GPTQ
    # processor built first must still be released.
    def failing_processor_class(step):
        if step is ProcessorStep.EORA:
            raise ValueError("EoRA processor construction failed")
        return real_processor_class(step)

    monkeypatch.setattr(base_module, "processor_class", failing_processor_class)

    with pytest.raises(ValueError, match="EoRA processor construction failed"):
        qmodel.quantize(
            CalibrationConfig(
                dataset=CALIBRATION,
                adapter_dataset=CALIBRATION,
                batch_size=1,
            ),
            backend=BACKEND.GPTQ_TORCH,
            adapter=Lora(rank=1),
        )

    # The prologue must release every processor it built, not only the ones a
    # looper would have seen.
    assert released, "the built processors were never released"
    assert "GPTQProcessor" in released, released
    assert "EoraProcessor" not in released
    _assert_no_new_smi_threads(baseline)


def test_release_failed_run_on_a_never_opened_processor_is_a_noop():
    """A processor that never queried metrics has no handles and no threads.

    The prologue release path hits exactly this case, so it must not raise and
    must leave the processor closed against a later lazy open.
    """

    from gptqmodel.looper.loop_processor import (
        _DEVICE_SMI_UNINITIALIZED,
        LoopProcessor,
    )

    processor = object.__new__(LoopProcessor)
    processor._device_smi_lock = threading.RLock()
    processor._device_smi_handles = _DEVICE_SMI_UNINITIALIZED
    processor._cpu_device_smi = None

    processor.release_failed_run()
    processor.release_failed_run()

    assert processor._device_smi_handles == {}
    assert processor._cpu_device_smi is None
