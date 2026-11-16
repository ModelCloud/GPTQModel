# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4321: every quantization method exports its log.

The looper used to copy a processor's per-module ``log`` into
``model.quant_log`` only for a hardcoded list of processor names. FOEM, QQQ,
VPTQ, ParoQuant, FP8-GSQ and RTN were therefore quantized but produced no
``quant_log.csv``. These tests drive the same helper that populates
``model.quant_log`` and check that the writer emits the CSV when it is set.
"""

import builtins
import csv
import io
from types import SimpleNamespace

import pytest

from gptqmodel.looper.module_looper import publish_quant_log
from gptqmodel.models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_TIME,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_LOSS_KIND,
    QUANT_LOG_NSAMPLES,
)
from tests.test_writer_quant_log import writer  # noqa: F401  (pytest fixture)


class _StubProcessor:
    """Minimal processor exposing the two attributes the looper reads."""

    def __init__(self, name: str, log=None):
        self._name = name
        self.log = [] if log is None else log

    def name(self) -> str:
        return self._name


@pytest.mark.parametrize(
    "name",
    [
        "foem",
        "qqq",
        "vptq",
        "paroquant",
        "fp8_gsq",
        "weight_only_rtn",
        # Existing methods must keep working unchanged.
        "gptq",
        "gptaq",
        "awq",
        "exl3",
        "macaq",
    ],
)
def test_processor_log_is_published_for_any_name(name: str) -> None:
    rows = [{PROCESS_LOG_MODULE: "model.layers.0.mlp.up_proj"}]
    model = SimpleNamespace(quant_log=[])

    publish_quant_log(model, [_StubProcessor("native"), _StubProcessor(name, rows)])

    assert model.quant_log is rows


def test_empty_processor_log_is_not_published() -> None:
    model = SimpleNamespace(quant_log=[])

    publish_quant_log(model, [_StubProcessor("native"), _StubProcessor("foem", [])])

    assert model.quant_log == []


def test_primary_log_wins_over_auxiliary_processor_log() -> None:
    primary = [{PROCESS_LOG_MODULE: "model.layers.0.mlp.up_proj"}]
    adapter = [{PROCESS_LOG_MODULE: "model.layers.0.self_attn.q_proj"}]
    model = SimpleNamespace(quant_log=[])

    publish_quant_log(
        model,
        [
            _StubProcessor("native", []),
            _StubProcessor("foem", primary),
            _StubProcessor("eora", adapter),
        ],
    )

    assert model.quant_log is primary


def test_empty_log_does_not_clobber_restored_quant_log() -> None:
    restored = [{PROCESS_LOG_MODULE: "model.layers.0.mlp.down_proj"}]
    model = SimpleNamespace(quant_log=restored)

    publish_quant_log(model, [_StubProcessor("foem", [])])

    assert model.quant_log is restored


class _RecordingBuffer(io.StringIO):
    """StringIO that survives the writer's ``with`` block closing it."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def test_writer_writes_quant_log_csv_when_log_is_non_empty(writer, tmp_path, monkeypatch) -> None:  # noqa: F811
    writer.quant_log = [
        {
            PROCESS_LOG_LAYER: 0,
            PROCESS_LOG_MODULE: "proj",
            QUANT_LOG_LOSS: 0.25,
            QUANT_LOG_DAMP: 0.05,
            PROCESS_LOG_TIME: 1.5,
        }
    ]
    opened = {}
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("quant_log.csv"):
            buffer = _RecordingBuffer()
            opened["path"] = str(path)
            opened["buffer"] = buffer
            return buffer
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    writer.save_quantized(str(tmp_path / "saved"))

    assert opened["path"].endswith("quant_log.csv")
    rows = list(csv.DictReader(io.StringIO(opened["buffer"].getvalue())))
    assert rows == [
        {
            PROCESS_LOG_LAYER: "0",
            PROCESS_LOG_MODULE: "proj",
            QUANT_LOG_LOSS: "0.25",
            QUANT_LOG_LOSS_KIND: "",
            QUANT_LOG_NSAMPLES: "",
            QUANT_LOG_DAMP: "0.05",
            PROCESS_LOG_TIME: "1.5",
        }
    ]
