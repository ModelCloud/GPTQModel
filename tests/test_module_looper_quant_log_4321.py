# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for quant-log publication (#4321 and #4367).

#4321: the looper used to copy a processor's per-module ``log`` into
``model.quant_log`` only for a hardcoded list of processor names. FOEM, QQQ,
VPTQ, ParoQuant, FP8-GSQ and RTN were therefore quantized but produced no
``quant_log.csv``.

#4367: the GPTQ+EoRA chain then concatenated the logs of *every* processor, so
each module appeared twice and the EoRA copy had empty loss/samples/damp. The
model-level log now comes from the first log-owning processor, and EoRA rows
never duplicate or replace it.
"""

import builtins
import csv
import io
from types import SimpleNamespace

import pytest

from gptqmodel.looper.loop_processor import record_quant_log
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
    """Minimal processor exposing the attributes the looper reads."""

    def __init__(self, name: str, log=None, *, owns_quant_log: bool = True):
        self._name = name
        self.log = [] if log is None else log
        self.owns_quant_log = owns_quant_log

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

    record_quant_log(model, [_StubProcessor("native"), _StubProcessor(name, rows)])

    assert model.quant_log == rows


def test_empty_processor_log_is_not_published() -> None:
    model = SimpleNamespace(quant_log=[])

    record_quant_log(model, [_StubProcessor("native"), _StubProcessor("foem", [])])

    assert model.quant_log == []


def test_primary_log_wins_over_auxiliary_processor_log() -> None:
    primary = [{PROCESS_LOG_MODULE: "model.layers.0.mlp.up_proj"}]
    adapter = [{PROCESS_LOG_MODULE: "model.layers.0.self_attn.q_proj"}]
    model = SimpleNamespace(quant_log=[])

    record_quant_log(
        model,
        [
            _StubProcessor("native", []),
            _StubProcessor("foem", primary),
            _StubProcessor("eora", adapter, owns_quant_log=False),
        ],
    )

    assert model.quant_log == primary


def test_empty_log_does_not_clobber_restored_quant_log() -> None:
    restored = [{PROCESS_LOG_MODULE: "model.layers.0.mlp.down_proj"}]
    model = SimpleNamespace(quant_log=restored)

    record_quant_log(model, [_StubProcessor("foem", [])])

    assert model.quant_log is restored


def test_gptq_eora_chain_publishes_one_row_per_module() -> None:
    """#4367: the adapter pass must not append a second, blank row per module."""

    primary = [
        {
            PROCESS_LOG_LAYER: 0,
            PROCESS_LOG_MODULE: "self_attn.q_proj",
            QUANT_LOG_LOSS: "0.1000000000",
            QUANT_LOG_NSAMPLES: "128",
            QUANT_LOG_DAMP: "0.01000",
        },
        {
            PROCESS_LOG_LAYER: 0,
            PROCESS_LOG_MODULE: "mlp.down_proj",
            QUANT_LOG_LOSS: "0.2000000000",
            QUANT_LOG_NSAMPLES: "128",
            QUANT_LOG_DAMP: "0.01000",
        },
    ]
    adapter = [
        {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: "self_attn.q_proj", PROCESS_LOG_TIME: "0.5"},
        {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: "mlp.down_proj", PROCESS_LOG_TIME: "0.5"},
    ]
    model = SimpleNamespace(quant_log=[])

    record_quant_log(
        model,
        [
            _StubProcessor("gptq", primary),
            _StubProcessor("eora", adapter, owns_quant_log=False),
        ],
    )

    assert model.quant_log == primary
    assert len(model.quant_log) == 2
    assert [row[PROCESS_LOG_MODULE] for row in model.quant_log] == [
        "self_attn.q_proj",
        "mlp.down_proj",
    ]
    # No EoRA-only blank copy leaked into the model log.
    assert all(row.get(QUANT_LOG_LOSS) for row in model.quant_log)


def test_eora_generate_pass_does_not_replace_primary_log() -> None:
    """#4367: the standalone EoRA loop must not clobber a restored quant log."""

    restored = [
        {
            PROCESS_LOG_LAYER: 0,
            PROCESS_LOG_MODULE: "self_attn.q_proj",
            QUANT_LOG_LOSS: "0.1000000000",
            QUANT_LOG_NSAMPLES: "128",
            QUANT_LOG_DAMP: "0.01000",
        }
    ]
    model = SimpleNamespace(quant_log=list(restored))
    eora_rows = [
        {PROCESS_LOG_LAYER: 0, PROCESS_LOG_MODULE: "self_attn.q_proj", PROCESS_LOG_TIME: "0.5"}
    ]

    record_quant_log(
        model,
        [
            _StubProcessor("dequantize"),
            _StubProcessor("eora", eora_rows, owns_quant_log=False),
        ],
    )

    assert model.quant_log == restored


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
