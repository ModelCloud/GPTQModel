# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Every quantizer must publish ``model.quant_log`` and write ``quant_log.csv``.

`ModuleLooper` used to copy a processor's log into `model.quant_log` only when
the processor name was one of `gptq/gptaq/awq/exl3/macaq`, so FOEM, QQQ, VPTQ,
Paro and FP8-GSQ runs ended with an empty log and the writer skipped
`quant_log.csv` (issue #4321). The model-level log is now assembled in one place
from the first log-owning processor, and both loopers use it.
"""

import csv
import os

import pytest
import torch

from gptqmodel import CalibrationConfig
from gptqmodel.looper.loop_processor import record_quant_log


# Processor labels that the old allow-list did not know about. The wiring test
# below must publish all of them without naming any of them in the looper.
_LOGGING_PROCESSORS = (
    "foem",
    "qqq",
    "vptq",
    "paro",
    "fp8_gsq",
    "weight_only_rtn",
)


class _Model:
    quant_log = ["stale-entry"]


class _Processor:
    """Minimal processor exposing the attributes ``record_quant_log`` reads."""

    def __init__(self, name, log=None, *, owns_quant_log=True):
        self._name = name
        self.log = [] if log is None else log
        self.owns_quant_log = owns_quant_log

    def name(self):
        return self._name


@pytest.mark.parametrize("name", _LOGGING_PROCESSORS)
def test_record_quant_log_publishes_every_processor(name):
    model = _Model()
    processor_log = [{"module": f"{name}-0"}, {"module": f"{name}-1"}]

    entries = record_quant_log(model, [_Processor(name, processor_log)])

    assert model.quant_log == processor_log
    assert entries == processor_log


def test_record_quant_log_skips_processors_without_entries():
    model = _Model()
    record_quant_log(
        model,
        [
            _Processor("native"),
            _Processor("foem", [{"module": "a"}]),
            _Processor("eora", None, owns_quant_log=False),
        ],
    )
    assert model.quant_log == [{"module": "a"}]


def test_record_quant_log_does_not_alias_the_processor_log():
    model = _Model()
    processor_log = [{"module": "a"}]
    record_quant_log(model, [_Processor("qqq", processor_log)])
    model.quant_log.append({"module": "b"})
    assert processor_log == [{"module": "a"}]


def test_record_quant_log_does_not_concatenate_auxiliary_rows():
    """#4367: the EoRA pass must not append a second row per module."""

    model = _Model()
    primary = [{"module": "a", "loss": "1.0"}]
    adapter = [{"module": "a"}]

    record_quant_log(
        model,
        [
            _Processor("gptq", primary),
            _Processor("eora", adapter, owns_quant_log=False),
        ],
    )

    assert model.quant_log == primary


def test_record_quant_log_preserves_restored_log_for_auxiliary_only_pass():
    """#4367: the standalone EoRA-generate loop has no primary log to publish."""

    restored = [{"module": "a", "loss": "1.0"}]
    model = _Model()
    model.quant_log = list(restored)

    record_quant_log(
        model,
        [
            _Processor("dequantize"),
            _Processor("eora", [{"module": "a"}], owns_quant_log=False),
        ],
    )

    assert model.quant_log == restored


def _tiny_llama(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).half().eval()
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="0")),
        unk_token="0", pad_token="0", eos_token="0",
    )
    tokenizer.save_pretrained(path)
    return str(path)


def _configs():
    from gptqmodel.quantization.config import (
        FOEMConfig,
        FP8Config,
        GSQConfig,
        ParoConfig,
        QQQConfig,
        QuantizeConfig,
        RTNConfig,
        VPTQConfig,
    )

    return {
        "gptq": lambda: QuantizeConfig(bits=4, group_size=32),
        "rtn": lambda: RTNConfig(bits=4, group_size=32),
        "qqq": lambda: QQQConfig(bits=4, group_size=128),
        "vptq": lambda: VPTQConfig(vector_lens=[-1, 4], num_centroids=[-1, 4],
                                   num_res_centroids=[-1, 2], kmeans_iters=2,
                                   enable_norm=True, enable_perm=True, group_size=-1),
        "paro": lambda: ParoConfig(bits=4, group_size=32, opt_rotation_epochs=1,
                                        opt_finetune_epochs=0, opt_train_samples=4,
                                        opt_validation_samples=4, opt_batch_size=2),
        "fp8_gsq": lambda: FP8Config(gsq=GSQConfig(enabled=True, steps=1, candidates=2),
                                     gsq_calibration=True),
        "foem": lambda: QuantizeConfig(bits=4, group_size=32,
                                       foem=FOEMConfig(alpha=0.25, beta=0.2)),
    }


@pytest.mark.parametrize("method", ["gptq", "rtn", "qqq", "vptq", "paro", "fp8_gsq", "foem"])
def test_every_quantizer_writes_quant_log_csv(method, tmp_path):
    """Fail-before/pass-after coverage for #4321 (the log used to be empty)."""

    from gptqmodel import GPTQModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if method == "foem" and device == "cpu":
        pytest.skip("FOEM activation capture currently needs CUDA (unrelated CPU device bug)")

    source = _tiny_llama(tmp_path / "source")
    model = GPTQModel.load(source, quantize_config=_configs()[method](), device=device)
    model.quantize(CalibrationConfig(dataset=[{"input_ids": list(range(1, 33))}], min_length=1))

    assert len(model.quant_log) > 0, f"{method}: quant_log is empty"
    for entry in model.quant_log:
        assert isinstance(entry, dict)
        assert entry.get("layer") is not None
        assert entry.get("module")

    out = tmp_path / "quantized"
    model.save(str(out))
    csv_path = out / "quant_log.csv"
    assert csv_path.is_file(), f"{method}: quant_log.csv was not written"
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["layer", "module", "loss", "loss_kind", "samples", "damp", "time"]
    assert len(rows) == len(model.quant_log) + 1
    assert all(row[1] for row in rows[1:])
