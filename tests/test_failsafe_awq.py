import types
from types import SimpleNamespace

import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.pause_resume import PauseResumeController


def _dummy_prepare_dataset(
    *,
    calibration_dataset,
    calibration_dataset_concat_size,
    calibration_dataset_sort,
    batch_size,
    calibration_concat_separator=None,
):
    return calibration_dataset


class _DummyProgressBar:
    def title(self, _):
        return self

    def subtitle(self, _):
        return self

    def draw(self):
        return None


def test_awq_failsafe_falls_back_to_rtn_when_no_activations(monkeypatch):
    model = nn.Module()
    model.linear = nn.Linear(8, 8, bias=False)

    gptq_model = SimpleNamespace(model=model, lm_head=None, quant_region_timer=None)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=-1,
        failsafe={"strategy": "rtn", "threshold": "1.0%"},
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
    )
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=_dummy_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=gptq_model,
        model=model,
        require_fwd=False,
        calculate_w_wq_diff=False,
    )
    processor._pause_controller = PauseResumeController()
    processor.pb = _DummyProgressBar()

    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)
    processor.preprocess(named, failsafe=qcfg.failsafe)

    calls = {}

    def fake_pack(self, nm):
        calls["called"] = True
        calls["name"] = nm.full_name
        nm.state["wq"] = nm.module.weight.detach().clone()

    processor.pack_module = types.MethodType(fake_pack, processor)

    processor.process(named)

    processor.submodule_finalize(named, gptq_model)

    layer_state = processor._get_layer_state(0)

    assert calls.get("called") is True
    assert calls.get("name") == "linear"
    assert layer_state.quantized is True
    assert "wq" in named.state
