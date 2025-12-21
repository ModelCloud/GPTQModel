import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.failsafe import should_use_rtn_failsafe


def test_should_use_rtn_failsafe_parses_numeric_and_percent():
    assert should_use_rtn_failsafe(True, observed_samples=0, expected_total_samples=100)
    assert not should_use_rtn_failsafe(True, observed_samples=1, expected_total_samples=100)

    assert should_use_rtn_failsafe("10", observed_samples=5, expected_total_samples=100)
    assert not should_use_rtn_failsafe("10", observed_samples=11, expected_total_samples=100)

    assert should_use_rtn_failsafe("10%", observed_samples=9, expected_total_samples=90)
    assert should_use_rtn_failsafe("10%", observed_samples=10, expected_total_samples=200)


def test_gptq_failsafe_threshold_triggers_rtn_when_samples_below_percent():
    torch.manual_seed(0)
    layer = nn.Linear(8, 8, bias=False)

    qcfg = QuantizeConfig(bits=4, group_size=4, failsafe_with_rtn="75%")
    gptq = GPTQ(layer, qcfg)
    gptq.failsafe_with_rtn = qcfg.failsafe_with_rtn
    gptq.expected_nsamples = 4  # pretend we expected 4 token rows
    gptq.quantizer.configure(perchannel=True)

    # Capture only a single token worth of activations (< 75% of expected total)
    inp = torch.randn(1, 1, 8)
    gptq.add_batch(inp, None)

    _, _, _, _, _, avg_loss, _, nsamples = gptq.quantize(blocksize=4)

    assert nsamples == 1
    assert avg_loss == "rtn failsafe"
