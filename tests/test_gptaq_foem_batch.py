# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM consume one native calibration entry per sample.

`NativeProcessor` captures one entry per forward *batch*, while `GPTQProcessor`
calls `add_batch` once per *sample*. Before the split, sample 0 consumed the
whole batch tensor and later samples hit an empty list. These tests pin the
per-sample split against the batch_size=1 flow over the same samples.
"""

import threading

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


def _qcfg(kind):
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig())
    return QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25))


def _cls(kind):
    return GPTAQ if kind == "gptaq" else FOEM


def _run(kind, module, batches):
    """Capture ``batches`` forwards, then replay them through the quantizer."""

    qcfg = _qcfg(kind)
    named = NamedModule(module, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor = NativeProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(dataset=None, batch_size=2),
    )
    processor.preprocess(named)
    hook = processor.pre_process_fwd_hook(named.name)
    for x, keep_mask in batches:
        # ModuleLooper stores the active [B, S] keep-mask on the processor's
        # thread-local storage before every forward pass.
        processor._mask_tls = threading.local()
        processor._mask_tls.value = keep_mask
        hook(module, (x,), None)
    processor.process(named)

    quantizer = _cls(kind)(named, qcfg)
    for x, keep_mask in batches:
        for sample_index in range(x.shape[0]):
            if not keep_mask[sample_index].any():
                continue
            # GPTQProcessor slices the same per-sample keep-mask before it
            # calls add_batch.
            sample = x[sample_index : sample_index + 1][:, keep_mask[sample_index]]
            quantizer.add_batch(sample, None)
    return quantizer


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_batched_native_capture_matches_per_sample_reference(kind):
    torch.manual_seed(0)
    module = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 3, 8)
    keep_mask = torch.tensor([[True, True, True], [True, True, False]])

    batched = _run(kind, module, [(x, keep_mask)])
    reference = _run(
        kind,
        module,
        [(x[0:1], keep_mask[0:1]), (x[1:2], keep_mask[1:2])],
    )

    torch.testing.assert_close(batched.H, reference.H, rtol=1e-6, atol=1e-6)
    if kind == "gptaq" or batched.gptaq:
        assert batched.dXXT is not None
        torch.testing.assert_close(batched.dXXT, reference.dXXT, rtol=1e-6, atol=1e-6)
    assert batched.nsamples == reference.nsamples
    assert torch.isfinite(batched.H).all()


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_native_capture_skips_empty_samples_like_gptq_processor(kind):
    torch.manual_seed(1)
    module = nn.Linear(8, 4, bias=False)
    x = torch.randn(3, 2, 8)
    keep_mask = torch.tensor([[True, True], [False, False], [True, False]])

    quantizer = _run(kind, module, [(x, keep_mask)])

    # GPTQProcessor skips samples with no kept tokens, so the native list must
    # skip them too or every later sample would read the wrong entry.
    assert torch.isfinite(quantizer.H).all()
    assert quantizer.H.shape == (8, 8)
