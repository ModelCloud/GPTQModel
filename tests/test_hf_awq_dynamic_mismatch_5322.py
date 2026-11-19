# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Transformers 5.23 must not reject a ``dynamic`` AWQ checkpoint's packed tensors (#5322).

Transformers 5.23 changed ``core_model_loading.set_param_for_module`` so that a
checkpoint tensor whose shape differs from the meta module is recorded as a
mismatched key unless ``hf_quantizer.param_needs_quantization`` is true:

    ref is not None and param_value.shape != expected_shape
    and (hf_quantizer is None or not hf_quantizer.param_needs_quantization(...))

Before 5.23 any active quantizer skipped the check. ``AwqQuantizer`` inherits
the base implementation (always ``False``) because a pre-quantized checkpoint
has nothing to quantize, so on 5.23 the per-module ``qzeros``/``scales`` of a
``dynamic`` checkpoint were reported MISMATCH and ``_finalize_model_loading``
raised before the post-weight repair hook could rebuild the modules.

The bridge must therefore claim those packed buffers, and only for a
GPT-QModel AWQ checkpoint whose packed module genuinely disagrees with the
checkpoint tensor. Dense params and third-party AWQ checkpoints keep
Transformers' own behavior.
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils import hf_load


class _PackedAwqModule(BaseQuantLinear):
    """Minimal packed AWQ module: enough identity for the mismatch guard."""

    def __init__(self):
        # BaseQuantLinear.__init__ builds a full kernel; this fixture only needs
        # the type identity the guard checks, plus the packed buffers.
        nn.Module.__init__(self)
        self.qweight = nn.Parameter(torch.zeros(1, 1, dtype=torch.int32), requires_grad=False)
        self.qzeros = nn.Parameter(torch.zeros(1, 1, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.zeros(1, 1, dtype=torch.float16), requires_grad=False)


class _TinyModel(nn.Module):
    def __init__(self, dynamic=None, quant_method="awq", quantizer=("gptqmodel:7.11.6",)):
        super().__init__()
        self.proj = _PackedAwqModule()
        self.dense = nn.Linear(4, 4)
        meta = {"quantizer": list(quantizer)} if quantizer is not None else {}
        quantize_config = {"quant_method": quant_method, "bits": 4, "group_size": 128}
        if dynamic:
            quantize_config["dynamic"] = dynamic
        if meta:
            quantize_config["meta"] = meta
        self.config = type(
            "Config",
            (),
            {"quantization_config": quantize_config, "model_type": "qwen2"},
        )()


@pytest.mark.parametrize("buffer", ["qweight", "qzeros", "scales"])
def test_native_dynamic_awq_packed_buffers_are_tolerated(buffer):
    model = _TinyModel(dynamic={r"+:.*layers\.1\..*": {"group_size": 32}})

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, f"proj.{buffer}") is True


def test_dense_parameter_mismatch_is_still_reported():
    model = _TinyModel(dynamic={r"+:.*layers\.1\..*": {"group_size": 32}})

    # A dense tensor whose shape disagrees with the checkpoint is a real error.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "dense.weight") is False
    # A packed suffix on a module that is not a packed kernel is not tolerated.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "dense.qzeros") is False


def test_third_party_awq_checkpoint_keeps_transformers_behavior():
    model = _TinyModel(
        dynamic={r"+:.*layers\.1\..*": {"group_size": 32}},
        quantizer=("awq:0.2.0",),
    )

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_non_awq_checkpoint_is_untouched():
    model = _TinyModel(dynamic={r"+:.*layers\.1\..*": {"group_size": 32}}, quant_method="gptq")

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_gptqmodel_provenance_without_dynamic_is_still_native():
    model = _TinyModel(quantizer=("gptqmodel:7.11.6",))

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is True


def test_awq_moe_expert_per_expert_tensors_are_not_tolerated():
    """Only a packed module's own buffers may disagree with the checkpoint."""

    model = _TinyModel(dynamic={r"+:.*layers\.1\..*": {"group_size": 32}})

    # `experts.0.gate_proj` is not installed on this fixture, so the guard must
    # decline instead of claiming an unknown parameter path.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(
        model, "layers.0.mlp.experts.0.gate_proj.qzeros"
    ) is False
