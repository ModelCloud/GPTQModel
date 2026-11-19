# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Transformers 5.23 AWQ packed-shape guards must track the serialized config.

Transformers 5.23 changed ``core_model_loading.set_param_for_module`` so that a
checkpoint tensor whose shape differs from the meta module is recorded as a
mismatched key unless ``hf_quantizer.param_needs_quantization`` is true:

    ref is not None and param_value.shape != expected_shape
    and (hf_quantizer is None or not hf_quantizer.param_needs_quantization(...))

Before 5.23 any active quantizer skipped the check. ``AwqQuantizer`` inherits
the base implementation (always ``False``) because a pre-quantized checkpoint
has nothing to quantize, so on 5.23 the per-module ``qzeros``/``scales`` of a
``dynamic`` checkpoint were reported MISMATCH and ``_finalize_model_loading``
raised before the post-weight repair hook could rebuild the modules (#5322).

The bridge may claim a packed buffer only when the built module's geometry
disagrees with that module's serialized config: that is the case the repair
rebuilds. A native checkpoint without such an override must keep Transformers'
rejection instead of accepting a corrupted tensor silently (#5345). Dense
params and third-party AWQ checkpoints keep Transformers' own behavior.
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils import hf_load


class _PackedAwqModule(BaseQuantLinear):
    """Minimal packed AWQ module: enough identity for the mismatch guard."""

    def __init__(self, bits=4, group_size=128, desc_act=False, sym=True):
        # BaseQuantLinear.__init__ builds a full kernel; this fixture only needs
        # the type identity and geometry the guard compares, plus the buffers.
        nn.Module.__init__(self)
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
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
def test_native_dynamic_geometry_override_is_tolerated(buffer):
    # Transformers built `proj` from the global group size; the checkpoint
    # serializes group size 32 for it, so the repair will rebuild it.
    model = _TinyModel(dynamic={"proj": {"group_size": 32}})

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, f"proj.{buffer}") is True


def test_native_dynamic_bits_override_is_tolerated():
    model = _TinyModel(dynamic={"proj": {"bits": 8}})

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is True


def test_native_awq_without_dynamic_rejects_packed_shape_mismatch():
    # #5345: no per-module override means the built module already has the
    # serialized geometry; a differing tensor is a corrupted checkpoint.
    model = _TinyModel()

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.scales") is False


def test_native_dynamic_override_on_another_module_does_not_tolerate():
    # `proj` has no override of its own: a mismatch there is still corruption.
    model = _TinyModel(dynamic={"other": {"group_size": 32}})

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_native_dynamic_override_with_same_geometry_is_not_tolerated():
    model = _TinyModel(dynamic={"proj": {"group_size": 128}})

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_dense_parameter_mismatch_is_still_reported():
    model = _TinyModel(dynamic={"proj": {"group_size": 32}})

    # A dense tensor whose shape disagrees with the checkpoint is a real error.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "dense.weight") is False
    # A packed suffix on a module that is not a packed kernel is not tolerated.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "dense.qzeros") is False


def test_third_party_awq_checkpoint_keeps_transformers_behavior():
    model = _TinyModel(
        dynamic={"proj": {"group_size": 32}},
        quantizer=("awq:0.2.0",),
    )

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_non_awq_checkpoint_is_untouched():
    model = _TinyModel(dynamic={"proj": {"group_size": 32}}, quant_method="gptq")

    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(model, "proj.qzeros") is False


def test_awq_moe_expert_per_expert_tensors_are_not_tolerated():
    """Only a packed module's own buffers may disagree with the checkpoint."""

    model = _TinyModel(dynamic={"proj": {"group_size": 32}})

    # `experts.0.gate_proj` is not installed on this fixture, so the guard must
    # decline instead of claiming an unknown parameter path.
    assert hf_load.hf_gptqmodel_awq_tolerated_mismatch(
        model, "layers.0.mlp.experts.0.gate_proj.qzeros"
    ) is False
