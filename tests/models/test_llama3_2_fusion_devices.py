# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Local Llama 3.2 1B checkpoint inference on each actual device family."""

import os
from pathlib import Path

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.backend import BACKEND
from tests.benchmark_llama_fusion import drift, storage_footprint
from tests.module_tree.test_fusion_device_inference import _assert_no_expanded_weight_cache, _cached_decode, _target


@pytest.mark.parametrize("family", [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM])
@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_llama32_checkpoint_fusion_on_actual_device(family, method, dtype, record_property):
    defaults = {
        "gptq": "/monster/data/model/Llama-3.2-1B-Instruct-HF-GPTQV2",
        "awq": "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-bit4-g128-symTrue-descFalse",
    }
    path = os.environ.get(f"GPTQMODEL_LLAMA32_FUSION_{method.upper()}", defaults[method])
    if not Path(path).is_dir():
        pytest.skip(f"Local symmetric INT4/group-128 Llama 3.2 1B {method} checkpoint missing: {path}")
    device = _target(family)
    model = GPTQModel.load(path, backend=BACKEND.AUTO, device=family, dtype=dtype, attn_implementation="eager")
    kernel = {
        (DEVICE.CPU, "gptq"): ("TorchAtenLinear", "TorchCpuAvx512Linear"),
        (DEVICE.CPU, "awq"): ("TorchAtenAwqLinear", "TorchCpuAvx512AwqLinear"),
        (DEVICE.CUDA, "gptq"): ("MarlinLinear", "MarlinLinear"),
        (DEVICE.CUDA, "awq"): ("AwqMarlinLinear", "AwqMarlinLinear"),
        (DEVICE.ROCM, "gptq"): ("TorchLinear", "TorchLinear"),
        (DEVICE.ROCM, "awq"): ("AwqGEMMTritonLinear", "AwqTorchLinear"),
    }[family, method][dtype == torch.bfloat16]
    assert {type(m).__name__ for m in model.model.modules() if isinstance(m, BaseQuantLinear)} == {kernel}
    assert model.quantize_config.bits == 4 and model.quantize_config.group_size == 128
    assert model.quantize_config.sym and not model.quantize_config.desc_act
    assert model._fusion_plan is None
    inputs = model.tokenizer("The capital of France is", return_tensors="pt").to(device)
    with torch.inference_mode():
        before = _cached_decode(model.model, inputs["input_ids"])
        repeats = [_cached_decode(model.model, inputs["input_ids"]) for _ in range(2)]
        tokens = model.model.generate(**inputs, max_new_tokens=4, do_sample=False)
    repeat_max = max(drift(value, before)["max_abs"] for value in repeats)
    storage = storage_footprint(model)
    planner = model.fuse()
    groups = 16 if kernel == "AwqTorchLinear" else 32
    assert len(planner.groups) == groups
    if groups == 16:
        assert all("AWQ GQA" in d.reason for d in planner.decisions if not d.enabled)
    with torch.inference_mode():
        after = _cached_decode(model.model, inputs["input_ids"])
        generated = model.model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert torch.equal(after.argmax(-1), before.argmax(-1))
    assert torch.equal(generated, tokens)
    assert planner.to_dict()["fused_calls"] > 0
    assert {type(g.packed).__name__ for g in planner.groups} == {kernel}
    fused_storage = storage_footprint(model, planner)
    assert fused_storage["packed_operand_storage_bytes"] == storage["packed_operand_storage_bytes"]
    for group in planner.groups:
        _assert_no_expanded_weight_cache(group.packed)
    model.fuse(False)
    with torch.inference_mode():
        restored = _cached_decode(model.model, inputs["input_ids"])
    if kernel == "AwqGEMMTritonLinear":
        # Split-K atomic reduction order can vary even before fusion. Packed
        # bytes are pinned exactly by the native backend tests; distinguish
        # this baseline noise from a change introduced by fusion/unfusion.
        assert drift(restored, before)["max_abs"] <= 2 * repeat_max + 2e-3
        assert torch.equal(restored.argmax(-1), before.argmax(-1))
    else:
        torch.testing.assert_close(restored, before, rtol=0, atol=0)
    metrics = drift(after, before)
    record_property("family", family.value)
    record_property("method", method)
    record_property("kernel", kernel)
    record_property("dtype", str(dtype))
    record_property("fused_groups", groups)
    record_property("max_abs", metrics["max_abs"])
    record_property("relative_l2", metrics["relative_l2"])
    record_property("rms", metrics["rms"])
    record_property("top1_agreement", metrics["top1_agreement"])
    record_property("generated_ids_equal", True)
    record_property("extra_packed_bytes", 0)
    record_property("baseline_repeat_max_abs", repeat_max)
    record_property("unfuse_max_abs", drift(restored, before)["max_abs"])
    record_property("torch", torch.__version__)
    record_property("hip", str(torch.version.hip))
