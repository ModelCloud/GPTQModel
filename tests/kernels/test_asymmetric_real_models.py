# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from typing import Iterable

import pytest
import torch
from tests.models.model_test import ModelTest

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.model,
    pytest.mark.slow,
]


_DEVICE = torch.device("cuda:0")
_DTYPE = torch.float16
_PROMPT = "What is the surface area of the Sun?"
_AWQ_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
_GPTQ_MODEL_ID = "ruikangliu/DeepSeek-R1-Distill-Qwen-1.5B-quantized.gptq-gptqmodel-w4g128"
_TARGET_MODULE = "model.layers.0.mlp.up_proj"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "approximately",
    "around",
    "as",
    "at",
    "be",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "roughly",
    "the",
    "to",
}


@dataclass(frozen=True)
class _RealModelCase:
    name: str
    model_id: str
    baseline_backend: BACKEND
    candidate_backend: BACKEND
    baseline_cls: type[torch.nn.Module]
    candidate_cls: type[torch.nn.Module]
    atol: float
    rtol: float


_AWQ_REAL_CASES = (
    _RealModelCase(
        name="awq_marlin",
        model_id=_AWQ_MODEL_ID,
        baseline_backend=BACKEND.TORCH_AWQ,
        candidate_backend=BACKEND.MARLIN,
        baseline_cls=AwqTorchLinear,
        candidate_cls=AwqMarlinLinear,
        atol=1e-2,
        rtol=1e-2,
    ),
    _RealModelCase(
        name="awq_machete",
        model_id=_AWQ_MODEL_ID,
        baseline_backend=BACKEND.TORCH_AWQ,
        candidate_backend=BACKEND.MACHETE,
        baseline_cls=AwqTorchLinear,
        candidate_cls=AwqMacheteLinear,
        atol=1.5e-2,
        rtol=1.5e-2,
    ),
)

_GPTQ_REAL_CASES = (
    _RealModelCase(
        name="gptq_machete",
        model_id=_GPTQ_MODEL_ID,
        baseline_backend=BACKEND.TORCH,
        candidate_backend=BACKEND.MACHETE,
        baseline_cls=TorchLinear,
        candidate_cls=MacheteLinear,
        atol=1.5e-2,
        rtol=1.5e-2,
    ),
)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters(recurse=False):
        if tensor is not None and not tensor.is_meta:
            return tensor.device
    for tensor in module.buffers(recurse=False):
        if tensor is not None and not tensor.is_meta:
            return tensor.device
    raise RuntimeError(f"Unable to infer runtime device for `{module.__class__.__name__}`.")


def _release_model(model) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(model_id: str, backend: BACKEND):
    model = GPTQModel.load(
        model_id,
        backend=backend,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert model.quantize_config.sym is False
    return model


def _target_module(model, expected_cls: type[torch.nn.Module]) -> torch.nn.Module:
    module_map = dict(model.model.named_modules())
    if _TARGET_MODULE not in module_map:
        raise KeyError(f"Target module `{_TARGET_MODULE}` not found in model `{model.model_id_or_path}`.")
    module = module_map[_TARGET_MODULE]
    assert isinstance(module, expected_cls), (
        f"Expected `{_TARGET_MODULE}` to use `{expected_cls.__name__}`, got `{module.__class__.__name__}`."
    )
    return module


def _layer_inputs(in_features: int, *, device: torch.device) -> list[torch.Tensor]:
    torch.manual_seed(17)
    return [
        torch.randn((1, in_features), device=device, dtype=_DTYPE),
        torch.randn((8, in_features), device=device, dtype=_DTYPE),
        torch.randn((2, 3, in_features), device=device, dtype=_DTYPE),
    ]


def _forward_module(module: torch.nn.Module, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    module_device = _module_device(module)
    with torch.inference_mode():
        for x in inputs:
            current = x if x.device == module_device else x.to(module_device)
            outputs.append(module(current).detach().to(device="cpu", dtype=torch.float32))
    torch.cuda.synchronize(module_device)
    return outputs


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalized_text(text))
        if token not in _STOPWORDS
    }


def _assert_solar_answer_not_garbled(text: str) -> None:
    normalized = _normalized_text(text)
    assert normalized, "generation output is empty"
    assert "\ufffd" not in normalized
    assert sum(ch.isprintable() for ch in normalized) / len(normalized) > 0.98
    assert any(term in normalized for term in ("surface", "square", "kilometer", "kilometers", "km", "solar", "sun")), (
        f"expected a surface-area style answer, got: {text}"
    )
    assert re.search(r"\d", normalized) or any(
        term in normalized for term in ("sphere", "formula", "radius")
    ), f"expected a numeric or formula-based answer, got: {text}"


def _generation_inputs(tokenizer):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": _PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = _PROMPT

    inputs = tokenizer(prompt_text, return_tensors="pt")
    return prompt_text, inputs, int(inputs.input_ids.shape[1])


def _generate_completion(model) -> str:
    tokenizer = model.tokenizer
    prompt_text, inputs, decode_start_idx = _generation_inputs(tokenizer)
    return ModelTest.generate_stable_with_limit(
        model,
        tokenizer,
        prompt_text,
        inputs=inputs,
        decode_start_idx=decode_start_idx,
        max_new_tokens=48,
    )


def _assert_generation_tracks_torch(candidate_text: str, baseline_text: str) -> None:
    _assert_solar_answer_not_garbled(baseline_text)
    _assert_solar_answer_not_garbled(candidate_text)

    baseline_tokens = _content_tokens(baseline_text)
    candidate_tokens = _content_tokens(candidate_text)
    shared = baseline_tokens & candidate_tokens

    assert len(shared) >= 4, (
        "candidate generation diverged too far from torch baseline.\n"
        f"baseline={baseline_text!r}\n"
        f"candidate={candidate_text!r}"
    )
    assert any(
        token in shared for token in {"surface", "area", "sun", "solar", "square", "kilometer", "kilometers", "km"}
    ), (
        "candidate generation does not preserve the core answer terms from torch baseline.\n"
        f"baseline={baseline_text!r}\n"
        f"candidate={candidate_text!r}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _AWQ_REAL_CASES, ids=lambda case: case.name)
def test_awq_asymmetric_real_layer_outputs_match_torch(case: _RealModelCase) -> None:
    baseline_model = None
    candidate_model = None
    try:
        baseline_model = _load_model(case.model_id, case.baseline_backend)
        baseline_module = _target_module(baseline_model, case.baseline_cls)
        inputs = _layer_inputs(baseline_module.in_features, device=_DEVICE)
        baseline_outputs = _forward_module(baseline_module, inputs)
    finally:
        _release_model(baseline_model)

    try:
        candidate_model = _load_model(case.model_id, case.candidate_backend)
        candidate_module = _target_module(candidate_model, case.candidate_cls)
        candidate_outputs = _forward_module(candidate_module, inputs)
    finally:
        _release_model(candidate_model)

    for actual, expected in zip(candidate_outputs, baseline_outputs):
        torch.testing.assert_close(actual, expected, atol=case.atol, rtol=case.rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _GPTQ_REAL_CASES, ids=lambda case: case.name)
def test_gptq_asymmetric_real_layer_outputs_match_torch(case: _RealModelCase) -> None:
    baseline_model = None
    candidate_model = None
    try:
        baseline_model = _load_model(case.model_id, case.baseline_backend)
        baseline_module = _target_module(baseline_model, case.baseline_cls)
        inputs = _layer_inputs(baseline_module.in_features, device=_DEVICE)
        baseline_outputs = _forward_module(baseline_module, inputs)
    finally:
        _release_model(baseline_model)

    try:
        candidate_model = _load_model(case.model_id, case.candidate_backend)
        candidate_module = _target_module(candidate_model, case.candidate_cls)
        candidate_outputs = _forward_module(candidate_module, inputs)
    finally:
        _release_model(candidate_model)

    for actual, expected in zip(candidate_outputs, baseline_outputs):
        torch.testing.assert_close(actual, expected, atol=case.atol, rtol=case.rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _AWQ_REAL_CASES + _GPTQ_REAL_CASES, ids=lambda case: f"{case.name}_generation")
def test_asymmetric_real_generation_matches_torch_and_is_sane(case: _RealModelCase) -> None:
    baseline_model = None
    candidate_model = None
    try:
        baseline_model = _load_model(case.model_id, case.baseline_backend)
        baseline_text = _generate_completion(baseline_model)
    finally:
        _release_model(baseline_model)

    try:
        candidate_model = _load_model(case.model_id, case.candidate_backend)
        candidate_text = _generate_completion(candidate_model)
    finally:
        _release_model(candidate_model)

    _assert_generation_tracks_torch(candidate_text, baseline_text)
