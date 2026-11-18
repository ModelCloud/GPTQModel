# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 families must equal Transformers 5.23 exactly.

Each fixture under tests/fixtures/ports/native_parity was dumped from
Transformers' own model tester by scripts/generate_native_port_parity_fixtures.py.
For every model class, the native class (the oracle; tests may import
``transformers.models``) and the vendored class are built from the same config,
must have identical state_dict keys/shapes, and with identical weights must
produce bit-identical outputs.
"""

import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"
CASES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def _classes(family, module, name):
    native = getattr(importlib.import_module(f"transformers.models.{family}.{module}"), name)
    vendored = getattr(importlib.import_module(f"gptqmodel.models.ports.native.{family}.{module}"), name)
    return native, vendored


def _inputs(family, spec):
    tensors = {}
    path = FIXTURES / f"{family}.safetensors"
    if path.exists():
        tensors.update(load_file(str(path)))
    generator = torch.Generator().manual_seed(0)
    for key in sorted(spec["float_inputs"]):
        meta = spec["float_inputs"][key]
        dtype = getattr(torch, meta["dtype"])
        tensors[key] = torch.randn(meta["shape"], generator=generator, dtype=torch.float32).to(dtype)
    tensors.update(spec["extra_inputs"])
    return tensors


def _flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict) or hasattr(output, "to_tuple"):
        values = output.values() if isinstance(output, dict) else output.to_tuple()
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        return []
    return [tensor for value in values for tensor in _flatten(value)]


# Families whose *native* Transformers forward cannot run on the tester inputs here.
ORACLE_UNAVAILABLE = {
    "inkling": "native Inkling requires scipy",
    "roberta": "vendored only as a dependency; the tester's inputs target encoder heads",
}


# gptqmodel/__init__.py replaces Transformers' causal-conv1d hub kernel globally;
# its kernel rejects the tester's activation in both native and vendored Jamba.
# strict: this must start passing (and the marker be removed) once that patch goes.
GLOBAL_KERNEL_OVERRIDE = {"jamba"}


@pytest.mark.parametrize(
    "family",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason="global causal-conv1d kernel override"))
        if name in GLOBAL_KERNEL_OVERRIDE
        else name
        for name in CASES
    ],
)
def test_vendored_family_matches_transformers(family):
    if family in ORACLE_UNAVAILABLE:
        if family == "inkling":
            pytest.importorskip("scipy", reason=ORACLE_UNAVAILABLE[family])
        else:
            pytest.skip(ORACLE_UNAVAILABLE[family])
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    native_config_class, vendored_config_class = _classes(family, spec["config_module"], spec["config_class"])
    assert vendored_config_class is not native_config_class
    inputs = _inputs(family, spec)

    for module, name in spec["model_classes"]:
        native_class, vendored_class = _classes(family, module, name)
        native_config = native_config_class.from_dict(dict(spec["config"]))
        vendored_config = vendored_config_class.from_dict(dict(spec["config"]))
        torch.manual_seed(0)
        native = native_class(native_config).eval()
        torch.manual_seed(0)
        vendored = vendored_class(vendored_config).eval()

        native_state = native.state_dict()
        vendored_state = vendored.state_dict()
        assert list(vendored_state) == list(native_state), f"{name}: state_dict keys differ"
        assert all(vendored_state[k].shape == native_state[k].shape for k in native_state), f"{name}: shapes differ"
        vendored.load_state_dict(native_state, strict=True)
        for key, buffer in native.named_buffers():
            vendored.get_buffer(key).copy_(buffer)

        with torch.no_grad():
            expected = _flatten(native(**inputs))
            actual = _flatten(vendored(**inputs))
        assert expected, f"{name}: no tensor outputs"
        assert len(actual) == len(expected), f"{name}: output structure differs"
        for index, (want, got) in enumerate(zip(expected, actual)):
            assert torch.equal(want, got), f"{name}: output {index} differs"


def test_every_vendored_family_has_parity_coverage():
    native = Path(__file__).resolve().parents[1] / "gptqmodel" / "models" / "ports" / "native"
    families = {p.name for p in native.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"}}
    # No Transformers 5.23 model tester exists for these; they need hand-written parity tests.
    without_tester = {"bart", "llama4", "opt"}
    assert families - without_tester <= set(CASES), sorted(families - without_tester - set(CASES))
