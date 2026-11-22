# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dump tiny configs and inputs from Transformers 5.23's own model testers.

Usage:
    python scripts/generate_native_port_parity_fixtures.py <transformers-v5.23.0-checkout> [family ...]

For every vendored family under gptqmodel/models/ports/native that has a model
tester in the Transformers checkout's ``tests/models/<family>``, writes
``tests/fixtures/ports/native_parity/<family>.json`` (config + model class
names) and ``<family>.safetensors`` (the exact tester inputs). The parity test
rebuilds the native and vendored models from these files only.
"""

import importlib
import inspect
import json
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[1]
NATIVE = REPO / "gptqmodel" / "models" / "ports" / "native"
OUT = REPO / "tests" / "fixtures" / "ports" / "native_parity"


def _test_cases(module):
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, unittest.TestCase) and obj.__module__ == module.__name__:
            if getattr(obj, "all_model_classes", None) or getattr(obj, "model_tester_class", None):
                yield obj


def _model_classes(case, tester):
    classes = getattr(case, "all_model_classes", None)
    if classes:
        return list(classes)
    return list(getattr(tester, "all_model_classes", ()) or ())


def _vendored(cls, family):
    module = cls.__module__.replace(f"transformers.models.{family}.", f"gptqmodel.models.ports.native.{family}.")
    if not module.startswith("gptqmodel."):
        return None
    try:
        return getattr(importlib.import_module(module), cls.__name__)
    except (ImportError, AttributeError):
        return None


def dump(family):
    try:
        module = importlib.import_module(f"tests.models.{family}.test_modeling_{family}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        return f"no tester ({type(exc).__name__})"
    for case in _test_cases(module):
        try:
            instance = case("setUp") if hasattr(case, "setUp") else case()
            instance.setUp()
            tester = instance.model_tester
            config, inputs = tester.prepare_config_and_inputs_for_common()
        except Exception as exc:  # noqa: BLE001
            last = f"tester failed ({type(exc).__name__}: {str(exc)[:80]})"
            continue
        classes = [
            cls
            for cls in _model_classes(case, tester)
            if not cls.__name__.endswith(SKIP_HEADS) and _vendored(cls, family) is not None
        ]
        if not classes:
            last = "no vendored model classes"
            continue
        # Integer/bool inputs (ids, masks, grids) carry structure and are stored
        # exactly. Floating inputs (pixels, audio) only need to be identical for
        # both models, so the test regenerates them from a fixed seed.
        tensors = {
            k: v.contiguous()
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor) and not v.is_floating_point()
        }
        float_inputs = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")}
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        }
        extras = {k: v for k, v in inputs.items() if not isinstance(v, torch.Tensor)}
        if any(not isinstance(v, (int, float, bool, str, type(None))) for v in extras.values()):
            last = "non-tensor inputs"
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        if tensors:
            save_file(tensors, str(OUT / f"{family}.safetensors"))
        (OUT / f"{family}.json").write_text(
            json.dumps(
                {
                    "config_class": type(config).__name__,
                    "config_module": type(config).__module__.split(".")[-1],
                    "config": config.to_dict(),
                    "model_classes": [[cls.__module__.split(".")[-1], cls.__name__] for cls in classes],
                    "extra_inputs": extras,
                    "float_inputs": float_inputs,
                },
                indent=1,
                sort_keys=True,
                default=str,
            )
        )
        return f"ok ({len(classes)} classes)"
    return locals().get("last", "no test case")


SKIP_HEADS = ("ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering", "ForMaskedLM")


def main():
    # The Transformers checkout's `tests` package must win over this repo's.
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    families = sys.argv[2:] or sorted(p.name for p in NATIVE.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"})
    for family in families:
        print(f"{family}: {dump(family)}", flush=True)


if __name__ == "__main__":
    main()
