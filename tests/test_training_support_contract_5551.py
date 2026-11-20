# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The trainable-kernel set is an explicit, documented decision (#5551).

A quant kernel opts into training. ``BaseQuantLinear`` declares
``SUPPORTS_TRAINING = None`` so a kernel that never states the decision cannot
silently inherit one, and the kernels that declare ``True`` are the small set
``docs/trainable-kernels.md`` names. Adding or dropping training support
therefore needs an explicit edit of the decision record, not an accident of
inheritance or a copy-paste from a sibling kernel.
"""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import gptqmodel.nn_modules.qlinear as qlinear
from gptqmodel.nn_modules.qlinear import BaseQuantLinear


#: Every kernel allowed to declare ``SUPPORTS_TRAINING = True``. Keep this, the
#: table in ``docs/trainable-kernels.md`` and the kernel in sync.
TRAINABLE_KERNELS = {
    "TorchLinear",
    "TritonV2Linear",
    "AwqTorchLinear",
    "AwqGEMMLinear",
    "AwqGEMMTritonLinear",
    "AwqGEMVTritonLinear",
    "TorchFusedLinear",
    "TorchFusedAwqLinear",
    "TorchAtenLinear",
    "TorchAtenAwqLinear",
    "TorchCpuAvx512Linear",
    "TorchCpuAvx512AwqLinear",
    "TorchFP8Linear",
    "GGUFTorchLinear",
    "ParoLinear",
    "ParoTritonLinear",
}

DOC = Path(__file__).resolve().parents[1] / "docs" / "trainable-kernels.md"


def _load_kernels():
    """Import every kernel module and return the concrete kernel classes."""

    for module in pkgutil.iter_modules(qlinear.__path__):
        if module.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{qlinear.__name__}.{module.name}")
        except Exception:
            # An optional accelerator dependency is not installed here; its
            # kernels cannot be selected on this host either.
            continue

    found = {}

    def walk(cls):
        for sub in cls.__subclasses__():
            found[sub.__name__] = sub
            walk(sub)

    walk(BaseQuantLinear)
    found = {
        name: cls for name, cls in found.items() if cls.__module__.startswith("gptqmodel")
    }

    def is_family(cls):
        """An intermediate family: no payload of its own, so it declares ``None``.

        ``AWQuantLinear``, ``GroupedQuantLinear`` and friends are families
        because another discovered kernel derives from them. A concrete kernel
        may also be a parent (``TorchAtenAwqLinear`` derives from
        ``TorchAtenLinear``), so a declared ``bool`` always wins over this rule.
        """

        if cls.SUPPORTS_TRAINING is not None:
            return False
        return any(
            other is not cls and issubclass(other, cls) for other in found.values()
        )

    return {name: cls for name, cls in found.items() if not is_family(cls)}


@pytest.fixture(scope="module")
def kernels():
    return _load_kernels()


def test_the_base_class_states_no_decision():
    assert BaseQuantLinear.SUPPORTS_TRAINING is None


def test_every_kernel_states_the_training_decision(kernels):
    assert kernels, "no quant kernels were discovered"
    undeclared = sorted(
        name
        for name, cls in kernels.items()
        if not isinstance(cls.SUPPORTS_TRAINING, bool)
    )
    assert not undeclared, (
        "these kernels never declare SUPPORTS_TRAINING, so they inherit `None` "
        f"instead of deciding: {undeclared}"
    )


def test_the_trainable_set_is_the_documented_one(kernels):
    trainable = {
        name for name, cls in kernels.items() if cls.SUPPORTS_TRAINING is True
    }
    assert trainable == TRAINABLE_KERNELS, (
        "the trainable kernel set changed without updating "
        "docs/trainable-kernels.md and TRAINABLE_KERNELS: "
        f"added {sorted(trainable - TRAINABLE_KERNELS)}, "
        f"removed {sorted(TRAINABLE_KERNELS - trainable)}"
    )


def test_the_decision_record_lists_every_trainable_kernel():
    text = DOC.read_text(encoding="utf-8")
    missing = sorted(name for name in TRAINABLE_KERNELS if f"`{name}`" not in text)
    assert not missing, f"docs/trainable-kernels.md does not list {missing}"


def test_the_decision_record_names_the_inference_only_families(kernels):
    """A kernel that is not trainable must be covered by the doc's rule."""

    text = DOC.read_text(encoding="utf-8")
    inference_only = {
        name for name, cls in kernels.items() if cls.SUPPORTS_TRAINING is False
    }
    assert inference_only, "every discovered kernel declared training support"
    # The record groups them instead of repeating 50 rows; make sure the
    # grouping sentence still covers the families the kernels live in.
    assert "inference-only" in text
    assert "Marlin" in text and "GGUF C++" in text


_DEVICE_LABELS = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "rocm": "ROCm",
    "xpu": "XPU",
    "npu": "NPU",
    "mps": "MPS",
}


def test_the_decision_record_lists_each_kernel_s_devices(kernels):
    """The device column must match ``SUPPORTS_DEVICES``, not an older guess."""

    text = DOC.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        match = re.match(r"^\| `([A-Za-z0-9_]+)`", line)
        if match is None:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[match.group(1)] = cells[2]

    wrong = {}
    for name in sorted(TRAINABLE_KERNELS):
        devices = {
            _DEVICE_LABELS.get(getattr(device, "value", str(device)).lower(), str(device))
            for device in kernels[name].SUPPORTS_DEVICES
        }
        documented = {label for label in _DEVICE_LABELS.values() if label in rows.get(name, "")}
        if devices != documented:
            wrong[name] = (sorted(devices), sorted(documented))
    assert not wrong, f"docs/trainable-kernels.md device column is stale: {wrong}"
