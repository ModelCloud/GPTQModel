# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5776: cold kernel discovery must work on the meta device.

The MACaQ E2M1 codebook is a process-wide constant, but it was created from the
ambient default device and its range was derived with data-dependent boolean
indexing at import time.  Importing the kernel registry inside
``torch.device("meta")`` therefore raised ``NotImplementedError`` from
``torch.nonzero()``, which blocked checkpoint restore and GPTQ/AWQ reload in a
fresh process.

Each scenario runs in its own subprocess: once a previous CPU import has put the
kernel modules in ``sys.modules``, the cold import path can no longer fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


pytestmark = [pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[1]

_DISCOVERY_SCRIPT = r"""
import torch
from gptqmodel.utils.importer import get_auto_backend_kernel_mapping

with torch.device("meta"):
    kernels = get_auto_backend_kernel_mapping()
    assert kernels

    import gptqmodel.nn_modules.qlinear.macaq as macaq_linear
    import gptqmodel.quantization.macaq as macaq_core

    expected = {
        "E2M1_VALUES": (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        "E2M1_MAGNITUDES": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    }
    for name, values in expected.items():
        table = getattr(macaq_core, name)
        assert table.device.type == "cpu", (name, table.device)
        assert not table.is_meta, name
        assert table.dtype == torch.float32, (name, table.dtype)
        assert table.tolist() == list(values), name
    assert macaq_core.E2M1_MAGNITUDE_INDEX.dtype == torch.uint8
    assert macaq_core.E2M1_MAGNITUDE_INDEX.tolist() == [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    assert macaq_core.E2M1_SIGN_BIT.dtype == torch.uint8
    assert macaq_core.E2M1_SIGN_BIT.tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert macaq_linear._E2M1_MIN_NONZERO == 0.5
    assert macaq_linear._E2M1_MAX == 6.0

print("META_DISCOVERY_OK", len(kernels))
"""

_BUILD_SCRIPT = r"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import torch
from tests.test_tiny_moe_quant_smoke import (
    _build_calibration_dataset,
    _build_tiny_qwen3_moe_fixture,
)

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

root = Path(sys.argv[1])
native = root / "native"
_build_tiny_qwen3_moe_fixture(native)
tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(native)
calibration = _build_calibration_dataset(tokenizer)
torch.save(torch.tensor([[1, 10, 11, 12, 13, 14, 15, 2]], dtype=torch.long), root / "input_ids.pt")

for method, fmt, name in (
    (METHOD.GPTQ, FORMAT.GPTQ, "gptq"),
    (METHOD.AWQ, FORMAT.GEMM, "awq"),
):
    config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=method,
        format=fmt,
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )
    model = GPTQModel.load(str(native), quantize_config=config, backend=BACKEND.TORCH)
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(root / name)
print("BUILT")
"""

_LOAD_SCRIPT = r"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import torch

meta_first = sys.argv[1] == "1"
if meta_first:
    from gptqmodel.utils.importer import get_auto_backend_kernel_mapping

    with torch.device("meta"):
        assert get_auto_backend_kernel_mapping()

from gptqmodel import BACKEND, GPTQModel

root = Path(sys.argv[2])
input_ids = torch.load(root / "input_ids.pt", weights_only=True)
results = {}
for name in ("gptq", "awq"):
    model = GPTQModel.load(str(root / name), backend=BACKEND.TORCH, device="cpu")
    with torch.inference_mode():
        logits = model.model(input_ids=input_ids).logits.detach().cpu().clone()
    state = {key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()}
    results[name] = {"logits": logits, "state": state}
    del model
torch.save(results, root / f"loaded-{'meta' if meta_first else 'control'}.pt")
print("LOADED", meta_first, {name: tuple(value["logits"].shape) for name, value in results.items()})
"""


def _run_python(script: str, *args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(_REPO_ROOT), env.get("PYTHONPATH")) if part)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["LOGBAR_ANIMATION"] = "0"
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_ok(completed: subprocess.CompletedProcess, marker: str) -> None:
    assert completed.returncode == 0, (
        f"subprocess failed (exit {completed.returncode}):\n" + completed.stdout[-8000:] + completed.stderr[-8000:]
    )
    assert marker in completed.stdout, completed.stdout[-8000:]


@pytest.fixture(scope="module")
def tiny_moe_checkpoints(tmp_path_factory) -> Path:
    """Quantize tiny GPTQ and AWQ Qwen3-MoE checkpoints on CPU in a fresh process."""

    root = tmp_path_factory.mktemp("meta-macaq-5776")
    _assert_ok(_run_python(_BUILD_SCRIPT, str(root)), "BUILT")
    return root


def test_cold_kernel_discovery_on_meta_keeps_macaq_codebook_real():
    """The E2M1 codebook is a CPU constant even when discovery runs on meta."""

    _assert_ok(_run_python(_DISCOVERY_SCRIPT), "META_DISCOVERY_OK")


def test_gptq_and_awq_reload_after_meta_discovery_matches_cpu_control(tiny_moe_checkpoints: Path):
    """Cold discovery on meta cannot change restored tensors or logits."""

    _assert_ok(_run_python(_LOAD_SCRIPT, "1", str(tiny_moe_checkpoints)), "LOADED True")
    _assert_ok(_run_python(_LOAD_SCRIPT, "0", str(tiny_moe_checkpoints)), "LOADED False")

    meta_results = torch.load(tiny_moe_checkpoints / "loaded-meta.pt", weights_only=True)
    control_results = torch.load(tiny_moe_checkpoints / "loaded-control.pt", weights_only=True)
    assert set(meta_results) == set(control_results) == {"gptq", "awq"}

    for name in ("gptq", "awq"):
        actual = meta_results[name]
        expected = control_results[name]
        assert torch.equal(actual["logits"], expected["logits"]), name
        assert set(actual["state"]) == set(expected["state"]), name
        mismatched = [
            key for key in expected["state"] if not torch.equal(actual["state"][key], expected["state"][key])
        ]
        assert not mismatched, (name, mismatched[:5])
