# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The plain-Transformers AWQ repair must inherit the model's module mode (#4907).

``hf_gptqmodel_repair_awq_load`` rebuilds a ``dynamic`` AWQ checkpoint's modules
in the post-weight hook (#4795). A freshly constructed ``nn.Module`` is in
training mode, and the rebuild runs after Transformers put the model in eval, so
every repaired linear stayed ``training=True``. The CPU int4pack kernels choose
their layout from ``self.training`` on the first forward and latch it, so the
model then dequantized every weight on every forward, and the latched ``"train"``
mode was sticky across a later ``model.eval()``.

The load must run in a fresh interpreter: ``import gptqmodel`` in this process
would install the AWQ bridge before the load and take the native path instead of
the repair path being tested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_hf_awq_bridge_import_order_4795 import _build_dynamic_awq_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MARKER = "AWQ_REPAIR_RESULT "

# Runs in a separate process. Do not import gptqmodel before the load: the whole
# point is that Transformers imports it from inside its AWQ loader, which leaves
# the repair to the post-weight hook.
_PROBE = r"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

checkpoint = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cpu", dtype=torch.float16)

from gptqmodel.nn_modules.qlinear import BaseQuantLinear

modules = [module for module in model.modules() if isinstance(module, BaseQuantLinear)]
vocab = int(model.config.vocab_size)
ids = torch.randint(0, vocab, (1, 32), generator=torch.Generator().manual_seed(7))
with torch.inference_mode():
    plain_logits = model(ids).logits.clone()

plain = {
    "model_training": bool(model.training),
    "training_modules": int(sum(module.training for module in modules)),
    "n": len(modules),
    "modes": sorted({str(module.linear_mode) for module in modules}),
}

# A forward taken in training mode latches the dense-dequant path; `eval()`
# must let the next forward choose the int4pack layout again.
model.train()
with torch.inference_mode():
    model(ids)
latched = sorted({str(module.linear_mode) for module in modules})
model.eval()
with torch.inference_mode():
    model(ids)
recovered = sorted({str(module.linear_mode) for module in modules})

# The native path (gptqmodel imported before the load) for the bitwise check.
from gptqmodel import GPTQModel

native = GPTQModel.load(checkpoint, device="cpu", dtype=torch.float16)
native_modules = [module for module in native.model.modules() if isinstance(module, BaseQuantLinear)]
with torch.inference_mode():
    native_logits = native.model(ids).logits.clone()

print(
    "AWQ_REPAIR_RESULT "
    + json.dumps(
        {
            "plain": plain,
            "latched": latched,
            "recovered": recovered,
            "recovered_training_modules": int(sum(module.training for module in modules)),
            "native_modes": sorted({str(module.linear_mode) for module in native_modules}),
            "bitwise_equal": bool(torch.equal(plain_logits, native_logits)),
            "max_abs_delta": float((plain_logits - native_logits).abs().max()),
        }
    )
)
"""


@pytest.fixture(scope="module")
def awq_dynamic_checkpoint(tmp_path_factory):
    return _build_dynamic_awq_checkpoint(tmp_path_factory.mktemp("awq_repair_mode_4907"))


def _run_probe(checkpoint) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_REPO_ROOT), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(checkpoint)],
        capture_output=True,
        cwd=str(checkpoint.parent),
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_MARKER)]
    assert lines, completed.stdout + completed.stderr
    return json.loads(lines[-1][len(_MARKER) :])


@pytest.fixture(scope="module")
def repair_probe(awq_dynamic_checkpoint):
    return _run_probe(awq_dynamic_checkpoint)


def test_plain_load_leaves_quant_modules_in_the_model_mode(repair_probe):
    """A repaired module must not outlive the load in training mode."""

    plain = repair_probe["plain"]

    assert plain["n"] > 0
    assert plain["model_training"] is False
    assert plain["training_modules"] == 0, (
        f"{plain['training_modules']}/{plain['n']} repaired quant modules are still in "
        "training mode after a plain from_pretrained"
    )


def test_inference_forward_reaches_the_int4pack_layout(repair_probe):
    """The first inference forward must pick the fused layout, not dense dequant."""

    assert repair_probe["plain"]["modes"] == ["inference"]


def test_eval_recovers_a_module_that_latched_the_train_path(repair_probe):
    """The latched train mode must not survive `eval()`."""

    assert repair_probe["latched"] == ["train"]
    assert repair_probe["recovered"] == ["inference"], (
        "a module that latched linear_mode='train' stayed on the dense-dequant path "
        "after model.eval()"
    )
    assert repair_probe["recovered_training_modules"] == 0


def test_repaired_forward_matches_the_native_path(repair_probe):
    """The repair's logits stay bit-identical to `import gptqmodel`-first."""

    assert repair_probe["native_modes"] == ["inference"]
    assert repair_probe["bitwise_equal"] is True, (
        f"plain-transformers load differs from the native load by "
        f"{repair_probe['max_abs_delta']:.6g}"
    )
    assert repair_probe["max_abs_delta"] == 0.0
