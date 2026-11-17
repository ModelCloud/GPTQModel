# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Import-order regression test for the Optimum GPTQ load bridge (#4749).

``optimum.gptq`` imports ``gptqmodel`` while it is still executing. The package
init probe therefore observes a partially initialized module and, before the
fix, cached that ``ImportError`` as "Optimum is not installed". The bridge was
never installed in the plain-Transformers flow, so a mixed-bit ``dynamic``
checkpoint kept its 8-bit ``qweight`` inside a 4-bit module and produced wrong
logits with no warning.

The load runs in a *fresh interpreter that never imports gptqmodel first*,
which is exactly what ``AutoModelForCausalLM.from_pretrained`` does in
``docs/peft.md``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("optimum")

from optimum.utils import is_gptqmodel_available


pytestmark = pytest.mark.skipif(
    not is_gptqmodel_available(),
    reason="optimum cannot see the gptqmodel package metadata",
)


# Runs in a separate process. Do not import gptqmodel here: the whole point is
# that Transformers/Optimum import it while Optimum itself is initializing.
_PLAIN_LOAD_SCRIPT = r"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

checkpoint = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="cpu", dtype=torch.float16)

from optimum.gptq import GPTQQuantizer

layers = {}
for index in (0, 1):
    module = model.model.layers[index].self_attn.q_proj
    layers[index] = {
        "type": type(module).__name__,
        "bits": getattr(module, "bits", None),
        "shape": list(module.qweight.shape),
    }

print(
    "BRIDGE_RESULT "
    + json.dumps(
        {
            "bridge": bool(getattr(GPTQQuantizer.convert_model, "_gptqmodel_load_bridge", False)),
            "gptqmodel_imported": "gptqmodel" in sys.modules,
            "layers": layers,
        }
    )
)
"""


def _build_mixed_bit_checkpoint(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import RTNConfig

    source = tmp_path / "source"
    checkpoint = tmp_path / "mixed-bit"

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).to(torch.float16).save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        RTNConfig(
            bits=4,
            group_size=32,
            dynamic={r"+:.*layers\.1\..*": {"bits": 8}},
        ),
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize()
    model.save(str(checkpoint))
    return checkpoint


def _load_in_fresh_interpreter(checkpoint):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH")) if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", _PLAIN_LOAD_SCRIPT, str(checkpoint)],
        capture_output=True,
        cwd=str(checkpoint.parent),
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    marker = "BRIDGE_RESULT "
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert lines, completed.stdout + completed.stderr
    return json.loads(lines[-1][len(marker) :])


def test_plain_transformers_load_installs_bridge_and_keeps_dynamic_bits(tmp_path):
    """No ``import gptqmodel`` first: the bridge must still be installed."""

    checkpoint = _build_mixed_bit_checkpoint(tmp_path)

    from gptqmodel import GPTQModel

    native = GPTQModel.load(str(checkpoint), device="cpu", dtype=torch.float16)
    expected = {}
    for index in (0, 1):
        module = native.model.model.layers[index].self_attn.q_proj
        expected[index] = {
            "bits": int(module.bits),
            "shape": list(module.qweight.shape),
        }

    # The checkpoint really is mixed-bit: layer 1 carries the 8-bit payload.
    assert expected[0]["bits"] == 4
    assert expected[1]["bits"] == 8
    # 8-bit packs twice as many weights per int32 row as 4-bit.
    assert expected[1]["shape"][0] == 2 * expected[0]["shape"][0]

    result = _load_in_fresh_interpreter(checkpoint)

    assert result["gptqmodel_imported"] is True
    assert result["bridge"] is True
    for index in (0, 1):
        assert result["layers"][str(index)]["bits"] == expected[index]["bits"]
        assert result["layers"][str(index)]["shape"] == expected[index]["shape"]
