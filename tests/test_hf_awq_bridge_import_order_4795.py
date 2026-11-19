# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Import-order regression test for the Transformers AWQ load bridge (#4795).

``transformers.integrations.awq.replace_with_awq_linear`` imports ``gptqmodel``
from inside its own body, i.e. while ``AwqQuantizer`` is already running. The
package-init attempt therefore installs the AWQ bridge too late for the load
that triggered the import: transformers builds every module from the *global*
``bits``/``group_size`` and the checkpoint's ``dynamic`` per-module overrides are
ignored. On CPU the checkpoint then decodes with the wrong group size (silently
wrong logits) and on CUDA the Marlin pack raises ``AWQ qzeros shape ...``.

The load runs in a *fresh interpreter that never imports gptqmodel first*, which
is exactly what ``AutoModelForCausalLM.from_pretrained`` does in
``docs/peft.md``. It must match ``GPTQModel.load``: same kernel class, same
per-module ``group_size``, same qzeros shape, and the same NLL.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import AWQConfig


_HIDDEN = 128
_INTERMEDIATE = 256
_VOCAB = 128
_LAYERS = 3
_GROUP_SIZE = 128
# Layer 1 carries the per-module override: 128 / 32 = 4 zero-point groups,
# against a single group for the global group size.
_DYNAMIC = {r"+:.*layers\.1\..*": {"group_size": 32}}


# Runs in a separate process. Do not import gptqmodel here: the whole point is
# that Transformers imports it from inside its AWQ loader.
_PLAIN_LOAD_SCRIPT = r"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

checkpoint = sys.argv[1]
device = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device, dtype=torch.float16)

inner = model
while not hasattr(inner, "layers") and hasattr(inner, "model"):
    inner = inner.model

layers = {}
for index in (0, 1):
    module = inner.layers[index].self_attn.q_proj
    layers[index] = {
        "type": type(module).__name__,
        "group_size": int(module.group_size),
        "qzeros": list(module.qzeros.shape),
    }

generator = torch.Generator().manual_seed(11)
ids = torch.randint(0, 128, (1, 32), generator=generator).to(next(model.parameters()).device)
with torch.inference_mode():
    loss = float(model(input_ids=ids, labels=ids).loss)

from transformers.quantizers.quantizer_awq import AwqQuantizer

print(
    "BRIDGE_RESULT "
    + json.dumps(
        {
            "bridge": bool(
                getattr(
                    AwqQuantizer._process_model_after_weight_loading,
                    "_gptqmodel_load_bridge",
                    False,
                )
            ),
            "gptqmodel_imported": "gptqmodel" in sys.modules,
            "layers": layers,
            "loss": loss,
        }
    )
)
"""


def _build_awq_checkpoint(tmp_path, dynamic):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    source = tmp_path / "source"
    checkpoint = tmp_path / ("awq-dynamic" if dynamic else "awq-static")

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).to(torch.float16).save_pretrained(source)

    generator = torch.Generator().manual_seed(1)
    calibration = [
        {"input_ids": torch.randint(0, _VOCAB, (1, 32), generator=generator)}
        for _ in range(8)
    ]

    model = GPTQModel.load(
        str(source),
        AWQConfig(
            bits=4,
            group_size=_GROUP_SIZE,
            dynamic=dynamic,
            offload_to_disk=False,
        ),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(checkpoint))

    manifest = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
    assert manifest["quant_method"] == "awq"
    assert manifest["group_size"] == _GROUP_SIZE
    assert manifest.get("dynamic") == dynamic
    return checkpoint


def _build_dynamic_awq_checkpoint(tmp_path):
    return _build_awq_checkpoint(tmp_path, _DYNAMIC)


def _build_static_awq_checkpoint(tmp_path):
    """No per-module overrides: this load takes the bridge's no-context branch (#5511)."""

    return _build_awq_checkpoint(tmp_path, None)


def _load_in_fresh_interpreter(checkpoint, device):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH")) if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", _PLAIN_LOAD_SCRIPT, str(checkpoint), device],
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


@pytest.fixture(scope="module")
def awq_dynamic_checkpoint(tmp_path_factory):
    return _build_dynamic_awq_checkpoint(tmp_path_factory.mktemp("awq_dynamic_4795"))


def _native_expectation(checkpoint, device):
    native = GPTQModel.load(str(checkpoint), device=device, dtype=torch.float16)
    inner = native.model
    layers = {}
    for index in (0, 1):
        module = inner.model.layers[index].self_attn.q_proj
        layers[index] = {
            "type": type(module).__name__,
            "group_size": int(module.group_size),
            "qzeros": list(module.qzeros.shape),
        }
    generator = torch.Generator().manual_seed(11)
    ids = torch.randint(0, 128, (1, 32), generator=generator).to(
        next(native.parameters()).device
    )
    with torch.inference_mode():
        loss = float(native(input_ids=ids, labels=ids).loss)
    return layers, loss


def _assert_matches_native(checkpoint, device):
    expected_layers, expected_loss = _native_expectation(checkpoint, device)

    # The checkpoint really is dynamic: layer 1 stores 4x the zero-point groups.
    assert expected_layers[0]["group_size"] == _GROUP_SIZE
    assert expected_layers[1]["group_size"] == 32
    assert expected_layers[1]["qzeros"][0] == 4 * expected_layers[0]["qzeros"][0]

    result = _load_in_fresh_interpreter(checkpoint, device)

    assert result["gptqmodel_imported"] is True
    assert result["bridge"] is True
    for index in (0, 1):
        assert result["layers"][str(index)] == expected_layers[index], (
            f"layer {index}: plain AutoModelForCausalLM built "
            f"{result['layers'][str(index)]}, GPTQModel.load built "
            f"{expected_layers[index]}"
        )
    assert result["loss"] == pytest.approx(expected_loss, abs=1e-3)


def test_plain_transformers_load_honors_dynamic_awq_on_cpu(awq_dynamic_checkpoint):
    """No ``import gptqmodel`` first: CPU must not decode at the global group size."""

    _assert_matches_native(awq_dynamic_checkpoint, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_plain_transformers_load_honors_dynamic_awq_on_cuda(awq_dynamic_checkpoint):
    """Without the fix this raises ``AWQ qzeros shape must be ...``."""

    _assert_matches_native(awq_dynamic_checkpoint, "cuda")


def test_awq_bridge_install_is_idempotent():
    from transformers.quantizers.quantizer_awq import AwqQuantizer

    import gptqmodel

    assert gptqmodel._ensure_transformers_awq_gptqmodel_load_bridge() is True
    assert gptqmodel._ensure_transformers_awq_gptqmodel_load_bridge() is True
    assert getattr(
        AwqQuantizer._process_model_before_weight_loading,
        "_gptqmodel_load_bridge",
        False,
    )
    assert getattr(
        AwqQuantizer._process_model_after_weight_loading,
        "_gptqmodel_load_bridge",
        False,
    )


# -- #5511: the no-context branch must not run the native post-init twice ----
#
# A checkpoint without ``dynamic`` overrides needs no module rebuild, so the
# bridge runs ``hf_gptqmodel_finish_awq_load_without_rebuild`` - which already
# calls ``gptqmodel_post_init`` - and then used to fall through to
# ``original_post_init``, i.e. ``hf_gptqmodel_post_init`` ->
# ``gptqmodel_post_init`` again. On CPU the AWQ kernel tolerates the second
# call; on CUDA ``AwqMarlinLinear`` re-pads an already repacked ``qweight`` and
# the plain-transformers load of *any* GPT-QModel AWQ checkpoint raises.


def _drive_no_context_branch(monkeypatch, native_qcfg):
    """Call the bridge post-init hook with no load context, recording calls."""

    from transformers.quantizers.quantizer_awq import AwqQuantizer

    import gptqmodel.utils.hf_load as hf_load
    import gptqmodel.utils.model as model_utils

    native_calls = []
    real_post_init = model_utils.gptqmodel_post_init

    def counting_post_init(*args, **kwargs):
        native_calls.append(kwargs.get("quantize_config"))
        return real_post_init(*args, **kwargs)

    monkeypatch.setattr(model_utils, "gptqmodel_post_init", counting_post_init)
    monkeypatch.setattr(hf_load, "hf_gptqmodel_repair_awq_load", lambda *a, **k: None)
    monkeypatch.setattr(
        hf_load,
        "hf_gptqmodel_native_awq_config",
        lambda *a, **k: native_qcfg,
    )
    without_rebuild = []
    monkeypatch.setattr(
        hf_load,
        "hf_gptqmodel_finish_awq_load_without_rebuild",
        lambda model, **kwargs: without_rebuild.append(kwargs.get("qcfg")),
    )

    model = torch.nn.Module()
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    quantizer = SimpleNamespace(
        quantization_config=SimpleNamespace(desc_act=False),
    )
    AwqQuantizer._process_model_after_weight_loading(quantizer, model)
    return without_rebuild, native_calls


def test_static_native_checkpoint_runs_post_init_once(monkeypatch):
    """The native checkpoint got its post-init from the helper; do not repeat it."""

    from gptqmodel.quantization import QuantizeConfig

    qcfg = QuantizeConfig(bits=4, group_size=_GROUP_SIZE)
    without_rebuild, native_calls = _drive_no_context_branch(monkeypatch, qcfg)

    assert without_rebuild == [qcfg]
    assert native_calls == []


def test_third_party_checkpoint_still_runs_transformers_post_init(monkeypatch):
    """A checkpoint that is not ours has no native post-init yet, so keep it."""

    without_rebuild, native_calls = _drive_no_context_branch(monkeypatch, None)

    assert without_rebuild == [None]
    assert native_calls == [None]


@pytest.fixture(scope="module")
def awq_static_checkpoint(tmp_path_factory):
    return _build_static_awq_checkpoint(tmp_path_factory.mktemp("awq_static_5511"))


def _assert_static_matches_native(checkpoint, device):
    expected_layers, expected_loss = _native_expectation(checkpoint, device)
    result = _load_in_fresh_interpreter(checkpoint, device)

    assert result["gptqmodel_imported"] is True
    assert result["bridge"] is True
    # Unlike the ``dynamic`` case the bridge does not rebuild the modules here,
    # so Transformers keeps its own kernel choice (``TorchCpuAvx512AwqLinear``
    # on CPU against ``GPTQModel.load``'s ``TorchAtenAwqLinear``). The stored
    # layout and the output are what must match.
    for index in (0, 1):
        got, expected = result["layers"][str(index)], expected_layers[index]
        assert got["group_size"] == expected["group_size"]
        assert got["qzeros"] == expected["qzeros"]
    assert result["loss"] == pytest.approx(expected_loss, abs=1e-3)


def test_plain_transformers_load_static_awq_on_cpu(awq_static_checkpoint):
    _assert_static_matches_native(awq_static_checkpoint, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_plain_transformers_load_static_awq_on_cuda(awq_static_checkpoint):
    """Without the #5511 fix this raises ``AWQ qweight shape must be ...``."""

    _assert_static_matches_native(awq_static_checkpoint, "cuda")
