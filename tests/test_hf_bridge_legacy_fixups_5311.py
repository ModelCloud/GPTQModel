# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #5311: the Transformers bridge must run the legacy remote-code repairs.

``BaseQModel.__init__`` restores rotary buffers that transformers 5 materializes
with ``torch.empty_like`` (a remote-code class that overrides ``_init_weights``
never re-derives ``inv_freq``), and installs the transformers-4 ``generate()``
adapter for remote ``prepare_inputs_for_generation`` implementations. The bridge
(``hf_gptqmodel_post_init_for_load``, used by
``AutoModelForCausalLM.from_pretrained`` and ``pipeline()``) mirrored the other
post-load steps but skipped these two, so a checkpoint that ``GPTQModel.load``
reads at PPL 13 produced garbage RoPE frequencies (PPL 478) through transformers.

Both repairs now live in ``apply_legacy_remote_code_fixups`` and both load paths
call it, so a new repair cannot be added to one path only.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gptqmodel.utils.hf as hf_utils
import gptqmodel.utils.structure as structure_utils
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.hf_load import (
    HFGPTQModelLoadContext,
    hf_gptqmodel_post_init_for_load,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cpu]


class _LegacyRotary(nn.Module):
    """Remote-code rotary module: ``inv_freq`` is a non-persistent buffer."""

    def __init__(self, dim: int, max_position_embeddings: int = 32, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class _LegacyRemoteModel(nn.Module):
    """Model whose ``_init_weights`` override never re-derives the rotary buffer."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim * 2, num_attention_heads=2)
        self.rotary_emb = _LegacyRotary(dim)

    def _init_weights(self, module):  # remote-code override
        pass


def _bridge_model() -> nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    model.hf_device_map = {"": "cpu"}
    return model


def _context() -> HFGPTQModelLoadContext:
    return HFGPTQModelLoadContext(
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        quant_linear=TorchLinear,
    )


def _stub_native_post_init(monkeypatch) -> None:
    """Skip the kernel-building steps; only the fix-up wiring is under test."""

    import gptqmodel.utils.model as model_utils

    monkeypatch.setattr(
        model_utils,
        "hf_convert_gptq_v1_to_v2_format",
        lambda model, **kwargs: (model, False),
    )
    monkeypatch.setattr(model_utils, "gptqmodel_post_init", lambda model, **kwargs: model)


def test_bridge_runs_the_legacy_remote_code_fixes(monkeypatch):
    _stub_native_post_init(monkeypatch)
    calls = []
    monkeypatch.setattr(
        structure_utils,
        "restore_init_owned_rotary_buffers",
        lambda model: calls.append(("rotary", model)) or 2,
    )
    monkeypatch.setattr(
        hf_utils,
        "install_legacy_generation_compat",
        lambda model: calls.append(("generation", model)) or True,
    )

    model = _bridge_model()

    assert hf_gptqmodel_post_init_for_load(model, _context()) is model
    assert [name for name, _target in calls] == ["rotary", "generation"]
    assert all(target is model for _name, target in calls)


def test_helper_restores_a_corrupted_rotary_buffer(monkeypatch):
    monkeypatch.setattr(hf_utils, "install_legacy_generation_compat", lambda model: False)
    model = _LegacyRemoteModel(dim=8)
    expected = model.rotary_emb.inv_freq.clone()
    # transformers 5 materializes checkpoint-absent non-persistent buffers with
    # `torch.empty_like`; the contents are whatever was in memory.
    model.rotary_emb.inv_freq.fill_(float("nan"))

    restored, installed = hf_utils.apply_legacy_remote_code_fixups(model)

    assert restored == 1
    assert installed is False
    assert torch.equal(model.rotary_emb.inv_freq, expected)


def test_helper_reports_the_generation_adapter(monkeypatch):
    monkeypatch.setattr(structure_utils, "restore_init_owned_rotary_buffers", lambda model: 0)
    monkeypatch.setattr(hf_utils, "install_legacy_generation_compat", lambda model: True)

    assert hf_utils.apply_legacy_remote_code_fixups(_LegacyRemoteModel()) == (0, True)


def test_both_load_paths_call_the_shared_helper():
    for relative in ("gptqmodel/models/base.py", "gptqmodel/utils/hf_load.py"):
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "apply_legacy_remote_code_fixups"
        ]
        assert calls, f"{relative} does not run the shared post-load fix-ups (#5311)"


# ---------------------------------------------------------------------------
# #5433: the AWQ bridge takes the same path, but only when a load context
# exists. A native AWQ checkpoint without `dynamic` needs no module rebuild, so
# the repair returned None and the post-init (and these fix-ups) never ran.
# ---------------------------------------------------------------------------

# A tiny remote-code decoder: Qwen2's layout with a legacy rotary module and a
# legacy `prepare_inputs_for_generation`, plus the `_init_weights` override that
# stops transformers from re-deriving `inv_freq`.
_REMOTE_CONFIG_PY = '''
from transformers import Qwen2Config


class ProbeConfig(Qwen2Config):
    model_type = "probe_legacy"
'''

_REMOTE_MODELING_PY = '''
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RotaryEmbedding

try:
    from .configuration_probe import ProbeConfig
except ImportError:  # direct import while the fixture is built
    from configuration_probe import ProbeConfig


class ProbeRotary(Qwen2RotaryEmbedding):
    """Legacy rotary: no ``original_inv_freq`` for transformers to re-derive."""

    def __init__(self, config, device=None):
        super().__init__(config)
        if "original_inv_freq" in self._buffers:
            del self._buffers["original_inv_freq"]


class ProbeForCausalLM(Qwen2ForCausalLM):
    config_class = ProbeConfig

    def __init__(self, config):
        super().__init__(config)
        self.model.rotary_emb = ProbeRotary(config)

    def _init_weights(self, module):
        # Remote-code override: transformers' default restore never runs.
        pass

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # transformers-4 contract: `past_key_values` is None until the first
        # forward wrote the cache, and this code slices the new tokens itself.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
'''

_AWQ_CONFIG = {
    "architectures": ["ProbeForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_probe.ProbeConfig",
        "AutoModelForCausalLM": "modeling_probe.ProbeForCausalLM",
    },
    "model_type": "probe_legacy",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    # AVX512 CPU kernels need every out_features divisible by 32; 4 KV heads
    # of head_dim 8 keeps the KV projections at 32.
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "torch_dtype": "float16",
}

_AWQ_GROUP_SIZE = 16
# `probe_legacy` has no registered definition, so the fixture names the same
# Llama/Qwen2 decoder layout the tiny model uses.
_AWQ_MODULE_TREE = [
    "model",
    "layers",
    "#",
    {
        "input_layernorm": ("input_layernorm:!:norm",),
        "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o"),
        "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
        "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
    },
]

# Runs in a fresh interpreter that does not import gptqmodel: transformers
# imports it from inside `replace_with_awq_linear`, i.e. after the prepare hook
# could run, which is the import order #5433 is about.
_PLAIN_AWQ_LOAD_SCRIPT = r"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM

checkpoint = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map="cpu", dtype=torch.float16, trust_remote_code=True
)
rotary = model.model.rotary_emb
generator = torch.Generator().manual_seed(7)
ids = torch.randint(2, 128, (1, 8), generator=generator)
with torch.inference_mode():
    logits = model(ids).logits
    generated = model.generate(ids, max_new_tokens=4, do_sample=False)

print(
    "AWQ_RESULT "
    + json.dumps(
        {
            "inv_freq": rotary.inv_freq.tolist(),
            "inv_freq_sum": float(rotary.inv_freq.sum()),
            "logits": logits.flatten()[:8].tolist(),
            "generated": generated.tolist(),
            "legacy_generation_compat": bool(
                getattr(type(model), "_gptqmodel_legacy_generation_compat", False)
            ),
            "has_quantize_config": hasattr(model, "quantize_config"),
            "gptqmodel_imported": "gptqmodel" in sys.modules,
        }
    )
)
"""


def _write_remote_code_model(directory: Path) -> None:
    import json

    from tokenizers import Tokenizer, models, pre_tokenizers

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "configuration_probe.py").write_text(_REMOTE_CONFIG_PY, encoding="utf-8")
    (directory / "modeling_probe.py").write_text(_REMOTE_MODELING_PY, encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(_AWQ_CONFIG, indent=2), encoding="utf-8")

    tokenizer = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(128)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "t0",
                "pad_token": "t0",
                "eos_token": "t1",
                "bos_token": "t1",
                "model_max_length": 64,
            }
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "t1", "eos_token": "t1", "pad_token": "t0"}),
        encoding="utf-8",
    )


def _build_legacy_awq_checkpoint(tmp_path: Path) -> Path:
    """A tiny remote-code decoder quantized with GPT-QModel's own AWQ."""

    import json
    import sys

    from transformers import AutoConfig
    from transformers.modeling_utils import PreTrainedModel

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import AWQConfig

    source = tmp_path / "source"
    checkpoint = tmp_path / "awq"
    _write_remote_code_model(source)

    sys.path.insert(0, str(source))
    try:
        from modeling_probe import ProbeForCausalLM

        config = AutoConfig.from_pretrained(str(source), trust_remote_code=True)

        class _Builder(ProbeForCausalLM):
            # Build real values once; the reloaded class keeps the legacy
            # no-op override, which is what the load under test exercises.
            _init_weights = PreTrainedModel._init_weights

        torch.manual_seed(0)
        _Builder(config).save_pretrained(str(source))
    finally:
        sys.path.remove(str(source))

    model = GPTQModel.load(
        str(source),
        AWQConfig(bits=4, group_size=_AWQ_GROUP_SIZE, offload_to_disk=False),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
        module_tree=_AWQ_MODULE_TREE,
    )
    generator = torch.Generator().manual_seed(1)
    calibration = [
        {"input_ids": torch.randint(2, 128, (1, 16), generator=generator)}
        for _ in range(4)
    ]
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(checkpoint))

    manifest = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
    assert manifest["quant_method"] == "awq"
    # The checkpoint really is a plain (non-`dynamic`) AWQ manifest: this is the
    # shape whose repair returns None and used to skip the native post-init.
    assert not manifest.get("dynamic")
    return checkpoint


def _load_awq_without_importing_gptqmodel(checkpoint: Path) -> dict:
    import json
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PLAIN_AWQ_LOAD_SCRIPT, str(checkpoint)],
        capture_output=True,
        cwd=str(checkpoint.parent),
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = "AWQ_RESULT "
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert lines, completed.stdout + completed.stderr
    return json.loads(lines[-1][len(marker):])


def _native_awq_expectation(checkpoint: Path) -> dict:
    from gptqmodel import GPTQModel

    native = GPTQModel.load(
        str(checkpoint),
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
        module_tree=_AWQ_MODULE_TREE,
    )
    inner = native.model
    rotary = inner.model.rotary_emb
    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(2, 128, (1, 8), generator=generator)
    with torch.inference_mode():
        logits = inner(ids).logits
        generated = inner.generate(ids, max_new_tokens=4, do_sample=False)
    return {
        "inv_freq": rotary.inv_freq.tolist(),
        "logits": logits.flatten()[:8].tolist(),
        "generated": generated.tolist(),
    }


@pytest.fixture(scope="module")
def legacy_awq_checkpoint(tmp_path_factory):
    return _build_legacy_awq_checkpoint(tmp_path_factory.mktemp("awq_legacy_5433"))


def test_plain_transformers_awq_load_runs_the_legacy_fixups(legacy_awq_checkpoint):
    """#5433: a native AWQ load with nothing to rebuild still needs the fix-ups."""

    plain = _load_awq_without_importing_gptqmodel(legacy_awq_checkpoint)
    native = _native_awq_expectation(legacy_awq_checkpoint)

    # transformers imported gptqmodel mid-load, which is the import order this
    # regression is about (the prepare hook never ran for this load).
    assert plain["gptqmodel_imported"] is True

    # The RoPE frequencies are restored instead of keeping the zeros
    # `torch.empty_like` materialized, and the forward matches the native load.
    assert plain["inv_freq_sum"] > 0
    assert plain["inv_freq"] == pytest.approx(native["inv_freq"], rel=1e-6)
    assert plain["logits"] == pytest.approx(native["logits"], rel=1e-3, abs=1e-3)

    # The transformers-4 `generate()` adapter is installed and generation
    # matches the native load instead of pre-filling on the last token only.
    assert plain["legacy_generation_compat"] is True
    assert plain["generated"] == native["generated"]
    # The native post-init also attaches the serialized config for `save()`.
    assert plain["has_quantize_config"] is True


def _stub_awq_finish(monkeypatch, calls):
    import gptqmodel.utils.importer as importer_utils
    import gptqmodel.utils.model as model_utils
    from gptqmodel.models import loader as loader_module

    monkeypatch.setattr(
        hf_utils,
        "apply_legacy_remote_code_fixups",
        lambda model: calls.append("fixups") or (0, False),
    )
    monkeypatch.setattr(
        model_utils,
        "gptqmodel_post_init",
        lambda model, **kwargs: calls.append("post_init") or model,
    )
    monkeypatch.setattr(
        loader_module,
        "_setup_rotation_online_had",
        lambda model, rotation, *args, **kwargs: calls.append("rotation"),
    )
    monkeypatch.setattr(
        loader_module,
        "_set_paged_attention_safe_cuda_graphs",
        lambda model: calls.append("paged"),
    )
    monkeypatch.setattr(
        importer_utils,
        "validate_device_map_capability_groups",
        lambda mapping: calls.append("device_map"),
    )


def test_awq_finish_without_rebuild_runs_the_shared_helper(monkeypatch):
    """#5433: a native AWQ load with nothing to rebuild gets the full finish."""

    from gptqmodel.utils.hf_load import hf_gptqmodel_finish_awq_load_without_rebuild

    calls = []
    _stub_awq_finish(monkeypatch, calls)
    model = _bridge_model()

    hf_gptqmodel_finish_awq_load_without_rebuild(
        model, qcfg=QuantizeConfig(bits=4, group_size=128)
    )

    assert calls == ["post_init", "rotation", "paged", "device_map", "fixups"]
    assert model.quantize_config.bits == 4


def test_awq_finish_without_rebuild_leaves_third_party_modules_alone(monkeypatch):
    """The prepare fallback's third-party checkpoint still needs the fix-ups."""

    from gptqmodel.utils.hf_load import hf_gptqmodel_finish_awq_load_without_rebuild

    calls = []
    _stub_awq_finish(monkeypatch, calls)
    model = _bridge_model()

    hf_gptqmodel_finish_awq_load_without_rebuild(model, qcfg=None)

    assert calls == ["fixups"]
    assert not hasattr(model, "quantize_config")
