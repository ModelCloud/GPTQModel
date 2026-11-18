# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel targets Transformers 5.23+ and never patches the HF ecosystem.

Importing ``gptqmodel``, loading a native model, building a package-owned port
and loading a GPT-QModel quantized checkpoint must leave Transformers' cache,
generation, RoPE, import-utils, quantizer registry, GGUF integration and
``PreTrainedModel`` APIs, plus ``torch.nn.init`` and ``safetensors.torch``,
exactly as they ship. There is no allow-list: GPT-QModel installs no
integration hook either. The check runs in a fresh interpreter so no other
test can have imported ``gptqmodel`` first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every entry here would be a patch GPT-QModel applies to a third-party API.
# The policy forbids them all, so the set stays empty.
_ALLOWED_INTEGRATION_PATCHES: set[tuple[str, str]] = set()

_SCRIPT = textwrap.dedent(
    r"""
    import json
    import sys

    import importlib.util

    import safetensors.torch as safetensors_torch
    import torch
    import torch.nn.init as torch_init
    import transformers
    import transformers.cache_utils as cache_utils
    import transformers.integrations.ggml as hf_ggml
    import transformers.modeling_gguf_pytorch_utils as hf_gguf_utils
    import transformers.quantizers.auto as quant_auto
    import transformers.generation.utils as generation_utils
    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils as hf_utils
    import transformers.utils.generic as generic
    import transformers.utils.import_utils as import_utils
    from transformers import PreTrainedModel, PreTrainedConfig
    from transformers.generation import GenerationMixin
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_awq import AwqQuantizer
    from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

    WATCHED = {
        "cache_utils": cache_utils,
        "cache_utils.Cache": cache_utils.Cache,
        "cache_utils.DynamicCache": cache_utils.DynamicCache,
        "generation_utils": generation_utils,
        "GenerationMixin": GenerationMixin,
        "rope_utils.ROPE_INIT_FUNCTIONS": rope_utils.ROPE_INIT_FUNCTIONS,
        "transformers.utils": hf_utils,
        "generic": generic,
        "import_utils": import_utils,
        "PreTrainedModel": PreTrainedModel,
        "PreTrainedConfig": PreTrainedConfig,
        "hub_kernels": hub_kernels,
        "quant_auto": quant_auto,
        "quant_auto.AUTO_QUANTIZER_MAPPING": quant_auto.AUTO_QUANTIZER_MAPPING,
        "quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING": quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING,
        "AutoHfQuantizer": quant_auto.AutoHfQuantizer,
        "AutoQuantizationConfig": quant_auto.AutoQuantizationConfig,
        "GptqHfQuantizer": GptqHfQuantizer,
        "AwqQuantizer": AwqQuantizer,
        "modeling_gguf_pytorch_utils": hf_gguf_utils,
        "integrations.ggml": hf_ggml,
        "torch.nn.init": torch_init,
        "safetensors.torch": safetensors_torch,
    }
    # Optional HF-ecosystem loaders GPT-QModel used to bridge into.
    for _label, _name in (("optimum.gptq", "optimum.gptq"), ("openvino.gptq", "openvino.frontend.pytorch.gptq")):
        try:
            if importlib.util.find_spec(_name) is not None:
                import importlib

                _module = importlib.import_module(_name)
                WATCHED[_label] = _module
                if hasattr(_module, "GPTQQuantizer"):
                    WATCHED[_label + ".GPTQQuantizer"] = _module.GPTQQuantizer
        except Exception:
            pass

    import types

    def snapshot():
        state = {}
        for label, obj in WATCHED.items():
            items = obj if isinstance(obj, dict) else vars(obj)
            state[label] = {
                name: id(value)
                for name, value in list(items.items())
                # Lazily created dunders (PEP 649 annotations) and submodules
                # imported on first use are not patches.
                if not (name.startswith("__") and name.endswith("__"))
                and not isinstance(value, types.ModuleType)
            }
        return state

    before = snapshot()

    import gptqmodel  # noqa: F401
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    after_import = snapshot()

    model_dir = sys.argv[1]
    loaded = GPTQModel.load(
        model_dir, device="cpu", dtype=torch.float32,
        quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False),
    )
    with torch.inference_mode():
        loaded.model(torch.tensor([[1, 2, 3]]))

    from gptqmodel.models.ports.brumby.configuration_brumby import BrumbyConfig
    from gptqmodel.models.ports.brumby.modeling_brumby import BrumbyForCausalLM

    port = BrumbyForCausalLM(BrumbyConfig(
        vocab_size=24, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, chunk_size=4,
    )).eval()
    with torch.inference_mode():
        port(torch.tensor([[2, 4, 6]]))

    after_load = snapshot()

    from gptqmodel import BACKEND

    quantized = GPTQModel.load(
        sys.argv[2], backend=BACKEND.GPTQ_TORCH, device="cpu", dtype=torch.float16,
        local_files_only=True,
    )
    with torch.inference_mode():
        quantized.model(torch.tensor([[1, 2, 3]]))

    # GGUF path resolution is where Transformers' GGUF loader used to be
    # redirected; it must only resolve the file.
    from gptqmodel.utils.hf import normalize_model_id_or_path_for_hf_gguf

    normalize_model_id_or_path_for_hf_gguf(sys.argv[3], {}, api_name="probe")

    after_quantized = snapshot()

    def diff(a, b):
        changed = {}
        for label in a:
            added = sorted(set(b[label]) - set(a[label]))
            removed = sorted(set(a[label]) - set(b[label]))
            replaced = sorted(
                name for name in set(a[label]) & set(b[label]) if a[label][name] != b[label][name]
            )
            if added or removed or replaced:
                changed[label] = {"added": added, "removed": removed, "replaced": replaced}
        return changed

    print(json.dumps({
        "import": diff(before, after_import),
        "load": diff(before, after_load),
        "quantized": diff(before, after_quantized),
        "legacy": {
            "DynamicCache.from_legacy_cache": hasattr(cache_utils.DynamicCache, "from_legacy_cache"),
            "DynamicCache.to_legacy_cache": hasattr(cache_utils.DynamicCache, "to_legacy_cache"),
            "DynamicCache.__getitem__": hasattr(cache_utils.DynamicCache, "__getitem__"),
            "Cache.seen_tokens": hasattr(cache_utils.Cache, "seen_tokens"),
            "Cache.get_usable_length": hasattr(cache_utils.Cache, "get_usable_length"),
            "cache_utils.HybridCache": hasattr(cache_utils, "HybridCache"),
            "import_utils.is_torch_fx_available": hasattr(import_utils, "is_torch_fx_available"),
            "generic.OutputRecorder": hasattr(generic, "OutputRecorder"),
            "ROPE_INIT_FUNCTIONS.default": "default" in rope_utils.ROPE_INIT_FUNCTIONS,
            "NEED_SETUP_CACHE_CLASSES_MAPPING": hasattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"),
            "PreTrainedModel.is_parallelizable": hasattr(PreTrainedModel, "is_parallelizable"),
            "PreTrainedModel.get_head_mask": hasattr(PreTrainedModel, "get_head_mask"),
            "sys.modules.causal_conv1d": "causal_conv1d" in sys.modules
            and sys.modules["causal_conv1d"].__name__.startswith("gptqmodel"),
            # GPT-QModel's GGUF reader must never pose as the `gguf` package.
            "sys.modules.gguf": "gguf" in sys.modules
            and getattr(sys.modules["gguf"], "__name__", "").startswith("gptqmodel"),
            "quant_auto.gptqmodel_entries": sorted(
                name for name, cls in quant_auto.AUTO_QUANTIZER_MAPPING.items()
                if getattr(cls, "__module__", "").startswith("gptqmodel")
            ),
        },
    }))
    """
)


def _write_native_checkpoint(path: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.test_internlm_local_port import _write_tokenizer

    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    )
    LlamaForCausalLM(config).eval().save_pretrained(path)
    _write_tokenizer(path)


def test_gptqmodel_leaves_transformers_unpatched(tmp_path):
    from tests.test_checkpoint_interop_4053 import _build_gptq_v1_checkpoint

    model_dir = tmp_path / "llama"
    _write_native_checkpoint(model_dir)
    quantized_dir = tmp_path / "gptq"
    _build_gptq_v1_checkpoint(str(quantized_dir))
    gguf_file = tmp_path / "probe" / "probe.gguf"
    gguf_file.parent.mkdir()
    gguf_file.write_bytes(b"GGUF")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(model_dir), str(quantized_dir), str(gguf_file)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=500,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    for phase in ("import", "load", "quantized"):
        for label, change in list(report[phase].items()):
            change["replaced"] = [
                name for name in change["replaced"]
                if (label, name) not in _ALLOWED_INTEGRATION_PATCHES
            ]
            if not any(change.values()):
                del report[phase][label]

    assert report["import"] == {}, f"`import gptqmodel` patched Transformers: {report['import']}"
    assert report["load"] == {}, f"native/port model use patched Transformers: {report['load']}"
    assert report["quantized"] == {}, (
        f"GPTQModel.load of a quantized checkpoint patched the HF ecosystem: {report['quantized']}"
    )
    # None of these exist in Transformers 5.23; GPT-QModel must not recreate them.
    assert not any(report["legacy"].values()), report["legacy"]
