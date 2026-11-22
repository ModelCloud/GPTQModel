# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel targets Transformers 5.23+ and never patches third-party libraries.

Importing ``gptqmodel``, loading a native model, building a package-owned port,
loading a GPT-QModel quantized checkpoint and disk-offloading a loaded layer
must leave Transformers' cache, generation, RoPE, import-utils, quantizer
registry, GGUF integration and ``PreTrainedModel`` APIs, ``torch.nn`` /
``torch.nn.init``, Accelerate (``accelerate.utils.modeling``, ``accelerate.hooks``
and its hook classes), Defuser, ``safetensors``, ``huggingface_hub`` and Triton
(when importable) exactly as they ship. Every model class a load instantiated
(vendored, port or Transformers framework class) must keep its class
``__dict__``: GPT-QModel never assigns onto classes it does not own. The only
allowed entries are the five wrappers of the model-global-state construction
guard (#5304, #5421), which keeps concurrent constructions from observing
another thread's process-global dtype/tie/``apply_patches`` window. The check
runs in a fresh interpreter so no other test can have imported ``gptqmodel``
first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

# The model-global-state construction guard is the one documented exception:
# it wraps Transformers' construction entry points so a concurrent
# ``from_pretrained``/``from_config`` cannot observe another thread's
# process-global dtype/tie/``apply_patches`` window (#5304, #5421). Every other
# patch remains forbidden.
_ALLOWED_INTEGRATION_PATCHES: set[tuple[str, str]] = {
    ("PreTrainedModel", "get_init_context"),
    ("PreTrainedModel", "_from_config"),
    ("modeling_utils", "local_torch_dtype"),
    ("modeling_utils", "apply_patches"),
    ("initialization", "no_tie_weights"),
}

_SCRIPT = textwrap.dedent(
    r"""
    import importlib
    import importlib.util
    import json
    import sys

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
    from transformers import PreTrainedModel, PreTrainedConfig, initialization, modeling_utils
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
        "modeling_utils": modeling_utils,
        "initialization": initialization,
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
    import pkgutil

    import accelerate
    import accelerate.big_modeling as accelerate_big_modeling
    import accelerate.hooks as accelerate_hooks
    import accelerate.utils as accelerate_utils
    import accelerate.utils.memory as accelerate_memory
    import accelerate.utils.modeling as accelerate_modeling
    import defuser
    import huggingface_hub
    import huggingface_hub.file_download as hf_file_download
    import huggingface_hub.hf_api as hf_api
    import safetensors
    import torch.nn as torch_nn
    import torch.nn.modules.linear as torch_linear

    WATCHED.update({
        "accelerate": accelerate,
        "accelerate.utils": accelerate_utils,
        "accelerate.utils.modeling": accelerate_modeling,
        "accelerate.utils.memory": accelerate_memory,
        "accelerate.hooks": accelerate_hooks,
        "accelerate.big_modeling": accelerate_big_modeling,
        "safetensors": safetensors,
        "huggingface_hub": huggingface_hub,
        "huggingface_hub.file_download": hf_file_download,
        "huggingface_hub.hf_api": hf_api,
        "huggingface_hub.HfApi": hf_api.HfApi,
        "torch.nn": torch_nn,
        "torch.nn.Module": torch_nn.Module,
        "torch.nn.Linear": torch_linear.Linear,
        "torch.nn.Embedding": torch_nn.Embedding,
    })
    for _name in ("ModelHook", "SequentialHook", "AlignDevicesHook", "CpuOffload", "UserCpuOffloadHook"):
        WATCHED["accelerate.hooks." + _name] = getattr(accelerate_hooks, _name)
    WATCHED["defuser"] = defuser
    for _info in pkgutil.walk_packages(defuser.__path__, "defuser."):
        WATCHED[_info.name] = importlib.import_module(_info.name)
    if hasattr(torch._C, "_CudaDeviceProperties"):
        WATCHED["torch._C._CudaDeviceProperties"] = torch._C._CudaDeviceProperties
    try:
        import triton
        import triton.runtime.autotuner as triton_autotuner
        import triton.runtime.jit as triton_jit_runtime
    except Exception:
        triton = None
    if triton is not None:
        WATCHED.update({
            "triton": triton,
            "triton.runtime.jit": triton_jit_runtime,
            "triton.runtime.jit.JITFunction": triton_jit_runtime.JITFunction,
            "triton.runtime.autotuner": triton_autotuner,
            "triton.runtime.autotuner.Autotuner": triton_autotuner.Autotuner,
            "triton.KernelInterface": triton.KernelInterface,
        })
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

    # Transformers installs `nn.Module.smart_apply` itself on the first
    # `initialize_weights`; exercise it before the baseline so only GPT-QModel
    # changes remain.
    transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=8, hidden_size=8, intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1,
    ))

    # Likewise Dynamo tags `nn.Module.__init__` the first time it runs.
    import torch._dynamo.mutation_guard as dynamo_mutation_guard

    dynamo_mutation_guard.install_generation_tagging_init()

    before = snapshot()

    import gptqmodel  # noqa: F401
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    after_import = snapshot()

    # Every class defined by a model module (Transformers, vendored natives,
    # package ports) and torch.nn: the loads below must not assign onto them.
    import gptqmodel.models.ports.brumby.modeling_brumby  # noqa: F401
    import gptqmodel.models.ports.native.llama.modeling_llama  # noqa: F401
    import gptqmodel.models.ports.native.qwen3_omni_moe.modeling_qwen3_omni_moe  # noqa: F401

    MODEL_MODULE_PREFIXES = ("transformers.", "gptqmodel.models.ports.", "torch.nn.modules.")
    # Written onto the class by Transformers' own `PreTrainedModel.__init__`
    # (`modeling_utils.py`, `_can_set_experts_implementation`), not by GPT-QModel.
    TRANSFORMERS_CLASS_CACHES = {"_can_set_experts_implementation_cached_value"}

    def class_snapshot():
        state = {}
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith(MODEL_MODULE_PREFIXES):
                continue
            for value in list(vars(module).values()):
                if isinstance(value, type) and value.__module__ == module_name:
                    state[value] = {
                        name: id(attr) for name, attr in vars(value).items()
                        if not (name.startswith("__") and name.endswith("__"))
                        and name not in TRANSFORMERS_CLASS_CACHES
                    }
        return state

    classes_before = class_snapshot()
    loaded_models = []

    model_dir = sys.argv[1]
    loaded = GPTQModel.load(
        model_dir, device="cpu", dtype=torch.float32,
        quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False),
    )
    loaded_models.append(loaded.model)
    with torch.inference_mode():
        reference = loaded.model(torch.tensor([[1, 2, 3]])).logits

    # GPT-QModel's own disk offload: package-owned hooks, Accelerate untouched.
    import tempfile

    from accelerate.hooks import SequentialHook
    from gptqmodel.utils.offload import offload_to_disk, restore_disk_offloaded_modules
    from gptqmodel.utils.offload_hooks import GPTQModelAlignDevicesHook

    layer = loaded.model.model.layers[0]
    offload_to_disk(module=layer, model=loaded.model, disk_path=tempfile.mkdtemp())
    offload_hook_types = set()
    for _module in loaded.model.modules():
        _hook = _module.__dict__.get("_hf_hook")
        for _inner in (_hook.hooks if isinstance(_hook, SequentialHook) else (_hook,)):
            if _inner is not None:
                offload_hook_types.add(type(_inner).__name__)
    with torch.inference_mode():
        offloaded_equal = torch.equal(loaded.model(torch.tensor([[1, 2, 3]])).logits, reference)
    restore_disk_offloaded_modules(loaded.model)

    # Qwen Omni's thinker getters are installed on the instance only.
    from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ
    from gptqmodel.models.ports.native.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from gptqmodel.models.ports.native.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration,
    )

    with torch.device("meta"):
        omni = Qwen3OmniMoeForConditionalGeneration(Qwen3OmniMoeConfig(
            enable_audio_output=False,
            thinker_config={
                "text_config": {
                    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
                    "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
                },
                "vision_config": {
                    "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                    "out_hidden_size": 32, "num_position_embeddings": 16,
                },
                "audio_config": {
                    "d_model": 16, "encoder_layers": 1, "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 32, "num_mel_bins": 8, "output_dim": 32,
                },
            },
        ))
    # The two loader hooks a definition gets around construction.
    Qwen3OmniMoeGPTQ.before_model_load(Qwen3OmniMoeGPTQ, "", False)
    after_conversion = getattr(Qwen3OmniMoeGPTQ, "after_defuser_conversion", None)
    if callable(after_conversion):
        after_conversion(omni)
    omni_getters = omni.get_output_embeddings() is omni.thinker.lm_head
    loaded_models.append(omni)

    from gptqmodel.models.ports.brumby.configuration_brumby import BrumbyConfig
    from gptqmodel.models.ports.brumby.modeling_brumby import BrumbyForCausalLM

    port = BrumbyForCausalLM(BrumbyConfig(
        vocab_size=24, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, chunk_size=4,
    )).eval()
    loaded_models.append(port)
    with torch.inference_mode():
        port(torch.tensor([[2, 4, 6]]))

    after_load = snapshot()

    from gptqmodel import BACKEND

    quantized = GPTQModel.load(
        sys.argv[2], backend=BACKEND.GPTQ_TORCH, device="cpu", dtype=torch.float16,
        local_files_only=True,
    )
    loaded_models.append(quantized.model)
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

    classes_after = class_snapshot()
    class_changes = {}
    for cls, names in classes_before.items():
        now = classes_after.get(cls, {})
        added = sorted(set(now) - set(names))
        removed = sorted(set(names) - set(now))
        replaced = sorted(name for name in set(now) & set(names) if now[name] != names[name])
        if added or removed or replaced:
            class_changes[f"{cls.__module__}.{cls.__qualname__}"] = {
                "added": added, "removed": removed, "replaced": replaced,
            }
    # Every model class the loads used must have been under watch: a class
    # GPT-QModel derived at runtime (Defuser's path alias) is checked through
    # the vendored class it derives from.
    unwatched = set()
    for model in loaded_models:
        for module in model.modules():
            cls = type(module)
            if not cls.__module__.startswith(MODEL_MODULE_PREFIXES):
                continue
            owner = cls if cls in classes_before else next(
                (base for base in cls.__mro__[1:] if base in classes_before), None
            )
            if owner is None:
                unwatched.add(f"{cls.__module__}.{cls.__qualname__}")

    print(json.dumps({
        "classes": class_changes,
        "unwatched_classes": sorted(unwatched),
        "offload": {
            "hook_types": sorted(offload_hook_types),
            "equal": offloaded_equal,
            "omni_instance_getters": omni_getters,
        },
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


def test_gptqmodel_leaves_third_party_libraries_unpatched(tmp_path):
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

    assert report["import"] == {}, f"`import gptqmodel` patched a third-party library: {report['import']}"
    assert report["load"] == {}, f"native/port model use or disk offload patched a third-party library: {report['load']}"
    assert report["quantized"] == {}, (
        f"GPTQModel.load of a quantized checkpoint patched the HF ecosystem: {report['quantized']}"
    )
    assert report["classes"] == {}, f"loading mutated model classes: {report['classes']}"
    assert report["unwatched_classes"] == [], report["unwatched_classes"]
    # GPT-QModel's offload uses its own hook class; Accelerate's stays stock.
    assert report["offload"] == {
        "hook_types": ["GPTQModelAlignDevicesHook"],
        "equal": True,
        "omni_instance_getters": True,
    }, report["offload"]
    # None of these exist in Transformers 5.23; GPT-QModel must not recreate them.
    assert not any(report["legacy"].values()), report["legacy"]
