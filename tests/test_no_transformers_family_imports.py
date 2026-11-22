# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Runtime guard: supported loads never import a Transformers model family.

Measured in a fresh interpreter whose baseline is ``import transformers`` alone:
importing gptqmodel, building native models, loading a tokenizer whose
checkpoint names a Transformers model-family class, and ``GPTQModel.load`` of a
tiny quantized checkpoint must not add any
``transformers.models.<family>.(modeling|tokenization|processing|...)_*`` module.
The AST counterpart is
tests/test_model_port_policy.py::test_package_never_resolves_transformers_classes_dynamically.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]

_MEASURE = r"""
import json, re, sys
import transformers  # baseline: whatever Transformers imports by itself
FAMILY = re.compile(
    r"^transformers\.models\.(?!auto\.)[^.]+\."
    r"(modeling|tokenization|processing|image_processing|video_processing|feature_extraction)_"
)
before = set(sys.modules)

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.ports.native.auto import CONFIG_MAPPING
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config

for model_type in ("llama", "qwen3", "mistral", "gemma2"):
    config = CONFIG_MAPPING[model_type](
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
    )
    MODEL_MAP[model_type].loader.from_config(config)

declared = sys.argv[2]
tokenizer = load_hf_tokenizer(declared, model_config=load_model_config(declared))
model = GPTQModel.load(sys.argv[1], backend=BACKEND.GPTQ_TORCH, device="cpu", dtype=torch.bfloat16)
assert model.tokenizer is not None
print("RESULT " + json.dumps({
    "new": sorted(name for name in set(sys.modules) - before if FAMILY.match(name)),
    "tokenizer": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
}))
"""


def _bert_declaring_llama(directory):
    """A llama checkpoint whose tokenizer_config.json names BertTokenizer."""

    from tokenizers import Tokenizer, models, pre_tokenizers

    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 8, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "pad_token_id": 0,
    }))
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "[UNK]": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "BertTokenizer"}))
    return directory


def test_llama_checkpoint_declaring_bert_tokenizer_stays_package_owned(tmp_path):
    from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config

    path = str(_bert_declaring_llama(tmp_path / "llama"))
    tokenizer = load_hf_tokenizer(path, model_config=load_model_config(path))
    assert not type(tokenizer).__module__.startswith("transformers.models."), type(tokenizer)
    assert tokenizer("a b")["input_ids"] == [3, 4]


def test_definition_model_class_ignores_checkpoint_architecture_names():
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.loader import _definition_model_class
    from gptqmodel.models.ports.native.auto import CONFIG_MAPPING

    config = CONFIG_MAPPING["llama"](vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, num_key_value_heads=2)
    config.architectures = ["BertModel"]
    resolved = _definition_model_class(MODEL_MAP["llama"], config)
    assert resolved.__module__ == "gptqmodel.models.ports.native.llama.modeling_llama"
    assert resolved._supports_flash_attn


@pytest.mark.cpu
def test_supported_loads_import_no_transformers_model_family(tmp_path):
    from test_resave_preserves_checkpoint_dtypes_4379 import CALIBRATION, _tiny_source

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    torch.manual_seed(0)
    source = _tiny_source(tmp_path / "source")
    qcfg = QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu", offload_to_disk=False)
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.GPTQ_TORCH)
    model.quantize(CalibrationConfig(dataset=CALIBRATION, batch_size=1, min_length=1), backend=BACKEND.GPTQ_TORCH)
    model.save(str(tmp_path / "quant"))
    declared = _bert_declaring_llama(tmp_path / "bert_declared")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    completed = subprocess.run(
        [sys.executable, "-c", _MEASURE, str(tmp_path / "quant"), str(declared)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=600,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert completed.returncode == 0 and lines, completed.stdout[-2000:] + completed.stderr[-4000:]
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["new"] == [], f"supported loads imported Transformers model families: {result['new']}"
    assert not result["tokenizer"].startswith("transformers.models."), result["tokenizer"]


def test_every_vendored_definition_resolves_its_package_model_class():
    """The flash-attention probe reads the class the definition builds."""

    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.loader import _definition_model_class
    from gptqmodel.models.ports.native.auto import CONFIG_MAPPING

    unresolved = {}
    for model_type, definition in MODEL_MAP.items():
        if model_type not in CONFIG_MAPPING:
            continue
        resolved = _definition_model_class(definition, CONFIG_MAPPING[model_type]())
        if resolved is None or not resolved.__module__.startswith("gptqmodel."):
            unresolved[model_type] = resolved
    assert not unresolved, unresolved


_VISION_FREE_IMPORT_SCRIPT = r"""
import sys

for name in ("torchvision", "timm", "einops"):
    sys.modules[name] = None
import transformers.utils.import_utils as import_utils

_is_package_available = import_utils._is_package_available
import_utils._is_package_available = (
    lambda name, *args, **kwargs: (False, None)
    if str(name).split(".")[0] in {"torchvision", "timm", "einops"}
    else _is_package_available(name, *args, **kwargs)
)

from importlib import import_module

from gptqmodel.models.auto import MODEL_MAP

for model_type in ("kimi_k25", "lfm3_vl", "unlimited_ocr", "nemotronh_nano_omni_reasoning_v3"):
    loader = MODEL_MAP[model_type].loader
    for path in (loader.config_class, loader.model_class, loader.tokenizer_class):
        if path is None:
            continue
        module, _, name = path.rpartition(".")
        getattr(import_module(module), name)
print("RESULT ok")
"""


def test_config_model_and_tokenizer_import_without_vision_extras():
    """A sibling processor import must not make config/model/tokenizer imports fail."""

    completed = subprocess.run(
        [sys.executable, "-c", _VISION_FREE_IMPORT_SCRIPT],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-4000:]
    assert "RESULT ok" in completed.stdout
