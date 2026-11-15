# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Build tiny random-weight models from a checkpoint's own config and remote code.

Several model families in issue #4093 only exist as ``trust_remote_code``
architectures. Their checkpoints are far too large to quantize in a unit test,
so these helpers derive a tiny config from the checkpoint's ``config.json`` and
instantiate the checkpoint's own modeling classes, exactly the repro method the
issue used. The fixtures are local paths; callers skip with a precise reason
when a path is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from gptqmodel.utils.hf import (
    _normalize_legacy_tied_weights_keys,
    _patch_transformers_remote_code_compat,
)


# Old remote model files import optional runtime kernels at module import time.
# The fixtures disable those code paths, so the imports only need to succeed.
_REMOTE_IMPORT_PRELUDE = '''
import sys as _fixture_sys
import types as _fixture_types

def _fixture_module(name, **attrs):
    module = _fixture_sys.modules.get(name)
    if module is None:
        module = _fixture_types.ModuleType(name)
        _fixture_sys.modules[name] = module
    for key, value in attrs.items():
        if not hasattr(module, key):
            setattr(module, key, value)
    return module

_flash = _fixture_module(
    "flash_attn.flash_attn_interface",
    flash_attn_unpadded_func=lambda *args, **kwargs: None,
    flash_attn_varlen_func=lambda *args, **kwargs: None,
)
_fixture_module("flash_attn", flash_attn_interface=_flash)
_fixture_module("einops", rearrange=lambda *args, **kwargs: None)
'''


def build_wordlevel_tokenizer(vocab_size: int = 128) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "<pad>"])
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
        vocab_size=vocab_size,
    )
    tokenizer.train_from_iterator(
        ["the quick brown fox", "lorem ipsum dolor", "hello world"],
        trainer=trainer,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )


def _copy_remote_code(source: Path, target: Path) -> None:
    for path in sorted(source.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        (target / path.name).write_text(_REMOTE_IMPORT_PRELUDE + text, encoding="utf-8")


def build_tiny_remote_model(
    source: str | Path,
    target: str | Path,
    overrides: Mapping[str, Any],
    *,
    model_class_ref: Optional[str] = None,
    architecture: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """Materialize a tiny checkpoint of a remote-code family under ``target``."""

    _patch_transformers_remote_code_compat()
    source = Path(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    _copy_remote_code(source, target)

    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    for attr in ("dtype", "torch_dtype"):
        if hasattr(config, attr):
            setattr(config, attr, dtype)
    config._attn_implementation = "eager"

    class_ref = model_class_ref
    if class_ref is None:
        class_ref = config.auto_map["AutoModelForCausalLM"]
    model_cls = get_class_from_dynamic_module(class_ref, str(target))
    if architecture is not None:
        config.architectures = [architecture]
        # The auto_map entry names the module *file* and class, not the
        # importable `transformers_modules.<dir>.<file>` path.
        config.auto_map["AutoModelForCausalLM"] = f"{class_ref.split('.')[0]}.{architecture}"

    model = model_cls(config)
    model = model.to(dtype=dtype).eval()
    # Older remote classes declare `_tied_weights_keys` as a list, which
    # transformers 5's `save_pretrained` rejects. Normalize the fixture copy;
    # the loader-side fix is tracked separately.
    _normalize_legacy_tied_weights_keys(model)
    model.save_pretrained(target)
    build_wordlevel_tokenizer(int(getattr(config, "vocab_size", 128))).save_pretrained(target)
    return target
