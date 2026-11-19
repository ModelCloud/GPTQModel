# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every definition's tokenizer class is selected by the package.

``load_hf_tokenizer`` resolves the class through ``NativeTokenizer`` for
definitions whose loader names none, so a checkpoint's ``tokenizer_class`` can
select only a vendored or framework class. Token ids must stay what
Transformers 5.23 ``AutoTokenizer`` produces (the oracle).
"""

import glob
import json
import os

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


TEXTS = [
    "Hello world",
    "  leading spaces and\ttabs\n\nnewlines",
    "def f(x):\n    return x**2  # code",
    "数字 12345 和中文",
    "emoji 😀👍 ok",
    "a",
    "",
    "The quick brown fox jumps over the lazy dog. " * 3,
]

# Released tokenizers that may be present in the local HF cache, chosen to
# cover the AutoTokenizer selection branches: declared vendored class
# (GPT2Tokenizer), byte-level LlamaTokenizer -> backend, PreTrainedTokenizerFast,
# known-wrong hub class (openvla) and an unvendored framework fallback (xverse).
CACHED = (
    "HuggingFaceTB--SmolLM2-135M",
    "FlameF0X--TinyMoE-100m-2x8-retrained",
    "sliuau--llama3.2-1b-4bit-group128",
    "openvla--openvla-7b",
    "xverse--XVERSE-MoE-A4.2B",
    "marin-community--marin-32b-base",
    "sshleifer--tiny-gpt2",
)


def _snapshot(name):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    found = sorted(glob.glob(os.path.join(hub, f"models--{name}", "snapshots", "*", "tokenizer_config.json")))
    if not found:
        found = sorted(glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                                              f"models--{name}", "snapshots", "*", "tokenizer_config.json")))
    return os.path.dirname(found[0]) if found else None


def _write(directory, model_type, tokenizer_config, *, byte_level=False):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({
        "model_type": model_type, "vocab_size": 16, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2, "pad_token_id": 0,
    }))
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "[UNK]": 3, "a": 4, "b": 5, "Ġa": 6, "Ġb": 7}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    if byte_level:
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
    else:
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return str(directory)


def _assert_matches_oracle(path):
    from tokenicer import Tokenicer
    from transformers import AutoTokenizer

    from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config

    config = load_model_config(path)
    tokenizer = load_hf_tokenizer(path, model_config=config)
    assert not type(tokenizer).__module__.startswith("transformers.models."), type(tokenizer)
    # Same serialized-file preparation Tokenicer applies (legacy Metaspace data);
    # class selection is Transformers' own.
    oracle = AutoTokenizer.from_pretrained(path, **Tokenicer.prepare_load_kwargs(path))
    for text in TEXTS:
        assert tokenizer(text)["input_ids"] == oracle(text)["input_ids"], (path, text, type(tokenizer), type(oracle))
    return tokenizer, oracle


@pytest.mark.parametrize("name", CACHED)
def test_released_tokenizers_match_auto_tokenizer(name):
    path = _snapshot(name)
    if path is None:
        pytest.skip(f"{name} is not in the local HF cache")
    _assert_matches_oracle(path)


@pytest.mark.parametrize(
    ("model_type", "tokenizer_config", "byte_level"),
    [
        ("llama", {"tokenizer_class": "PreTrainedTokenizerFast"}, False),
        ("llama", {"tokenizer_class": "LlamaTokenizerFast"}, True),
        ("mixtral", {"tokenizer_class": "GPT2Tokenizer"}, True),
        ("qwen2", {"tokenizer_class": "Qwen2Tokenizer"}, True),
        ("falcon", {}, False),
    ],
)
def test_synthetic_tokenizers_match_auto_tokenizer(tmp_path, model_type, tokenizer_config, byte_level):
    path = _write(tmp_path / "ckpt", model_type, dict(tokenizer_config, pad_token="<pad>"), byte_level=byte_level)
    _assert_matches_oracle(path)


def test_checkpoint_cannot_select_a_transformers_model_family_tokenizer(tmp_path):
    """Transformers would build BertTokenizer; the package falls back to the backend."""

    from transformers import AutoTokenizer

    from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config

    path = _write(tmp_path / "ckpt", "llama", {"tokenizer_class": "BertTokenizer", "pad_token": "<pad>"})
    assert type(AutoTokenizer.from_pretrained(path)).__module__.startswith("transformers.models.bert.")
    tokenizer = load_hf_tokenizer(path, model_config=load_model_config(path))
    assert type(tokenizer).__name__ == "TokenizersBackend"
    assert tokenizer("a b")["input_ids"] == [4, 5]


def test_checkpoint_python_tokenizer_is_refused_without_the_remote_path(tmp_path):
    from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config

    path = _write(tmp_path / "ckpt", "llama", {
        "tokenizer_class": "EvilTok", "auto_map": {"AutoTokenizer": [None, "tok_evil.EvilTok"]},
    })
    (tmp_path / "ckpt" / "tok_evil.py").write_text("raise SystemExit('checkpoint python ran')\n")
    with pytest.raises(ValueError, match="checkpoint Python"):
        load_hf_tokenizer(path, model_config=load_model_config(path), trust_remote_code=True)
